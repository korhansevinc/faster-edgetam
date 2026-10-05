"""Track the selected objects in every clip with one EdgeTAM input-resolution variant.

Usage: python run_eval.py VARIANT [CLIP ...]
Saves results/<VARIANT>/<clip>.npz with binary masks (frames x objects x H x W)
and per-frame wall-clock times.
"""

import json
import os
import sys
import time

import numpy as np
import torch

import offline_timm  # noqa: F401
import sam2
from sam2.build_sam import build_sam2_video_predictor

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(ROOT, "clips")
# EdgeTAM checkpoint: $EDGETAM_CKPT, else checkpoints/edgetam.pt next to the sam2 package
CKPT = os.environ.get("EDGETAM_CKPT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(sam2.__file__))),
    "checkpoints",
    "edgetam.pt",
)

# input size -> config
VARIANTS = {
    "1024": "configs/edgetam.yaml",
    "768": "configs/edgetam_768.yaml",
    "512": "configs/edgetam_512.yaml",
}


def build(variant):
    return build_sam2_video_predictor(VARIANTS[variant], CKPT, device="cpu")


def track(predictor, clip, objects):
    state = predictor.init_state(video_path=os.path.join(CLIPS, clip))
    for obj in objects:
        x, y, w, h = obj["bbox_xywh"]
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj["obj_id"],
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
        )
    T, H, W = state["num_frames"], state["video_height"], state["video_width"]
    masks = np.zeros((T, len(objects), H, W), dtype=bool)
    times = np.zeros(T)
    t0 = time.perf_counter()
    for frame_idx, obj_ids, logits in predictor.propagate_in_video(state):
        t1 = time.perf_counter()
        times[frame_idx] = t1 - t0
        masks[frame_idx] = (logits[:, 0] > 0.0).cpu().numpy()
        t0 = t1
    return masks, times


def main():
    variant, clips = sys.argv[1], sys.argv[2:]
    prompts = json.load(open(os.path.join(ROOT, "prompts.json")))
    clips = clips or sorted(prompts)
    torch.set_num_threads(int(os.environ.get("THREADS", "2")))
    out_dir = os.path.join(ROOT, "results", variant)
    os.makedirs(out_dir, exist_ok=True)
    predictor = build(variant)
    for clip in clips:
        with torch.inference_mode():
            masks, times = track(predictor, clip, prompts[clip])
        np.savez_compressed(
            os.path.join(out_dir, f"{clip}.npz"), masks=masks, times=times
        )
        print(
            f"{variant} {clip}: {1000 * times[1:].mean():.0f} ms/frame "
            f"({len(prompts[clip])} objects)",
            flush=True,
        )


if __name__ == "__main__":
    main()
