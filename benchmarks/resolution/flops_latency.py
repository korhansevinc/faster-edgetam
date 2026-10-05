"""Per-frame compute of EdgeTAM video tracking at different input sizes.

Counts FLOPs for one steady-state tracking frame (1 object, full memory bank) split by
module, and measures CPU wall-clock per tracked frame.

Usage: python flops_latency.py [CLIP]
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import offline_timm  # noqa: F401
from run_eval import build, CLIPS, ROOT

torch.set_num_threads(int(os.environ.get("THREADS", "2")))
SIZES = ["1024", "768", "512"]
FLOP_FRAME = 25  # memory bank (7 frames) is full well before this


def sdpa_math(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw
):
    # same result as F.scaled_dot_product_attention, but made of matmuls that
    # FlopCounterMode is guaranteed to count on CPU
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v


def top_level_breakdown(counter):
    # keys are module paths such as "MemoryAttention" or "ImageEncoder.trunk";
    # the one-part keys are the top-level modules the predictor calls directly
    return {
        name: sum(ops.values())
        for name, ops in counter.get_flop_counts().items()
        if "." not in name and name != "Global"
    }


def run(variant, clip, obj):
    predictor = build(variant)
    # FlopCounterMode's module tracker trips over parameter views made under
    # inference_mode; frozen parameters avoid that and change nothing else
    for p in predictor.parameters():
        p.requires_grad_(False)
    s = predictor.image_size
    with torch.inference_mode():
        # image encoder alone
        with FlopCounterMode(display=False) as fc:
            predictor.forward_image(torch.randn(1, 3, s, s))
        enc_flops = fc.get_total_flops()

        state = predictor.init_state(video_path=os.path.join(CLIPS, clip))
        x, y, w, h = obj["bbox_xywh"]
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
        )
        times, frame_flops, breakdown = [], None, None
        gen = predictor.propagate_in_video(state)
        for i in range(state["num_frames"]):
            if i == FLOP_FRAME:
                orig = F.scaled_dot_product_attention
                F.scaled_dot_product_attention = sdpa_math
                try:
                    with FlopCounterMode(display=False) as fc:
                        next(gen)
                finally:
                    F.scaled_dot_product_attention = orig
                frame_flops = fc.get_total_flops()
                breakdown = top_level_breakdown(fc)
                continue
            t0 = time.perf_counter()
            next(gen)
            if i >= 5:
                times.append(time.perf_counter() - t0)
    return {
        "image_size": s,
        "encoder_gflops": enc_flops / 1e9,
        "frame_gflops": frame_flops / 1e9,
        "breakdown_gflops": {k: v / 1e9 for k, v in breakdown.items()},
        "cpu_ms_per_frame": 1000 * float(np.mean(times)),
        "cpu_ms_per_frame_std": 1000 * float(np.std(times)),
    }


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "19_cyclist"
    prompts = json.load(open(os.path.join(ROOT, "prompts.json")))
    obj = prompts[clip][0]
    results = {}
    for v in SIZES:
        results[v] = run(v, clip, obj)
        r = results[v]
        print(
            f"{v:>5s}: encoder {r['encoder_gflops']:6.1f} GFLOPs | full tracking frame "
            f"{r['frame_gflops']:6.1f} GFLOPs | CPU {r['cpu_ms_per_frame']:6.0f} ms/frame",
            flush=True,
        )
        print(
            "       "
            + ", ".join(
                f"{k} {g:.2f}"
                for k, g in sorted(r["breakdown_gflops"].items(), key=lambda kv: -kv[1])
            )
        )
    base = results["1024"]
    for v in SIZES[1:]:
        r = results[v]
        print(
            f"{v}: {base['frame_gflops'] / r['frame_gflops']:.2f}x fewer FLOPs, "
            f"{base['cpu_ms_per_frame'] / r['cpu_ms_per_frame']:.2f}x faster on this CPU"
        )
    json.dump(
        results,
        open(os.path.join(ROOT, "results", "flops_latency.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
