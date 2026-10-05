"""Time per tracked frame (one object) for one config on one video.

Runs from the root of any checkout, so upstream and this fork can be timed on the same
video (see latency_vs_upstream.sh). Prints one JSON line.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_timm  # noqa: E402,F401
import sam2  # noqa: E402
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


def run_once(predictor, video_dir, box, cuda):
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=cuda):
        state = predictor.init_state(video_path=video_dir)
        predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1, box=np.array(box, np.float32)
        )
        times = []
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, _ in enumerate(predictor.propagate_in_video(state)):
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= 8:  # memory bank is full
                times.append(t1 - t0)
            t0 = t1
    return 1000 * float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config")
    parser.add_argument("video_dir")
    parser.add_argument("--box", nargs=4, type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0, help="0 = torch default")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cuda = args.device.startswith("cuda")
    root = os.path.dirname(os.path.dirname(os.path.abspath(sam2.__file__)))
    ckpt = os.path.join(root, "checkpoints", "edgetam.pt")
    predictor = build_sam2_video_predictor(args.config, ckpt, device=args.device)
    runs = [
        run_once(predictor, args.video_dir, args.box, cuda) for _ in range(args.repeats)
    ]
    device = torch.cuda.get_device_name() if cuda else f"cpu x{torch.get_num_threads()}"
    result = {
        "checkout": root,
        "config": args.config,
        "device": device,
        "ms_per_frame": float(np.median(runs[1:] or runs)),  # first run warms up
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
