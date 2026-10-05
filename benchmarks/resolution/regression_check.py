"""Check that a change keeps EdgeTAM's 1024 outputs bit-identical.

Tracks one object for 12 frames of notebooks/videos/bedroom (box and click prompt, CPU).
Run it from the root of each checkout, then compare:

    PYTHONPATH=. python /path/to/regression_check.py run /tmp/main.npy    # upstream main
    PYTHONPATH=. python /path/to/regression_check.py run /tmp/branch.npy  # your branch
    python /path/to/regression_check.py compare /tmp/main.npy /tmp/branch.npy
"""

import os
import sys

import numpy as np


def run(out_path):
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import offline_timm  # noqa: F401  (skip timm's ImageNet download)
    import sam2
    from sam2.build_sam import build_sam2_video_predictor

    print("testing sam2 from", os.path.dirname(os.path.abspath(sam2.__file__)))
    predictor = build_sam2_video_predictor(
        "configs/edgetam.yaml", "checkpoints/edgetam.pt", device="cpu"
    )
    prompts = [
        dict(box=np.array([300, 0, 500, 400], np.float32)),
        dict(
            points=np.array([[210, 350]], np.float32),
            labels=np.array([1], np.int32),
        ),
    ]
    runs = []
    with torch.inference_mode():
        for prompt in prompts:
            state = predictor.init_state(video_path="notebooks/videos/bedroom")
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, **prompt)
            logits = [
                lg.numpy()
                for _, _, lg in predictor.propagate_in_video(
                    state, max_frame_num_to_track=11
                )
            ]
            runs.append(np.stack(logits))
    np.save(out_path, np.stack(runs))
    print("saved", out_path, np.stack(runs).shape)


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    if a.shape != b.shape:
        print(f"different shapes: {a.shape} vs {b.shape}")
        sys.exit(1)
    same = np.array_equal(a, b)
    print(f"identical: {same} (max abs diff {float(np.abs(a - b).max())})")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
