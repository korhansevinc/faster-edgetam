"""Repro: the video predictor crashes when tracking two objects (upstream issue #15).

Run from a checkout root. Upstream main fails with "view size is not compatible ..."
in PerceiverResampler.forward_2d; with the fix it prints "OK: no crash".
"""

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_video_predictor(
    "configs/edgetam.yaml", "checkpoints/edgetam.pt", device=device
)
with torch.inference_mode():
    state = predictor.init_state(video_path="notebooks/videos/bedroom")
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=2,
        points=np.array([[200, 300], [275, 175]], np.float32),
        labels=np.array([1, 0], np.int32),
    )
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=3,
        points=np.array([[400, 150]], np.float32),
        labels=np.array([1], np.int32),
    )
    for frame_idx, obj_ids, _ in predictor.propagate_in_video(
        state, max_frame_num_to_track=3
    ):
        print("frame", frame_idx, "objects", obj_ids)
print("OK: no crash")
