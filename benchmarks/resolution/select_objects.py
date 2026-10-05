"""Propose objects to track on frame 0 of each clip (1024 baseline model).

Writes candidates.json and one contact sheet per clip so prompts can be picked by eye.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

import offline_timm  # noqa: F401  (must come before building the model)
import sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(ROOT, "clips")
OUT = os.path.join(ROOT, "candidates")
# EdgeTAM checkpoint: $EDGETAM_CKPT, else checkpoints/edgetam.pt next to the sam2 package
CKPT = os.environ.get("EDGETAM_CKPT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(sam2.__file__))),
    "checkpoints",
    "edgetam.pt",
)

torch.set_num_threads(2)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def main(clips):
    os.makedirs(OUT, exist_ok=True)
    model = build_sam2("configs/edgetam.yaml", CKPT, device="cpu")
    amg = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=24,
        points_per_batch=64,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
    )
    all_cands = {}
    if os.path.exists(os.path.join(OUT, "candidates.json")):
        all_cands = json.load(open(os.path.join(OUT, "candidates.json")))
    for clip in clips:
        img = cv2.imread(os.path.join(CLIPS, clip, "00000.jpg"))[:, :, ::-1].copy()
        H, W = img.shape[:2]
        with torch.inference_mode():
            masks = amg.generate(img)
        masks = [m for m in masks if 0.002 <= m["area"] / (H * W) <= 0.35]
        masks.sort(key=lambda m: -m["predicted_iou"] * m["stability_score"])
        kept = []
        for m in masks:
            if all(mask_iou(m["segmentation"], k["segmentation"]) < 0.5 for k in kept):
                kept.append(m)
            if len(kept) == 14:
                break
        # contact sheet: one tile per candidate
        tiles = []
        for i, m in enumerate(kept):
            tile = img[:, :, ::-1].copy()
            overlay = tile.copy()
            overlay[m["segmentation"]] = (0, 0, 255)
            tile = cv2.addWeighted(overlay, 0.5, tile, 0.5, 0)
            x, y, w, h = [int(v) for v in m["bbox"]]
            cv2.rectangle(tile, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                tile,
                f"#{i} {100*m['area']/(H*W):.1f}%",
                (8, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3,
            )
            tiles.append(cv2.resize(tile, (256, 240)))
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r : r + 5]) for r in range(0, len(tiles), 5)]
        cv2.imwrite(os.path.join(OUT, f"{clip}.jpg"), np.vstack(rows))
        all_cands[clip] = [
            {
                "id": i,
                "bbox_xywh": [float(v) for v in m["bbox"]],
                "area_frac": float(m["area"] / (H * W)),
                "pred_iou": float(m["predicted_iou"]),
            }
            for i, m in enumerate(kept)
        ]
        np.savez_compressed(
            os.path.join(OUT, f"{clip}_masks.npz"),
            masks=np.stack([m["segmentation"] for m in kept]),
        )
        print(clip, len(kept), "candidates", flush=True)
        json.dump(all_cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(CLIPS)))
