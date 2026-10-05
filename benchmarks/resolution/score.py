"""Compare each variant's masks with the 1024 baseline.

Usage: python score.py  -> prints tables and writes results/summary.json
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
REF = "1024"
ORDER = ["768", "512"]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def size_bucket(area_frac):
    if area_frac < 0.01:
        return "small (<1% of frame)"
    if area_frac < 0.05:
        return "medium (1-5%)"
    return "large (>5%)"


def main():
    prompts = json.load(open(os.path.join(ROOT, "prompts.json")))
    variants = [v for v in ORDER if os.path.isdir(os.path.join(RES, v))]
    rows, speed = [], {v: [] for v in variants}
    for clip, objects in sorted(prompts.items()):
        ref_npz = np.load(os.path.join(RES, REF, f"{clip}.npz"))
        ref, ref_t = ref_npz["masks"], ref_npz["times"]
        for v in variants:
            path = os.path.join(RES, v, f"{clip}.npz")
            if not os.path.exists(path):
                continue
            npz = np.load(path)
            masks, times = npz["masks"], npz["times"]
            speed[v].append(ref_t[1:].mean() / times[1:].mean())
            for j, obj in enumerate(objects):
                ious = np.array([iou(ref[t, j], masks[t, j]) for t in range(len(ref))])
                rows.append(
                    {
                        "clip": clip,
                        "obj_id": obj["obj_id"],
                        "label": obj.get("label", ""),
                        # object size = 1024 baseline mask on the prompted frame
                        "area_frac": float(ref[0, j].mean()),
                        "variant": v,
                        "iou_frame0": float(ious[0]),
                        "miou": float(ious[1:].mean()),
                        "iou_p10": float(np.percentile(ious[1:], 10)),
                    }
                )

    print(
        f"{len(prompts)} clips, {sum(len(o) for o in prompts.values())} objects, "
        f"IoU against the 1024 baseline (frames 1..N, frame 0 = prompted frame)\n"
    )
    header = f"{'variant':10s} {'mean IoU':>9s} {'median':>7s} {'>=0.9':>6s} {'>=0.8':>6s} {'10th pct frame':>15s} {'frame0':>7s} {'speedup':>8s}"
    print(header)
    summary = {}
    for v in variants:
        r = [x for x in rows if x["variant"] == v]
        miou = np.array([x["miou"] for x in r])
        s = {
            "mean_iou": float(miou.mean()),
            "median_iou": float(np.median(miou)),
            "frac_obj_ge_0.9": float((miou >= 0.9).mean()),
            "frac_obj_ge_0.8": float((miou >= 0.8).mean()),
            "mean_p10_frame_iou": float(np.mean([x["iou_p10"] for x in r])),
            "mean_iou_frame0": float(np.mean([x["iou_frame0"] for x in r])),
            "cpu_speedup_geomean": float(np.exp(np.mean(np.log(speed[v])))),
            "by_size": {},
        }
        for b in sorted({size_bucket(x["area_frac"]) for x in r}):
            rb = [x["miou"] for x in r if size_bucket(x["area_frac"]) == b]
            s["by_size"][b] = {"n": len(rb), "mean_iou": float(np.mean(rb))}
        summary[v] = s
        print(
            f"{v:10s} {s['mean_iou']:9.3f} {s['median_iou']:7.3f} {s['frac_obj_ge_0.9']:6.0%} "
            f"{s['frac_obj_ge_0.8']:6.0%} {s['mean_p10_frame_iou']:15.3f} "
            f"{s['mean_iou_frame0']:7.3f} {s['cpu_speedup_geomean']:7.2f}x"
        )

    print("\nmean IoU by object size")
    for v in variants:
        print(
            f"  {v:10s} "
            + "  ".join(
                f"{b}: {d['mean_iou']:.3f} (n={d['n']})"
                for b, d in summary[v]["by_size"].items()
            )
        )

    print("\nper object (mean IoU)")
    for clip, objects in sorted(prompts.items()):
        for obj in objects:
            r = [x for x in rows if x["clip"] == clip and x["obj_id"] == obj["obj_id"]]
            if not r:
                continue
            vals = {x["variant"]: x["miou"] for x in r}
            print(
                f"  {clip:16s} {obj.get('label', ''):14s} {100 * r[0]['area_frac']:5.1f}%  "
                + "  ".join(f"{v}={vals[v]:.3f}" for v in variants if v in vals)
            )

    json.dump(
        {"summary": summary, "per_object": rows},
        open(os.path.join(RES, "summary.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
