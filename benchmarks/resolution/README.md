# Input-resolution benchmark

Scripts and raw results behind the 512/768 numbers in the top-level README and the upstream PRs.
Run from this folder with the repo root on `PYTHONPATH`. `make_clips.sh` needs `ffmpeg`.

```sh
cd benchmarks/resolution
export PYTHONPATH=../..
sh make_clips.sh              # 10 clips at 512x480 and one at 512x512 from the repo's sample videos
./run_all.sh 1024 768 512     # track the 27 boxes in prompts.json -> results/<size>/*.npz
python score.py               # IoU of the 768 / 512 masks against 1024
python flops_latency.py       # GFLOPs per tracked frame (one object)
UPSTREAM=/path/to/upstream/EdgeTAM sh latency_vs_upstream.sh cpu 2   # or: cuda
DAVIS=/path/to/DAVIS EVAL=/path/to/davis2017-evaluation sh davis_eval.sh 1024 768 512
```

| File | Purpose |
|---|---|
| `prompts.json` | The 27 objects (a box on frame 0 of each clip) |
| `select_objects.py` | How the candidate boxes were proposed (automatic mask generator on frame 0) |
| `run_eval.py`, `run_all.sh` | Track all objects with one config and save masks and timings |
| `score.py` | IoU against 1024: mean, median, by object size and per object |
| `flops_latency.py` | FLOPs per module for one tracked frame |
| `video_latency.py`, `latency_vs_upstream.sh` | Time per frame on the 512x512 clip, upstream EdgeTAM vs this fork |
| `davis_eval.sh` | DAVIS 2017 val J&F with `tools/vos_inference.py` and the [official evaluator](https://github.com/davisvideochallenge/davis2017-evaluation) |
| `regression_check.py` | Checks that a change keeps the 1024 outputs bit-identical |
| `repro_multi_object_crash.py` | Reproduces the multi-object crash fixed in this fork |
| `offline_timm.py` | Skips timm's ImageNet download (the checkpoint overwrites those weights) |
| `results/` | Raw numbers: i7-12800HX, RTX A4500 Laptop GPU, PyTorch 2.12 (times in `video_latency.json`) |

DAVIS J&F measures accuracy against ground truth. IoU against 1024 only measures how close the masks stay to the default model.
