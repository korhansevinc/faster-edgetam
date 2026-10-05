#!/bin/sh
# DAVIS 2017 val J&F per input size, with tools/vos_inference.py and the official evaluator.
# Usage: DAVIS=/path/to/DAVIS EVAL=/path/to/davis2017-evaluation sh davis_eval.sh 1024 768 512
set -e
cd "$(dirname "$0")/../.."
OUT=$(pwd)/benchmarks/resolution/davis
for s in "$@"; do
  cfg=configs/edgetam_$s.yaml
  [ "$s" = 1024 ] && cfg=configs/edgetam.yaml
  python tools/vos_inference.py --sam2_cfg $cfg --sam2_checkpoint checkpoints/edgetam.pt \
    --base_video_dir "$DAVIS/JPEGImages/480p" --input_mask_dir "$DAVIS/Annotations/480p" \
    --video_list_file "$DAVIS/ImageSets/2017/val.txt" --output_mask_dir "$OUT/$s"
  (cd "$EVAL" && python evaluation_method.py --task semi-supervised \
    --davis_path "$DAVIS" --results_path "$OUT/$s")
done
