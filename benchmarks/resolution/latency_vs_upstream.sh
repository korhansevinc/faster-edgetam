#!/bin/sh
# Time per frame on the 512x512 clip: upstream EdgeTAM (1024) vs this fork (1024 / 768 / 512).
# Usage: UPSTREAM=/path/to/upstream/EdgeTAM sh latency_vs_upstream.sh [cpu|cuda] [threads]
# A warm-up run comes first, since a GPU that was idle is slow for the first run.
cd "$(dirname "$0")"
HERE=$(pwd)
FORK=$(cd ../.. && pwd)
DEVICE=${1:-cpu}
THREADS=${2:-0}
run() {
  (cd "$1" && PYTHONPATH=. python "$HERE/video_latency.py" "$2" "$HERE/clips/bedroom_512" \
    --box 160 0 267 379 --device "$DEVICE" --threads "$THREADS")
}
run "$UPSTREAM" configs/edgetam.yaml > /dev/null
run "$UPSTREAM" configs/edgetam.yaml
for cfg in edgetam edgetam_768 edgetam_512; do
  run "$FORK" configs/$cfg.yaml
done
