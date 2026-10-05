#!/bin/sh
# Run each variant in turn (one process at a time, so CPU timings are comparable)
cd "$(dirname "$0")"
for v in "$@"; do
  python3 run_eval.py "$v"
done
