#!/usr/bin/env bash
# Robust multi-day training launcher: runs train.py, auto-resumes from
# checkpoints/last.pt if it dies (transient CUDA/WSL hiccups over days), stops
# when the loop reports it finished all steps.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
# IMPORTANT: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — it caused
# intermittent "CUDA driver error: device not ready" crashes under sustained
# async training load in this WSL2 setup. The default allocator is stable.

# --compile: measured ~2.3x speedup (0.18 vs 0.32s/microbatch) with dynamic=True,
#   no OOM at max_tokens 4096. Needs linux-libc-dev installed (Triton builds CUDA
#   utils with gcc). One-time ~1-3 min compile warmup on each (re)start.
# log-every 5: first log appears in ~1 min so it doesn't look "stuck".
ARGS="--max-tokens 4096 --grad-accum 32 --max-steps 14000 --save-every 100 --milestone-every 2000 --log-every 5 --compile"

rm -f checkpoints/PAUSED   # a fresh launch clears any previous pause flag

for i in $(seq 1 200); do
  echo ">>> launch attempt $i $(date -u +%H:%M:%S)" >> checkpoints/train.log
  python3 scripts/train.py $ARGS
  if [ -f checkpoints/PAUSED ]; then
    echo ">>> PAUSED by user, launcher stopping (resume with run_training.sh)" >> checkpoints/train.log
    break
  fi
  if grep -q "training loop exited" checkpoints/train.log; then
    echo ">>> TRAINING COMPLETE" >> checkpoints/train.log
    break
  fi
  echo ">>> crashed (attempt $i), resuming in 15s" >> checkpoints/train.log
  sleep 15
done
