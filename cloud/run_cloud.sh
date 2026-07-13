#!/usr/bin/env bash
# Chạy training TIẾP TỤC từ checkpoint (checkpoints/last.pt) trên A4000 16GB.
# Tự resume đúng step đang có; tự khởi động lại nếu crash. Chạy nền:
#   nohup bash cloud/run_cloud.sh > checkpoints/cloud.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Conda python (torch+cuda) — shell nền/detached KHÔNG source profile nên phải tự thêm PATH.
export PATH=/opt/conda/bin:$PATH

# Linux thật + A4000 16GB: bật lại tối ưu mà WSL2 ở nhà chặn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Effective batch = max_tokens * grad_accum = 8192*16 = 131072, GIỐNG HỆT cấu hình
# ở nhà (4096*32) -> train tiếp mượt, cùng LR schedule & dynamics, chỉ nhanh hơn nhờ HW.
# Nếu báo OUT OF MEMORY: hạ --max-tokens 6144 (grad-accum 22) hoặc 4096 (grad-accum 32).
ARGS="--max-tokens 8192 --grad-accum 16 --max-steps 14000 --save-every 100 --milestone-every 2000 --log-every 5 --compile"

mkdir -p checkpoints
# Có last.pt -> train tiếp; không có -> train từ đầu (arch mới). Cả hai đều hợp lệ.
if [ -f checkpoints/last.pt ]; then echo "resume từ checkpoints/last.pt"; else echo "train MỚI từ đầu"; fi
rm -f checkpoints/PAUSED

for i in $(seq 1 500); do
  echo ">>> launch $i $(date -u +%H:%M:%S)" >> checkpoints/train.log
  python3 scripts/train.py $ARGS
  if [ -f checkpoints/PAUSED ]; then echo ">>> PAUSED, dừng launcher" >> checkpoints/train.log; break; fi
  if grep -q "training loop exited" checkpoints/train.log; then echo ">>> TRAINING COMPLETE" >> checkpoints/train.log; break; fi
  echo ">>> crashed lần $i, resume sau 10s" >> checkpoints/train.log
  sleep 10
done
