#!/usr/bin/env bash
# Tạm dừng training để chơi game / giải phóng GPU. Lưu checkpoint ngay (không
# mất tiến độ) rồi thoát. Chạy tiếp sau bằng: bash resume_training.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

touch checkpoints/PAUSED                       # báo launcher đừng restart
pkill -f "run_training.sh" 2>/dev/null         # dừng vòng auto-restart trước
sleep 1
# gửi SIGTERM cho python -> nó lưu last.pt và thoát sạch
pkill -TERM -f "python3 scripts/train.py" 2>/dev/null

echo "Đang chờ training lưu checkpoint & thoát..."
for i in $(seq 1 60); do
  pgrep -f "python3 scripts/train.py" >/dev/null || break
  sleep 2
done

if pgrep -f "python3 scripts/train.py" >/dev/null; then
  echo "Chưa thoát sau 120s, buộc dừng."; pkill -9 -f "python3 scripts/train.py"
fi
sleep 2
echo "=== ĐÃ TẠM DỪNG ==="
tail -2 checkpoints/train.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/GPU giờ: /'
echo "GPU đã rảnh — chơi game thoải mái. Chạy tiếp: bash resume_training.sh"
