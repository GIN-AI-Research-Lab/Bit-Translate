#!/usr/bin/env bash
# Chạy tiếp training từ checkpoint gần nhất (sau khi pause / tắt máy / buổi tối
# hôm sau). An toàn để gọi nhiều lần; tự resume từ checkpoints/last.pt.
cd "$(dirname "${BASH_SOURCE[0]}")"

if pgrep -f "python3 scripts/train.py" >/dev/null; then
  echo "Training đang chạy sẵn rồi — không cần resume."
  grep -E "^step " checkpoints/train.log | tail -1
  exit 0
fi

rm -f checkpoints/PAUSED
nohup bash scripts/run_training.sh > checkpoints/launcher.log 2>&1 &
echo "Đã chạy tiếp training (PID $!). Resume từ:"
ls -la checkpoints/last.pt 2>/dev/null || echo "  (chưa có last.pt -> bắt đầu từ đầu)"
sleep 8
grep -E "^step |resumed|launch attempt" checkpoints/train.log | tail -3
