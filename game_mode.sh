#!/usr/bin/env bash
# CHẾ ĐỘ GAME: chạy tiếp training nhưng chỉ chiếm ~nửa GPU + ~3.6GB VRAM,
# chừa phần còn lại cho game (LoL). Chậm hơn ~3-4x nhưng chơi được song song.
# Dùng khi muốn vừa train vừa chơi. Muốn full tốc: bash resume_training.sh
cd "$(dirname "${BASH_SOURCE[0]}")"

# dừng bản đang chạy (full-speed) nếu có, lưu checkpoint
if pgrep -f "python3 scripts/train.py" >/dev/null; then
  echo "Dừng bản full-speed để chuyển sang game mode (lưu checkpoint)..."
  bash pause_training.sh >/dev/null 2>&1
fi

source .venv/bin/activate
rm -f checkpoints/PAUSED
# mem-frac 0.45 (~3.6GB, chừa ~4.4GB cho game) + throttle 1.0 (~50% duty) + batch nhỏ
GM_ARGS="--max-tokens 1024 --grad-accum 32 --max-steps 14000 --save-every 100 --milestone-every 2000 --log-every 20 --mem-frac 0.45 --throttle 1.0"
nohup bash -c "source .venv/bin/activate; rm -f checkpoints/PAUSED; for i in \$(seq 1 200); do echo '>>> game-mode launch '\$i >> checkpoints/train.log; python3 scripts/train.py $GM_ARGS; [ -f checkpoints/PAUSED ] && { echo '>>> PAUSED' >> checkpoints/train.log; break; }; grep -q 'training loop exited' checkpoints/train.log && { echo '>>> TRAINING COMPLETE' >> checkpoints/train.log; break; }; sleep 15; done" > checkpoints/launcher.log 2>&1 &
echo "Đã chạy training ở CHẾ ĐỘ GAME (PID $!): ~3.6GB VRAM, ~50% GPU."
echo "Chơi LoL để setting Trung bình/Thấp cho mượt. Xong game, chạy: bash resume_training.sh (full tốc)."
sleep 8
grep -E "^step |resumed|game-mode|VRAM capped" checkpoints/train.log | tail -3
