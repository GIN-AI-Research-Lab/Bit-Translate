#!/usr/bin/env bash
# cloud/autostart.sh — tự khởi chạy lại training sau khi VM/container khởi động lại
# hoặc sau khi provider stop->start (với guest, việc này trông giống một lần boot).
# Gọi bởi @reboot cron (hoặc hook .bashrc). AN TOÀN khi gọi trùng: tự thoát nếu
# training đã chạy, hoặc nếu bạn đang cố ý PAUSED.
set -u

PROJ="${VIJA_PROJ:-$HOME/Train-model-translate}"
cd "$PROJ" 2>/dev/null || { echo "FATAL: không thấy $PROJ"; exit 1; }
mkdir -p checkpoints
exec >>"$PROJ/checkpoints/autostart.log" 2>&1
echo "===== $(date '+%F %T') autostart chạy ====="

# 1) Tôn trọng ý muốn dừng: có PAUSED -> KHÔNG tự chạy lại.
if [ -f checkpoints/PAUSED ]; then
  echo "-> có checkpoints/PAUSED, bỏ qua (đang cố ý dừng)."; exit 0
fi

# 2) Chống chạy trùng: đã có launcher hoặc train.py thì thôi.
if pgrep -f "scripts/train.py" >/dev/null || pgrep -f "cloud/run_cloud.sh" >/dev/null; then
  echo "-> training đã chạy, bỏ qua."; exit 0
fi

# 3) Chờ GPU driver sẵn sàng sau boot (có thể chậm vài chục giây).
export PATH=/opt/conda/bin:$PATH
ok=0
for i in $(seq 1 30); do
  if nvidia-smi >/dev/null 2>&1; then ok=1; echo "-> GPU sẵn sàng (lần $i)."; break; fi
  echo "-> chờ GPU $i/30..."; sleep 10
done
[ "$ok" = 1 ] || echo "-> GPU chưa sẵn sàng sau 5 phút, vẫn thử chạy."

# 4) Chạy launcher (tự có vòng restart 500 lần + resume từ checkpoints/last.pt).
echo "-> khởi chạy cloud/run_cloud.sh"
nohup bash cloud/run_cloud.sh >>checkpoints/cloud.log 2>&1 &
echo "-> đã chạy nền, PID=$!"
