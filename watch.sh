#!/usr/bin/env bash
# Xem nhanh tiến độ training. Dùng: bash watch.sh   (hoặc lặp: bash watch.sh -l)
cd "$(dirname "${BASH_SOURCE[0]}")"
MAX=14000

show() {
  clear 2>/dev/null
  echo "======== TIẾN ĐỘ TRAINING (Việt↔Nhật BitNet) ========"
  local last=$(grep -E "^step " checkpoints/train.log 2>/dev/null | tail -1)
  if [ -z "$last" ]; then
    echo "  (chưa có step nào — đang khởi động, chờ ~2 phút)"
  else
    local step=$(echo "$last" | grep -oE "^step [0-9]+" | grep -oE "[0-9]+")
    local sps=$(echo "$last" | grep -oE "[0-9.]+s/step" | grep -oE "[0-9.]+")
    local pct=$(awk "BEGIN{printf \"%.1f\", $step*100/$MAX}")
    local remain=$(awk "BEGIN{printf \"%.1f\", ($MAX-$step)*$sps/3600}")
    echo "  Bước:   $step / $MAX  ($pct%)"
    echo "  $last" | sed 's/^/  /'
    echo "  Còn lại ước tính: ~$remain giờ"
  fi
  echo "------------------------------------------------------"
  # trạng thái
  if [ -f checkpoints/PAUSED ]; then echo "  Trạng thái: ⏸  ĐANG TẠM DỪNG (chạy tiếp: bash resume_training.sh)"
  else
    local gu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)
    echo "  GPU: $gu"
    echo "$gu" | grep -qE "^[5-9][0-9] %|^100 %" && echo "  Trạng thái: ▶  ĐANG TRAIN" || echo "  Trạng thái: ⚠  GPU rảnh — có thể đã dừng"
  fi
  # chất lượng (chrF) nếu đã eval
  if [ -f eval/history.tsv ]; then
    echo "------------------------------------------------------"
    echo "  Chất lượng gần nhất (chrF, cao=tốt):"
    tail -1 eval/history.tsv | sed 's/^/    /'
  fi
  echo "======================================================"
  echo "  Checkpoint gần nhất: $(ls -t checkpoints/*.pt 2>/dev/null | head -1 | xargs -r basename || echo 'chưa có (mốc đầu step 100)')"
  echo "  (Cập nhật: $(TZ=Asia/Ho_Chi_Minh date '+%H:%M:%S' 2>/dev/null || date '+%H:%M:%S'))"
}

if [ "${1:-}" = "-l" ]; then
  while true; do show; sleep 15; done
else
  show
fi
