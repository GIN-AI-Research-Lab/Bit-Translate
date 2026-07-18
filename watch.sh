#!/usr/bin/env bash
# Xem nhanh tiến độ training. Dùng: bash watch.sh   (hoặc lặp: bash watch.sh -l)
# Đích step: tự dò --max-steps từ train.py đang chạy; override: MAX=25000 bash watch.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
MAX=${MAX:-}
if [ -z "$MAX" ]; then
  for pid in $(pgrep -f 'scripts/train.py' 2>/dev/null); do
    MAX=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -A1 -x -- '--max-steps' | tail -1)
    [ -n "$MAX" ] && break
  done
fi
# fallback: đọc từ dòng "start/resume ... max_steps=N" mới nhất trong log; cuối cùng mới 25000
[ -z "$MAX" ] && MAX=$(grep -oE 'max_steps=[0-9]+' checkpoints/train.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
[ -z "$MAX" ] && MAX=25000   # run hiện tại: 292M step 11400 -> 25000

show() {
  clear 2>/dev/null
  echo "======== TIẾN ĐỘ TRAINING (Việt↔Nhật BitNet 292M) ========"
  # Mốc resume của RUN HIỆN TẠI (bỏ qua các dòng step cũ của run trước trong log).
  local base=$(grep -oE 'start/resume step=[0-9]+' checkpoints/train.log 2>/dev/null | tail -1 | grep -oE '[0-9]+')
  base=${base:-0}
  # Dòng step mới nhất + số của nó.
  local last=$(grep -E "^step [0-9]+ " checkpoints/train.log 2>/dev/null | tail -1)
  local step=$(echo "$last" | grep -oE "^step [0-9]+" | grep -oE "[0-9]+")
  step=${step:-0}

  # tiến trình train còn sống? (phân biệt "đang compile" vs "đã dừng")
  local alive=0; pgrep -f 'scripts/train.py' >/dev/null 2>&1 && alive=1

  if [ "$step" -le "$base" ]; then
    # chưa log step nào của run hiện tại -> đang compile/khởi động (KHÔNG phải đã dừng)
    if [ "$alive" = 1 ]; then
      echo "  Đang COMPILE / khởi động từ step $base (torch.compile ~2-5' lần đầu)."
      echo "  GPU nghỉ lúc này là BÌNH THƯỜNG (inductor biên dịch trên CPU)."
    else
      echo "  ⚠ Chưa có step mới của run hiện tại VÀ không thấy tiến trình train."
    fi
    echo "  Đích: step $base -> $MAX"
  else
    local sps=$(echo "$last" | grep -oE "[0-9.]+s/step" | grep -oE "[0-9.]+")
    local pct=$(awk "BEGIN{printf \"%.1f\", $step*100/$MAX}")
    local done_run=$((step - base)); local togo=$((MAX - step))
    local remain="?"
    [ -n "$sps" ] && remain=$(awk "BEGIN{printf \"%.1f\", ($MAX-$step)*$sps/3600}")
    echo "  Bước:   $step / $MAX  ($pct%)   [run này: +$done_run, còn $togo step]"
    echo "  $last" | sed 's/^/  /'
    echo "  Còn lại ước tính: ~$remain giờ"
  fi
  echo "------------------------------------------------------"
  if [ -f checkpoints/PAUSED ]; then
    echo "  Trạng thái: ⏸  ĐANG TẠM DỪNG (chạy tiếp: bash resume_292m_local.sh)"
  else
    local gu=$(timeout 8 nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)
    echo "  GPU: ${gu:-'(không đọc được nvidia-smi)'}"
    if [ "$alive" != 1 ]; then
      echo "  Trạng thái: ⛔ KHÔNG có tiến trình train — đã dừng/crash. Xem: tail checkpoints/local292.log"
    elif echo "$gu" | grep -qE "^[3-9][0-9] %|^100 %"; then
      echo "  Trạng thái: ▶  ĐANG TRAIN"
    else
      echo "  Trạng thái: ⏳ tiến trình SỐNG nhưng GPU nhàn — đang compile hoặc giữa 2 microbatch (bình thường)"
    fi
  fi
  if [ -f eval/history.tsv ]; then
    echo "------------------------------------------------------"
    echo "  Eval gần nhất (chrF, cao=tốt) — mốc trước, chưa eval lại run này:"
    tail -1 eval/history.tsv | sed 's/^/    /'
  fi
  echo "======================================================"
  echo "  Checkpoint mới nhất: $(ls -t checkpoints/*.pt 2>/dev/null | head -1 | xargs -r basename || echo 'chưa có')"
  echo "  Log: checkpoints/train.log | launcher: checkpoints/local292.log"
  echo "  (Cập nhật: $(TZ=Asia/Ho_Chi_Minh date '+%H:%M:%S' 2>/dev/null || date '+%H:%M:%S'))"
}

if [ "${1:-}" = "-l" ]; then
  while true; do show; sleep 15; done
else
  show
fi
