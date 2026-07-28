#!/bin/bash
# Giữ lại các mốc NEO để dựng ĐƯỜNG CONG KỸ NĂNG THEO STEP (step % EVERY == 0).
#
# Vì sao cần folder riêng: backup_ckpt_loop.sh chạy với KEEP=n chỉ giữ n mốc mới nhất
# trong checkpoints_vX (đó là bản backup lăn, phòng ngắt bất ngờ). Muốn biết "bao nhiêu
# step là đủ để học xong một kỹ năng" thì phải chấm CÙNG MỘT BENCH trên các mốc RẢI ĐỀU
# cả run — mà những mốc đó nằm ngoài cửa sổ KEEP nên sẽ bị tỉa.
#
# Vì sao không đọc thẳng từ volume Modal: volume giữ đủ mọi mốc, NHƯNG nếu hết credit và
# account bị khoá thì mất luôn quyền đọc. Đây là bản sao offline.
#
#   nohup bash scripts/curve_anchor_loop.sh checkpoints_v6 curve_v6 2000 > logs/curve_v6.log 2>&1 &
#     $1 = thư mục backup lăn (nguồn)   $2 = thư mục neo (đích)   $3 = bước neo
set -u
cd "$(dirname "$0")/.."
SRC="/d/Bit-Translate-data/${1:-checkpoints_v6}"
DST="/d/Bit-Translate-data/${2:-curve_v6}"
# Bản Windows của DST: /d/... chỉ được Git Bash đổi thành D:/... khi nằm ở argv.
# Nhúng /d/... vào string literal trong `python -c` thì Python Windows KHÔNG hiểu.
DSTWIN="D:/Bit-Translate-data/${2:-curve_v6}"
EVERY="${3:-2000}"
PERIOD="${4:-120}"
mkdir -p "$DST"
echo "[neo] $SRC -> $DST | moc %$EVERY | kiem moi ${PERIOD}s"

while true; do
  for f in "$SRC"/step*.pt; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    n=${b#step}; n=${n%.pt}
    # chỉ nhận số; bỏ qua tên lạ
    case "$n" in ''|*[!0-9]*) continue ;; esac
    if [ $((n % EVERY)) -eq 0 ] && [ ! -f "$DST/$b" ]; then
      # copy ra .part rồi mới đổi tên -> không bao giờ để lại file nửa vời nếu bị ngắt
      if cp "$f" "$DST/$b.part" 2>/dev/null; then
        # BẮT BUỘC kiểm torch.load TRƯỚC khi nhận. Vì sao: backup_ckpt_loop.sh tải về
        # rồi MỚI kiểm, hỏng thì xoá và tải lại. Vòng lặp này poll mỗi ${PERIOD}s nên
        # có thể copy đúng vào khoảng giữa "đã tải" và "đã kiểm" -> bắt được bản hỏng.
        # Gặp thật 2026-07-28: step14000.pt copy ra ĐÚNG KÍCH THƯỚC nhưng nội dung hỏng
        # (PytorchStreamReader invalid header), làm chết cả job đo đường cong dev.
        if python -c "
import torch,sys
try:
    ck=torch.load(r'$DSTWIN/$b.part',map_location='cpu')
    assert 'model' in ck
except Exception as e:
    print('  CORRUPT',str(e)[:60]); sys.exit(1)
" 2>/dev/null; then
          mv "$DST/$b.part" "$DST/$b"
          echo "[neo $(date +%H:%M:%S)] giu $b"
        else
          rm -f "$DST/$b.part"
          echo "[neo $(date +%H:%M:%S)] $b HONG luc copy (se thu lai vong sau)"
        fi
      else
        rm -f "$DST/$b.part"
        echo "[neo $(date +%H:%M:%S)] copy LOI $b (se thu lai)"
      fi
    fi
  done
  sleep "$PERIOD"
done
