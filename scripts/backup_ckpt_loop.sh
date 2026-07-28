#!/bin/bash
# Backup checkpoint từ Modal volume về ổ D (và tuỳ chọn GitHub Release) trong lúc train.
#
# Vì sao cần: credit Modal có thể cạn giữa chừng. Guardian trong train script đã
# vol.commit() mỗi 5 phút nên checkpoint KHÔNG mất khỏi volume — nhưng nếu account
# bị khoá vì hết credit thì có thể không tải về được nữa. Script này kéo mọi
# milestone mới về máy ngay khi nó xuất hiện.
#
# Chạy song song với train:
#   bash scripts/backup_ckpt_loop.sh checkpoints_v3 600
#     $1 = thư mục checkpoint trên volume (mặc định checkpoints_v3)
#     $2 = chu kỳ kiểm tra, giây (mặc định 600 = 10 phút)
#
# Bật đẩy lên GitHub Release (file >100MB không push vào git được, phải qua release):
#   BACKUP_GH=1 GH_TAG=ckpt-200m bash scripts/backup_ckpt_loop.sh checkpoints_v3
set -u
cd "$(dirname "$0")/.."
VOLDIR="${1:-checkpoints_v3}"
PERIOD="${2:-600}"
VOL=vija-100m-kd-vol
DEST="/d/Bit-Translate-data/$VOLDIR"
DESTWIN="D:/Bit-Translate-data/$VOLDIR"
export PYTHONUTF8=1
mkdir -p "$DEST"

echo "[backup] volume=$VOL/$VOLDIR -> $DEST | chu ky ${PERIOD}s"
echo "[backup] Ctrl-C de dung. Log: logs/backup_$VOLDIR.log"

# CHỈ backup milestone stepN.pt (~608MB, model-only), KHÔNG backup last.pt.
# Vì sao: last.pt chứa cả optimizer state -> ~1,8GB, tải mất ~7 phút ở 4MB/s, chiếm
# gần hết chu kỳ. Mà để resume thì KHÔNG cần nó: train.py chịu được checkpoint thiếu
# "opt" (in "checkpoint không có optimizer state -> dùng optimizer mới"). Milestone
# mỗi 500 step (~11 phút) nên mất mát tối đa khi hết credit đột ngột là ~11 phút train.
# Muốn lấy cả last.pt thì chạy tay: modal volume get <vol> <dir>/last.pt <đích>
while true; do
  LIST=$(modal volume ls "$VOL" "$VOLDIR" 2>/dev/null | grep -oE 'step[0-9]+\.pt' | sort -u)
  # Với KEEP, phải LỌC DANH SÁCH TẢI luôn — không được tải hết rồi mới tỉa. Volume giữ
  # đủ 65 mốc, nên nếu chỉ tỉa ở cuối vòng thì mốc cũ bị tải lại rồi xoá, tải lại rồi
  # xoá... (đo thật: step250 bị tỉa 5 lần, 30 lượt tải cho 18 file, phí ~7GB băng thông).
  if [ "${KEEP:-0}" -gt 0 ]; then
    LIST=$(printf '%s\n' $LIST | sed 's/step\([0-9]*\)\.pt/\1 &/' | sort -n \
           | tail -n "${KEEP}" | cut -d' ' -f2-)
  fi
  for f in $LIST; do
    local_f="$DEST/$f"
    if [ ! -f "$local_f" ]; then
      echo "[backup $(date +%H:%M:%S)] tai $f ..."
      if modal volume get "$VOL" "$VOLDIR/$f" "$local_f" >/dev/null 2>&1; then
        # modal volume get CO THE ra file corrupt IM LANG -> luon kiem tra
        if python -c "
import torch,sys
try:
    ck=torch.load(r'$DESTWIN/$f',map_location='cpu')
    assert 'model' in ck
    print('  OK', '$f', 'step', ck.get('step'))
except Exception as e:
    print('  CORRUPT', '$f', str(e)[:70]); sys.exit(1)
"; then
          if [ "${BACKUP_GH:-0}" = "1" ]; then
            TAG="${GH_TAG:-ckpt-$VOLDIR}"
            gh release view "$TAG" >/dev/null 2>&1 || \
              gh release create "$TAG" -t "Checkpoint $VOLDIR" -n "Backup tu Modal volume" >/dev/null 2>&1
            echo "  -> GitHub Release $TAG"
            gh release upload "$TAG" "$local_f" --clobber >/dev/null 2>&1 \
              && echo "  đã upload $f" || echo "  upload GH LỖI (file >2GB? gh chưa login?)"
          fi
        else
          rm -f "$local_f"   # xoá file hỏng để lần sau tải lại
        fi
      else
        echo "  tai LOI $f (se thu lai lan sau)"
      fi
    fi
  done
  # CẮT TỈA: chỉ giữ $KEEP milestone mới nhất. Không đặt KEEP -> giữ hết (như cũ).
  # Vì sao: vòng 6 lưu mốc mỗi 250 step -> 65 file x 580MiB = 37GiB, lớn hơn chỗ
  # trống ổ D (29,7GB lúc 2026-07-28). auto_eval chỉ cần 7 mốc CUỐI để average,
  # nên giữ 12 là dư. Xoá theo SỐ STEP, không theo thời gian tạo file.
  if [ "${KEEP:-0}" -gt 0 ]; then
    ls "$DEST"/step*.pt 2>/dev/null \
      | sed 's/.*step\([0-9]*\)\.pt$/\1 &/' | sort -n | head -n "-${KEEP}" | cut -d' ' -f2- \
      | while read -r old; do
          echo "[backup $(date +%H:%M:%S)] tia $(basename "$old")"
          rm -f "$old"
        done
  fi
  # tải log train mỗi vòng (nhỏ, để theo dõi cả khi mất Modal)
  modal volume get "$VOL" "$VOLDIR/train.log" "$DEST/train.log" --force >/dev/null 2>&1
  sleep "$PERIOD"
done
