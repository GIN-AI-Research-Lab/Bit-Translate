#!/usr/bin/env bash
# backup_300m.sh — daemon backup train 292M lên MỘT release cố định (autosave),
# clobber tại chỗ, KHÔNG sinh release mới — chống mất trắng khi node chết/hết tiền.
#
# Chạy nền song song với train:
#   nohup bash cloud/backup_300m.sh > checkpoints/backup.log 2>&1 &
#
# Cách hoạt động:
# - Mỗi BK_INTERVAL giây (mặc định 1800 = 30'), nếu train đã sang step mới:
#   snapshot checkpoints/last.pt (fp32+optimizer ~3.5GB, VƯỢT hạn 2GB/file GitHub)
#   -> split 1900M -> upload --clobber + file .step (marker "bộ này nguyên vẹn").
# - Luân phiên 2 bộ last_a.* / last_b.*: bộ đang upload dở KHÔNG phá bộ lành trước.
#   .step upload CUỐI CÙNG — có .step mới = bộ đó chắc chắn đầy đủ.
# - Mất tối đa BK_INTERVAL phút train nếu node bay màu.
# Khôi phục trên node mới: bash cloud/restore_300m.sh
#
# Env: BK_TAG=autosave-scale300m  BK_INTERVAL=1800
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
REPO=trituenguyen97/Bit-Translate
TAG=${BK_TAG:-autosave-scale300m}
INTERVAL=${BK_INTERVAL:-1800}

# khoá chống 2 daemon
LOCK=checkpoints/.backup_300m.lock
mkdir -p checkpoints dist/autosave
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q "backup_300m.sh"; then
    echo "!! backup daemon ĐANG CHẠY (PID $OLD) -> thoát."; exit 1
  fi
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

gh auth status >/dev/null 2>&1 || { echo "!! gh chưa đăng nhập -> thoát."; exit 1; }
gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 || \
  gh release create "$TAG" --repo "$REPO" --title "AUTOSAVE scale 292M (ghi đè liên tục)" \
    --notes "Backup tự động mỗi ~30': last.pt (fp32+optimizer, split 2 phần <2GB) + train.log.
Bộ last_a/last_b luân phiên — bộ có file .step MỚI NHẤT là bộ nguyên vẹn.
Khôi phục node mới: git clone -> gh auth -> bash cloud/restore_300m.sh -> chạy lại run_300m.sh."

cur_step() { grep -E "^step " checkpoints/train.log 2>/dev/null | tail -1 | grep -oE "^step [0-9]+" | grep -oE "[0-9]+"; }

echo "[backup] tag=$TAG mỗi ${INTERVAL}s | chờ last.pt..."
LAST_UP=""
i=0
while :; do
  STEP=$(cur_step || true)
  if [ -f checkpoints/last.pt ] && [ -n "${STEP:-}" ] && [ "$STEP" != "$LAST_UP" ]; then
    SET=$([ $((i % 2)) -eq 0 ] && echo a || echo b)
    echo "[backup] $(date -u +%H:%M) step $STEP -> bộ $SET"
    rm -f dist/autosave/*
    # cp an toàn: train.py thay last.pt bằng os.replace (atomic) — cp giữ inode cũ trọn vẹn
    cp checkpoints/last.pt dist/autosave/snap.pt
    split -b 1900m dist/autosave/snap.pt "dist/autosave/last_${SET}.part_"
    rm -f dist/autosave/snap.pt
    cp checkpoints/train.log dist/autosave/train_autosave.log 2>/dev/null || true
    { echo "$STEP"; date -u; } > "dist/autosave/last_${SET}.step"
    OK=1
    for f in dist/autosave/last_${SET}.part_* dist/autosave/train_autosave.log; do
      [ -f "$f" ] || continue
      UP=0
      for try in 1 2 3; do
        gh release upload "$TAG" --repo "$REPO" --clobber "$f" && { UP=1; break; }
        echo "[backup] upload $f fail lần $try, thử lại sau 30s"; sleep 30
      done
      [ "$UP" = "1" ] || { OK=0; break; }
    done
    # .step upload CUỐI — chỉ khi mọi part đã lên đủ (marker bộ nguyên vẹn)
    if [ "$OK" = "1" ]; then
      if gh release upload "$TAG" --repo "$REPO" --clobber "dist/autosave/last_${SET}.step"; then
        echo "[backup] ✓ step $STEP đã lên release ($TAG, bộ $SET)"
        LAST_UP="$STEP"; i=$((i + 1))
      fi
    else
      echo "[backup] ✗ bộ $SET dở dang — bộ trước đó trên release vẫn nguyên, thử lại vòng sau"
    fi
    rm -f dist/autosave/*
  else
    echo "[backup] $(date -u +%H:%M) chưa có step mới (hiện: ${STEP:-chưa có})"
  fi
  # dừng cùng train: hết train.py sống VÀ log đã báo xong thì up nốt lần cuối rồi thoát
  if grep -q "VONG1 COMPLETE" checkpoints/train.log 2>/dev/null && ! pgrep -f 'scripts/train\.py' >/dev/null 2>&1; then
    if [ "$(cur_step || true)" = "$LAST_UP" ]; then echo "[backup] train XONG + đã backup bản cuối -> thoát."; break; fi
  fi
  sleep "$INTERVAL"
done
