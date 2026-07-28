#!/bin/bash
# Phần cuối chuỗi vòng 5: chờ binarize -> upload -> backup -> train.
# Tách khỏi overnight_v5.sh vì gộp/binarize đã chạy tay xong.
#
#   nohup bash scripts/finish_v5.sh 14000 > logs/finish_v5.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
STEPS="${1:-14000}"
D=D:/Bit-Translate-data
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "chờ binarize xong (bin_v5/train.index.npy)"
for i in $(seq 1 240); do
  if grep -q "^done ->" logs/prep_v5.log 2>/dev/null; then break; fi
  sleep 60
done
if ! grep -q "^done ->" logs/prep_v5.log 2>/dev/null; then
  say "BINARIZE CHƯA XONG SAU 4 GIỜ — dừng"; exit 1
fi
tail -3 logs/prep_v5.log

say "upload bin_v5 (~1GB)"
modal volume put vija-100m-kd-vol "$D/bin_v5" bin_v5 >/dev/null 2>&1
say "upload checkpoints_v5"
modal volume put vija-100m-kd-vol "$D/checkpoints_v5" checkpoints_v5 >/dev/null 2>&1
modal volume ls vija-100m-kd-vol 2>&1 | tail -4

say "bật backup checkpoint mỗi 5 phút"
nohup bash scripts/backup_ckpt_loop.sh checkpoints_v5 300 > logs/backup_v5.log 2>&1 &

say "TRAIN vòng 5 — $STEPS step"
modal run --detach cloud/modal_train_100m_kd.py::train_v5 --steps "$STEPS" > logs/train_v5.log 2>&1
say "=== đã phát lệnh train, xem logs/train_v5.log ==="
