#!/bin/bash
# FAILOVER: tritue12 chết -> chuyển mốc cao nhất sang nguyentuanngai, train tiếp tới 8500.
# Quy trình theo memory "doi-tai-khoan-modal-khi-het-credit": đổi tên stepN.pt -> last.pt
# (GIỮ step trong checkpoint để LR schedule nối; KHÔNG init_round vì nó reset step=0).
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
D=D:/Bit-Translate-data; V=/d/Bit-Translate-data
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== chờ tritue12 chết (backup log đứng 12 phút) ==="
LAST=""
STALL=0
while true; do
  sleep 180
  CUR=$(ls "$V/checkpoints_v7a"/step*.pt 2>/dev/null | sed 's/.*step//;s/\.pt//' | sort -n | tail -1)
  if [ "$CUR" = "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
  say "moc cao nhat local: step$CUR (stall=$STALL)"
  [ "$STALL" -ge 4 ] && break
  LAST=$CUR
done
TOP=$(ls "$V/checkpoints_v7a"/step*.pt | sed 's/.*step//;s/\.pt//' | sort -n | tail -1)
say "=== tritue12 chết. Mốc chuyển giao: step$TOP ==="
pkill -f "backup_ckpt_loop.sh checkpoints_v7a" 2>/dev/null

# kiểm toàn vẹn + dựng last.pt (giữ nguyên step)
python -c "import torch;ck=torch.load(r'$D/checkpoints_v7a/step$TOP.pt',map_location='cpu');assert ck['step']==$TOP;print('OK step',ck['step'])" || { say "MOC HỎNG - DỪNG"; exit 1; }
mkdir -p "$V/ckpt_v7a_ng"
cp "$V/checkpoints_v7a/step$TOP.pt" "$V/ckpt_v7a_ng/last.pt"

say "--- upload sang nguyentuanngai ---"
MODAL_PROFILE=nguyentuanngai modal volume create vija-100m-kd-vol >/dev/null 2>&1
MODAL_PROFILE=nguyentuanngai modal volume put vija-100m-kd-vol "$D/bin_v7g" bin_v7g --force >/dev/null 2>&1
MODAL_PROFILE=nguyentuanngai modal volume put vija-100m-kd-vol "$D/ckpt_v7a_ng" checkpoints_v7a --force >/dev/null 2>&1
MODAL_PROFILE=nguyentuanngai modal volume ls vija-100m-kd-vol 2>&1 | tail -3

say "--- train tiếp: step$TOP -> 8500 (~\$$(( (8500-TOP)/1000 )).x) ---"
MODAL_PROFILE=nguyentuanngai modal run --detach cloud/modal_train_100m_kd.py::train_v6 \
  --steps 8500 --lr 5e-05 --bin-dir bin_v7g --ckpt-dir checkpoints_v7a \
  > logs/train_v7a_p3.log 2>&1
say "đã phát lệnh. Bật backup nguyentuanngai."
MODAL_PROFILE=nguyentuanngai KEEP=12 nohup bash scripts/backup_ckpt_loop.sh checkpoints_v7a 180 \
  > logs/backup_v7a_ng.log 2>&1 &

# theo dõi tới xong/chết, rồi avg 4 mốc cuối
while true; do
  sleep 300
  MODAL_PROFILE=nguyentuanngai modal volume get vija-100m-kd-vol checkpoints_v7a/train.log \
    "$V/checkpoints_v7a/train.log" --force >/dev/null 2>&1
  CUR=$(grep -oE "step [0-9]+" "$V/checkpoints_v7a/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  DEV=$(grep -oE "DEV step [0-9]+ \| dev_loss [0-9.]+" "$V/checkpoints_v7a/train.log" 2>/dev/null | tail -1)
  say "p3 step ${CUR:-?}/8500 | ${DEV:-}"
  [ "${CUR:-0}" -ge 8500 ] && break
  grep -q "training loop exited" "$V/checkpoints_v7a/train.log" 2>/dev/null && break
done
say "P3 DỪNG. Avg 4 mốc cuối local:"
sleep 200   # cho backup vớt nốt
F=$(ls "$V/checkpoints_v7a"/step*.pt | sed 's/.*step//;s/\.pt//' | sort -n | tail -4 | sed "s|^|$D/checkpoints_v7a/step|;s|$|.pt|" | tr '\n' ' ')
python scripts/avg_ckpt.py $F -o "$D/checkpoints_v7a/v7a_avg.pt" 2>&1 | tail -1
say "--- dịch 2 bench bằng v7a_avg ---"
OMP_NUM_THREADS=5 python scripts/translate_bench.py "$D/checkpoints_v7a/v7a_avg.pt" --label v7aopus --probe eval/bench_opus100.jsonl >> logs/overnight_v7a.log 2>&1
OMP_NUM_THREADS=5 python scripts/translate_bench.py "$D/checkpoints_v7a/v7a_avg.pt" --label v7arand --probe eval/bench_rand.jsonl >> logs/overnight_v7a.log 2>&1
say "=== FAILOVER HOÀN TẤT — bench sẵn: bench_v7aopus/v7arand. Chấm mù để kết luận. ==="
