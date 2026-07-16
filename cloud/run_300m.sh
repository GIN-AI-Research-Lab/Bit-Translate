#!/usr/bin/env bash
# run_300m.sh — TRAIN FROM-SCRATCH model ~292M (d1152 / 16 layer / 18 head / ff3072)
# trên node 16GB, dùng data/bin sẵn có (premix vòng 2 = base + vòng 1 + vòng 2).
#
# Căn cứ scale: 4/4 tín hiệu trần 110M sau Vòng 2 (PLAN_BUOC5 §3.5 + eval/capacity_log.md);
# user chốt ~300M (thay vì 200M của §5.1) để dư sức chứa cho vòng 3-5.
#
# Chạy nền:   nohup bash cloud/run_300m.sh > checkpoints/scale300m.log 2>&1 &
# Theo dõi:   tail -f checkpoints/train.log
# Watcher (bật SAU khi train đã chạy — tái dùng watcher vòng 1, script này ghi
# cùng marker "VONG1 COMPLETE"):
#   WATCH_TAG=scale300m-step25000 nohup bash cloud/watch_vong1.sh > checkpoints/watch.log 2>&1 &
#
# Env: M300_STEPS=25000  M300_MT=8192  M300_GA=16  M300_LR=2.5e-4  M300_COMPILE=1
#      M300_GC=1 (gradient checkpointing — VRAM giảm mạnh, đổi ~30% tốc độ; dùng khi
#      node <12GB OOM cả ở MT nhỏ)   M300_FORCE=1 (bỏ guard backup 110M)
# Bài học node 4070 Ti 12GB: eager OOM cả MT=4096 (STE tạo bản sao fp32 mỗi BitLinear);
# compile fuse nên nhẹ hơn — thử compile+MT4096 trước, rồi mới tới M300_GC=1.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=trituenguyen97/Bit-Translate

STEPS=${M300_STEPS:-25000}
MT=${M300_MT:-8192}
GA=${M300_GA:-16}       # 8192*16 = 131072 token budget/step — giữ nguyên như 110M
LR=${M300_LR:-2.5e-4}   # thấp hơn 3e-4 của 110M (model rộng 1.5x); min-lr theo tỉ lệ
COMPILE=""; [ "${M300_COMPILE:-1}" = "1" ] && COMPILE="--compile"
GC="";      [ "${M300_GC:-0}" = "1" ] && GC="--grad-ckpt"

mkdir -p checkpoints dist

# ==== GUARD 1: cứu bản fp32+optimizer 110M step23000 TRƯỚC khi from-scratch đè ====
if ! gh release view vong2-step23000 --repo "$REPO" --json assets \
     --jq '.assets[].name' 2>/dev/null | grep -qx 'last_final_23000.pt'; then
  if [ -f dist/last_final_23000.pt ]; then
    echo ">> last_final_23000.pt CHƯA có trên release -> upload trước khi train..."
    if ! gh release upload vong2-step23000 --repo "$REPO" --clobber dist/last_final_23000.pt; then
      echo "!! upload FAIL — không dám train đè checkpoint 110M."
      [ "${M300_FORCE:-0}" = "1" ] || { echo "   Upload tay rồi chạy lại (hoặc M300_FORCE=1)."; exit 1; }
    fi
  else
    echo "!! KHÔNG thấy last_final_23000.pt trên release LẪN dist/ — fp32+optimizer 110M sẽ MẤT."
    [ "${M300_FORCE:-0}" = "1" ] || { echo "   Chặn lại. Chấp nhận mất thì chạy M300_FORCE=1."; exit 1; }
  fi
fi

# ==== GUARD 2: dọn checkpoint + log 110M sang một bên (KHÔNG xoá) ====
# Marker .scale300m = last.pt hiện tại LÀ của run 292M (đặt lúc launch đầu / restore_300m.sh)
# -> các lần chạy lại script sau đó KHÔNG dọn nó đi (không thì train lại từ 0!).
if [ ! -f checkpoints/.scale300m ]; then
  if [ -f checkpoints/last.pt ]; then
    mkdir -p checkpoints_110m
    mv -f checkpoints/last.pt checkpoints_110m/
    mv -f checkpoints/step*.pt checkpoints_110m/ 2>/dev/null || true
    echo ">> checkpoint 110M -> checkpoints_110m/"
  fi
  if [ -f checkpoints/train.log ]; then
    mv -f checkpoints/train.log "checkpoints/train.pre300m.$(date +%s).log"
  fi
fi
touch checkpoints/.scale300m

# ckpt 292M: last.pt (fp32+opt) ~3.5GB ghi đè mỗi 100 step + milestone ~1.2GB x ~12
echo ">> disk còn trống:"; df -h . | tail -1

ARGS="--d-model 1152 --n-layers 16 --n-heads 18 --d-ff 3072 \
--max-tokens $MT --grad-accum $GA --max-steps $STEPS \
--lr $LR --min-lr 2.5e-5 --warmup 1000 \
--save-every 100 --milestone-every 2000 --log-every 5 $COMPILE $GC"

# Chặn 2 train.py cùng chạy (y hệt run_vong2.sh)
for pid in $(pgrep -f 'scripts/train.py' 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'scripts/train.py'; then
    echo "!! đã có train.py chạy (PID $pid) -> không launch trùng."; exit 1
  fi
done

rm -f checkpoints/PAUSED
echo ">> FROM-SCRATCH ~292M: $ARGS"
for i in $(seq 1 500); do
  echo ">>> scale300m launch $i $(date -u +%H:%M:%S)" >> checkpoints/train.log
  python3 scripts/train.py $ARGS
  if [ -f checkpoints/PAUSED ]; then echo ">>> PAUSED, dừng launcher" >> checkpoints/train.log; break; fi
  if grep -q "training loop exited" checkpoints/train.log; then echo ">>> VONG1 COMPLETE" >> checkpoints/train.log; break; fi
  echo ">>> crashed lần $i, resume sau 10s" >> checkpoints/train.log
  sleep 10
done
