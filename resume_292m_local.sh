#!/usr/bin/env bash
# ============================================================================
# Resume train 292M LOCAL trên RTX 3060 Ti 8GB — config TỐI ƯU + bền vững qua đêm.
#
# FIX CRASH SÁNG 18/07: run nhầm cloud/run_300m.sh -> nó set
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# mà allocator này gây "CUDA driver error: device not ready" trên WSL2 (ghi rõ ở
# scripts/run_training.sh:8). Script NÀY dùng allocator MẶC ĐỊNH (không set) +
# --sdpa mem (né hẳn kernel flash-attn backward nơi lỗi từng nổ).
#
# Tối ưu tốc độ: 4 patch tương đương-toán-học (BITNET_OPT mặc định bật) +
# fused-adamw + selective/không gradient-checkpoint (điền theo bench).
#
# Chạy nền độc lập phiên làm việc:
#   setsid nohup bash resume_292m_local.sh </dev/null >checkpoints/local292.log 2>&1 &
# Theo dõi:  tail -f checkpoints/train.log   |   bash watch.sh
# ============================================================================
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

# KHÔNG expandable_segments (nguyên nhân crash sáng nay). Tường minh xoá.
unset PYTORCH_CUDA_ALLOC_CONF

DIMS="--d-model 1152 --n-layers 16 --n-heads 18 --d-ff 3072"
# KHÔNG --compile: torch.compile trên 292M/8GB với shape đa dạng -> hoặc shape-
#   explosion (compile graph mới mãi -> treo) hoặc phân mảnh VRAM 97% -> thrash
#   (step steady-state 300s+). Đã thử k4/full-GC/pad-multiple đều dính. Eager
#   KHÔNG có inductor buffer/phân mảnh -> CHẠY ỔN ĐỊNH.
# KHÔNG --sdpa mem cần thiết trong eager, nhưng giữ để né hẳn flash-attn backward
#   (nơi crash sáng nay nổ). Đo eager: sdpa mem ~= flash, giữ cho an toàn.
COMMON="--max-steps 25000 --lr 2.5e-4 --min-lr 2.5e-5 --warmup 1000 \
        --save-every 100 --milestone-every 2000 --log-every 5 --sdpa mem"

# === CONFIG (chốt từ THỰC CHIẾN 18/07 trên chính 3060 Ti 8GB này) ===
# Cả hai MT2048 GA64 (131072 token/step -> schedule LR y hệt run cũ) + full-GC.
# BÀI HỌC: MT4096 (config CLOUD của run_300m.sh) phân mảnh tới 7.98GB -> thrash cả
#   compile lẫn eager. MT2048 (config LOCAL đúng cho 8GB) chạy được. torch.compile
#   loại hẳn (treo/thrash trên 8GB). Đo ĐƯỢC: eager MT2048 = ~27s/step ỔN ĐỊNH,
#   VRAM 7.97GB đứng yên (không leo), loss ~1.3 khớp checkpoint.
# PRIMARY = eager MT2048 + 4 opt (BITNET_OPT) + fused-adamw: ~27s/step. fused-resume test PASS.
# SAFE = eager MT2048, BỎ wqcache (-0.5GB VRAM, thêm headroom) + AdamW THƯỜNG
#   (resume đã chứng minh). Dùng khi PRIMARY lỡ OOM/crash-loop.
PRIMARY="--max-tokens 2048 --grad-accum 64 --grad-ckpt --fused-adamw"
SAFE="--max-tokens 2048 --grad-accum 64 --grad-ckpt"
PRIMARY_OPT="ste,wqcache,fusedproj,maskce"
SAFE_OPT="ste,fusedproj,maskce"

# Marker: last.pt hiện tại LÀ của run 292M -> đừng để script khác dọn nhầm.
touch checkpoints/.scale300m

last_step() { grep -oE '^step [0-9]+' checkpoints/train.log 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }

ARGS="$PRIMARY"; mode="PRIMARY"; export BITNET_OPT="$PRIMARY_OPT"
noprogress=0
prev=$(last_step); prev=${prev:-0}

for i in $(seq 1 500); do
  echo ">>> local292 launch $i [$mode] BITNET_OPT=$BITNET_OPT $(date -u +%H:%M:%S) prevstep=$prev" >> checkpoints/train.log
  python3 scripts/train.py $DIMS $ARGS $COMMON
  rc=$?

  if [ -f checkpoints/PAUSED ]; then
    echo ">>> PAUSED by user, dừng launcher (resume: bash resume_292m_local.sh)" >> checkpoints/train.log
    break
  fi
  if grep -q "training loop exited" checkpoints/train.log; then
    echo ">>> TRAINING COMPLETE (step 25000)" >> checkpoints/train.log
    break
  fi

  cur=$(last_step); cur=${cur:-0}
  if [ "$cur" -gt "$prev" ]; then
    noprogress=0                         # có tiến -> crash chỉ là transient, resume tiếp
  else
    noprogress=$((noprogress+1))
    echo ">>> crash KHÔNG tiến step ($noprogress lần liên tiếp, rc=$rc)" >> checkpoints/train.log
  fi
  prev=$cur

  # Crash-loop bảo vệ: 3 lần liên tiếp không nhích step -> hạ về config SAFE.
  if [ "$noprogress" -ge 3 ] && [ "$mode" = "PRIMARY" ]; then
    ARGS="$SAFE"; mode="SAFE"; noprogress=0; export BITNET_OPT="$SAFE_OPT"
    echo ">>> !! PRIMARY crash-loop -> chuyển sang SAFE config (bỏ wqcache, +headroom), chạy tiếp qua đêm" >> checkpoints/train.log
  elif [ "$noprogress" -ge 6 ]; then
    echo ">>> !! SAFE cũng crash-loop 6 lần -> DỪNG để sáng xem (tránh phá đêm)" >> checkpoints/train.log
    break
  fi
  sleep 15
done
