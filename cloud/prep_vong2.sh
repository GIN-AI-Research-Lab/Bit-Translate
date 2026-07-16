#!/usr/bin/env bash
# prep_vong2.sh — chuẩn bị + train VÒNG 2 trên node mới, MỘT lệnh.
#   bash cloud/prep_vong2.sh
# Mặc định: tải bin ĐÃ MIX SẴN (bin_mix_vong2.tar.zst, mix + LaBSE làm ở local)
# + checkpoint 19000 -> train 19000->23000 (LR restart anchor 19000) NGAY.
# Nếu thiếu gói mix sẵn (hoặc PREMIX=0): tải base + data vòng1 + vòng2 -> tự mix.
# Yêu cầu: gh auth login + zstd + torch/cuda. Cờ: PREMIX=0, SKIP_LABSE=1,
# NO_TRAIN=1, MIX_PY=path (venv LaBSE riêng), NEW_FRAC=0.30, OS_SHORT=1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
PY=python3
MIX_PY=${MIX_PY:-$PY}
REPO=trituenguyen97/Bit-Translate
say(){ echo -e "\n==================== $* ===================="; }

LOCK=checkpoints/.prep_vong2.lock
mkdir -p checkpoints
if [ -f "$LOCK" ]; then
  OLD_PID=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$OLD_PID" ] && tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null | grep -q "prep_vong2.sh"; then
    echo "!! prep_vong2.sh ĐANG CHẠY (PID $OLD_PID)"; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

command -v gh >/dev/null 2>&1 || { echo "!! chưa cài gh (conda install -y gh -c conda-forge)"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "!! gh CHƯA đăng nhập -> gh auth login (PAT scope repo)"; exit 1; }

# 0) ĐƯỜNG TẮT: bin ĐÃ MIX SẴN ở local (mix + LaBSE + oversample xong rồi,
#    kèm dev + flores) -> node khỏi tải data lẻ, khỏi cài sentence-transformers.
#    Tắt bằng PREMIX=0 để chạy pipeline mix đầy đủ như cũ.
PREMIXED=0
if [ "${PREMIX:-1}" = "1" ]; then
  if [ -f data/bin/.premixed_vong2 ]; then
    PREMIXED=1
  elif gh release download train-assets-vong2 --repo "$REPO" --pattern 'bin_mix_vong2.tar.zst' 2>/dev/null; then
    say "Giải nén bin mix sẵn (bỏ tải data lẻ + mix + LaBSE)"
    tar -I zstd -xf bin_mix_vong2.tar.zst
    touch data/bin/.premixed_vong2
    rm -f bin_mix_vong2.tar.zst
    PREMIXED=1
  else
    echo "!! không thấy bin_mix_vong2.tar.zst trên release -> rơi về pipeline mix đầy đủ"
  fi
fi

# 1) checkpoint 19000 (fp32 + optimizer)
if [ ! -f checkpoints/last.pt ]; then
  say "Tải checkpoint 19000 (vong1-step19000)"
  rm -f last_final_19000.pt
  gh release download vong1-step19000 --repo "$REPO" --pattern 'last_final_19000.pt'
  mv last_final_19000.pt checkpoints/last.pt
fi

if [ "$PREMIXED" = "0" ]; then
# 2) base bin (tokens gốc 5.4M)
if [ ! -f data/bin/train.tokens.u16 ] && [ ! -f data/bin/base.train.tokens.u16 ]; then
  say "Tải base data (train-assets-step14000)"
  rm -f vija_data.tar.zst
  gh release download train-assets-step14000 --repo "$REPO" --pattern 'vija_data.tar.zst'
  tar -I zstd -xf vija_data.tar.zst
fi
# 3) data vòng 1 (synthetic + glossary) + bt
if [ ! -f data/synthetic/glossary_sents.jsonl ]; then
  say "Tải data Vòng 1 (train-assets-vong1)"
  rm -f vong1-data.tar.gz
  gh release download train-assets-vong1 --repo "$REPO" --pattern 'vong1-data.tar.gz'
  tar xzf vong1-data.tar.gz
fi
if [ ! -f data/synthetic/bt.ja ]; then
  say "Tải back-translation Vòng 1 (bt_vong1.tar.gz)"
  rm -f bt_vong1.tar.gz
  gh release download vong1-step19000 --repo "$REPO" --pattern 'bt_vong1.tar.gz'
  tar xzf bt_vong1.tar.gz
fi
# 4) data vòng 2
if [ ! -f data/synthetic/vong2_pairs.jsonl ]; then
  say "Tải data Vòng 2 (train-assets-vong2)"
  rm -f vong2-data.tar.gz
  gh release download train-assets-vong2 --repo "$REPO" --pattern 'vong2-data.tar.gz'
  tar xzf vong2-data.tar.gz
fi

# 5) mix: base + vòng1 + vòng2 (+ oversample hội thoại ngắn ×1)
say "Mix + binarize (LaBSE=$([ "${SKIP_LABSE:-0}" = 1 ] && echo off || echo on), new-frac=${NEW_FRAC:-0.30}, short×${OS_SHORT:-1})"
LABSE_FLAG=""; [ "${SKIP_LABSE:-0}" = "1" ] && LABSE_FLAG="--no-labse"
"$MIX_PY" scripts/mix_and_binarize.py --new-frac "${NEW_FRAC:-0.30}" --oversample-short "${OS_SHORT:-1}" $LABSE_FLAG
fi  # PREMIXED

# 5) train
if [ "${NO_TRAIN:-0}" = "1" ]; then
  say "NO_TRAIN=1 -> data sẵn ở data/bin. Chạy tay: nohup bash cloud/run_vong2.sh > checkpoints/vong2.log 2>&1 &"
  exit 0
fi
say "Train Vòng 2 (nền): 19000 -> 23000"
nohup bash cloud/run_vong2.sh > checkpoints/vong2.log 2>&1 &
echo "PID $! | theo dõi: tail -f checkpoints/train.log"
