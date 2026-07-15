#!/usr/bin/env bash
# prep_vong1.sh — chuẩn bị data + khởi động train Vòng 1 trên node mới, MỘT lệnh.
#
#   bash cloud/prep_vong1.sh
#
# Tự làm: (0) tải base data+checkpoint nếu thiếu, (1) tải data synthetic Vòng 1 nếu
# thiếu, (2) back-translation, (3) lọc LaBSE + trộn mix + binarize, (4) train nền.
# Yêu cầu: đã `gh auth login` (repo private) + zstd + môi trường torch/cuda.
#
# Cờ env:
#   SKIP_BT=1     bỏ back-translation (nhanh; vi->ja yếu hơn)
#   SKIP_LABSE=1  bỏ lọc LaBSE khi mix (nhanh; kém sạch)
#   NO_TRAIN=1    chỉ chuẩn bị data, KHÔNG tự train (tự kiểm tra rồi chạy tay)
#   BT_MAX=n      giới hạn số câu back-translate (chạy thử nhanh)
#   NEW_FRAC=x    tỉ trọng data mới (mặc định 0.30)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
PY=python3
REPO=trituenguyen97/Bit-Translate
say(){ echo -e "\n==================== $* ===================="; }

# preflight: gh phải đăng nhập mới tải được release của repo PRIVATE.
# (chưa auth -> API trả 404 "release not found", gây hiểu nhầm là mất release)
if ! command -v gh >/dev/null 2>&1; then
  echo "!! chưa cài gh. Chạy: conda install -y gh -c conda-forge  (rồi gh auth login)"; exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "!! gh CHƯA đăng nhập -> chạy: gh auth login (GitHub.com, HTTPS, PAT scope 'repo'). Rồi chạy lại."; exit 1
fi

# 0) base data + checkpoint (release train-assets-step14000) --------------------
if [ ! -f data/bin/train.tokens.u16 ]; then
  say "Thiếu data/bin base -> tải release train-assets-step14000"
  rm -f vija_data.tar.zst
  gh release download train-assets-step14000 --repo "$REPO" --pattern 'vija_data.tar.zst'
  tar -I zstd -xf vija_data.tar.zst
fi
if [ ! -f checkpoints/last.pt ]; then
  say "Thiếu checkpoints/last.pt -> dùng last_final_14000.pt"
  [ -f last_final_14000.pt ] || gh release download train-assets-step14000 --repo "$REPO" --pattern 'last_final_14000.pt'
  mkdir -p checkpoints && mv last_final_14000.pt checkpoints/last.pt
fi

# 1) data synthetic Vòng 1 (release train-assets-vong1) -------------------------
if [ ! -f data/synthetic/glossary_sents.jsonl ]; then
  say "Thiếu data synthetic -> tải release train-assets-vong1"
  rm -f vong1-data.tar.gz
  gh release download train-assets-vong1 --repo "$REPO" --pattern 'vong1-data.tar.gz'
  tar xzf vong1-data.tar.gz
fi

# 2) back-translation cứu chiều vi->ja ------------------------------------------
if [ "${SKIP_BT:-0}" = "1" ]; then
  say "SKIP_BT=1 -> bỏ back-translation"
else
  say "Back-translation (GPU, dùng checkpoint 14000) — có thể ~1-2h cho 86k câu"
  $PY cloud/backtranslate.py
fi

# 3) lọc LaBSE + trộn + binarize ------------------------------------------------
say "Mix + binarize (LaBSE=$([ "${SKIP_LABSE:-0}" = 1 ] && echo off || echo on), new-frac=${NEW_FRAC:-0.30})"
LABSE_FLAG=""; [ "${SKIP_LABSE:-0}" = "1" ] && LABSE_FLAG="--no-labse"
$PY scripts/mix_and_binarize.py --new-frac "${NEW_FRAC:-0.30}" $LABSE_FLAG

# 4) train ----------------------------------------------------------------------
if [ "${NO_TRAIN:-0}" = "1" ]; then
  say "NO_TRAIN=1 -> data sẵn sàng ở data/bin/. Chạy tay khi muốn:"
  echo "   nohup bash cloud/run_vong1.sh > checkpoints/vong1.log 2>&1 &"
  exit 0
fi
say "Khởi động train Vòng 1 (nền)"
nohup bash cloud/run_vong1.sh > checkpoints/vong1.log 2>&1 &
echo "PID $! | theo dõi: tail -f checkpoints/train.log"
echo "Xong ~5000 step (19000): convert GGUF + eval probe64/chrF (xem PLAN_BUOC5 §2.5, gate §2.6)."
