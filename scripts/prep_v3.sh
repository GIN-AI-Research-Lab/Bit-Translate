#!/bin/bash
# Chuỗi chuẩn bị data + khởi chạy vòng 3 (grow 12L->18L).
#   bash scripts/prep_v3.sh          # chỉ chuẩn bị data
#   RUN_TRAIN=1 bash scripts/prep_v3.sh   # chuẩn bị xong chạy train luôn
#
# oversample 3 (không phải 4 như vòng 2): data niche giờ ~370k cặp, nhân 3 cho ra
# ~10% mix — giữ đúng tỷ lệ đã hiệu quả ở vòng 2 (8,39%). Nhân 4 sẽ thành ~12,4%,
# lệch quá xa phân phối tự nhiên.
set -e
cd "$(dirname "$0")/.."
D=D:/Bit-Translate-data
export PYTHONUTF8=1

echo "=== 1) Merge niche (ca o E lan D) vao corpus ==="
python scripts/merge_niche_corpus.py --oversample 3 \
  --out "$D/kd_filtered_niche_v3.jsonl" 2>&1 | tail -22

echo "=== 2) Split train/dev ==="
python scripts/prep_clean_split.py "$D/kd_filtered_niche_v3.jsonl" "$D/clean_v3" 2>&1 | tail -3

echo "=== 3) Binarize ==="
python scripts/binarize_ja2vi.py --clean "$D/clean_v3" --out "$D/bin_v3" 2>&1 | tail -4

echo "=== 4) Upload len Modal ==="
modal volume put vija-100m-kd-vol "$D/bin_v3" bin_v3 2>&1 | tail -2

if [ "${RUN_TRAIN:-0}" = "1" ]; then
  echo "=== 5) Train vong 3 (grow 18L) ==="
  echo "  Nho chay backup song song o terminal khac:"
  echo "    bash scripts/backup_ckpt_loop.sh checkpoints_v3 600"
  modal run --detach cloud/modal_train_100m_kd.py::train_v3 \
    --steps 8000 --lr 1e-4 --n-layers 18 --bin-dir bin_v3 2>&1 | tail -5
else
  echo
  echo "Data san sang. Chay train bang:"
  echo "  modal run --detach cloud/modal_train_100m_kd.py::train_v3 \\"
  echo "      --steps 8000 --lr 1e-4 --n-layers 18 --bin-dir bin_v3"
  echo "  bash scripts/backup_ckpt_loop.sh checkpoints_v3 600   # terminal khac"
fi
