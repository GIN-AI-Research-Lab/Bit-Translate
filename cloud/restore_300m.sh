#!/usr/bin/env bash
# restore_300m.sh — dựng lại node train 292M từ autosave (node cũ chết/hết tiền), 1 lệnh.
# Yêu cầu: git clone xong + gh auth login xong. Chạy trong thư mục repo:
#   bash cloud/restore_300m.sh
# Tự làm: tải bin premix (nếu thiếu) + chọn bộ autosave a/b MỚI NHẤT nguyên vẹn
# -> ghép thành checkpoints/last.pt -> đặt marker để run_300m.sh KHÔNG dọn nó đi.
# Xong thì tự chạy tiếp:  nohup bash cloud/run_300m.sh > checkpoints/scale300m.log 2>&1 &
#   (nhớ env VRAM của node: 12GB -> M300_MT=4096 M300_GA=32)
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
REPO=trituenguyen97/Bit-Translate
TAG=${BK_TAG:-autosave-scale300m}

gh auth status >/dev/null 2>&1 || { echo "!! gh chưa đăng nhập (gh auth login)"; exit 1; }
mkdir -p checkpoints dist

# 1) bin premix (base + vòng1 + vòng2 đã mix/LaBSE)
if [ ! -f data/bin/train.tokens.u16 ]; then
  echo ">> Tải bin premix (452MB)..."
  gh release download train-assets-vong2 --repo "$REPO" --pattern 'bin_mix_vong2.tar.zst'
  tar -I zstd -xf bin_mix_vong2.tar.zst && touch data/bin/.premixed_vong2 && rm -f bin_mix_vong2.tar.zst
fi

# 2) chọn bộ autosave nguyên vẹn mới nhất (file .step upload cuối = chứng nhận đủ bộ)
rm -rf dist/restore && mkdir -p dist/restore
gh release download "$TAG" --repo "$REPO" --pattern 'last_*.step' --dir dist/restore 2>/dev/null || true
SA=$(head -1 dist/restore/last_a.step 2>/dev/null || echo -1)
SB=$(head -1 dist/restore/last_b.step 2>/dev/null || echo -1)
if [ "$SA" = "-1" ] && [ "$SB" = "-1" ]; then echo "!! release $TAG chưa có autosave nào."; exit 1; fi
SET=a; STEP=$SA
if [ "$SB" -gt "$SA" ] 2>/dev/null; then SET=b; STEP=$SB; fi
echo ">> Bộ mới nhất: last_${SET} (step $STEP) — tải parts..."
gh release download "$TAG" --repo "$REPO" --pattern "last_${SET}.part_*" --dir dist/restore

cat dist/restore/last_${SET}.part_* > checkpoints/last.pt
SZ=$(stat -c%s checkpoints/last.pt)
[ "$SZ" -gt 3000000000 ] || { echo "!! last.pt chỉ ${SZ}B (<3GB) — bộ hỏng? Thử bộ còn lại bằng tay."; exit 1; }
rm -rf dist/restore
touch checkpoints/.scale300m   # marker: run_300m.sh sẽ KHÔNG dọn last.pt này đi

echo "✓ Khôi phục xong: checkpoints/last.pt (step $STEP, $((SZ / 1024 / 1024))MB)."
echo "Chạy tiếp (node 12GB):"
echo "  M300_MT=4096 M300_GA=32 nohup bash cloud/run_300m.sh > checkpoints/scale300m.log 2>&1 &"
echo "  nohup bash cloud/backup_300m.sh > checkpoints/backup.log 2>&1 &"
echo "  WATCH_TAG=scale300m-step25000 nohup bash cloud/watch_vong1.sh > checkpoints/watch.log 2>&1 &"
