#!/usr/bin/env bash
# watch_vong1.sh — canh train Vòng 1 xong rồi TỰ ĐỘNG đóng gói + đẩy GitHub Release.
#
# Chạy nền trên node (song song với train, vô hại — chỉ đọc log):
#   nohup bash cloud/watch_vong1.sh > checkpoints/watch.log 2>&1 &
#   tail -f checkpoints/watch.log
#
# Khi checkpoints/train.log có "VONG1 COMPLETE" (run_vong1.sh ghi lúc đạt max-steps)
# và không còn train.py sống, script sẽ:
#   1) tạo bản fp16 model-only (nhẹ, để convert/deploy)
#   2) convert GGUF F16 (linears f16, norms/embed f32) — về local chỉ cần
#      llama-quantize <f16> <i2s> I2_S 1 (KHÔNG build bitnet.cpp trên node)
#   3) gom bt.ja/bt.vi (data back-translation, tốn GPU mới có — giữ lại)
#   4) gh release create/upload toàn bộ + train.log
#
# Env:
#   WATCH_TAG       tag release (mặc định vong1-step<step>)
#   WATCH_INTERVAL  giây giữa 2 lần kiểm (mặc định 120)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:$PATH
REPO=trituenguyen97/Bit-Translate
INTERVAL=${WATCH_INTERVAL:-120}
say(){ echo -e "\n==================== $* ===================="; }

# ---- khoá chống 2 watcher (2 watcher = 2 lần upload đè nhau) -------------------
LOCK=checkpoints/.watch_vong1.lock
mkdir -p checkpoints dist
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && tr '\0' ' ' < "/proc/$OLD/cmdline" 2>/dev/null | grep -q "watch_vong1.sh"; then
    echo "!! watcher ĐANG CHẠY (PID $OLD) -> thoát."; exit 1
  fi
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# ---- preflight: gh phải auth sẵn (đừng đợi 3h xong mới phát hiện thiếu) --------
if ! gh auth status >/dev/null 2>&1; then
  echo "!! gh chưa đăng nhập (gh auth login) -> thoát để bạn sửa NGAY thay vì lúc train xong."; exit 1
fi

# ---- python cho bước đóng gói: cần torch+sentencepiece+gguf --------------------
# ưu tiên conda (có torch GPU); thiếu gguf thì cài (pure-python, không đụng torch).
PKG_PY=python3
$PKG_PY -c "import gguf" 2>/dev/null || $PKG_PY -m pip install -q gguf || true
if ! $PKG_PY -c "import torch, sentencepiece, gguf" 2>/dev/null; then
  if [ -x .venv_labse/bin/python3 ]; then
    .venv_labse/bin/pip install -q gguf 2>/dev/null || true
    if .venv_labse/bin/python3 -c "import torch, sentencepiece, gguf" 2>/dev/null; then
      PKG_PY=.venv_labse/bin/python3
    fi
  fi
fi
$PKG_PY -c "import torch, sentencepiece, gguf" 2>/dev/null || { echo "!! không có python đủ torch+sentencepiece+gguf -> thoát."; exit 1; }
echo "[watch] python đóng gói: $PKG_PY | kiểm mỗi ${INTERVAL}s | chờ VONG1 COMPLETE..."

# ---- vòng canh -----------------------------------------------------------------
while :; do
  if grep -q "VONG1 COMPLETE" checkpoints/train.log 2>/dev/null; then
    # chắc chắn không còn train.py sống (launcher đã thoát hẳn)
    ALIVE=0
    for pid in $(pgrep -f 'scripts/train.py' 2>/dev/null); do
      tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'scripts/train.py' && ALIVE=1
    done
    [ "$ALIVE" = "0" ] && break
  fi
  if [ -f checkpoints/PAUSED ]; then
    echo "[watch] $(date -u +%H:%M) PAUSED đang bật — chờ resume..."
  else
    LAST=$(grep -E "^step " checkpoints/train.log 2>/dev/null | tail -1)
    echo "[watch] $(date -u +%H:%M) ${LAST:-chưa có step}"
  fi
  sleep "$INTERVAL"
done

say "TRAIN XONG — bắt đầu đóng gói"
STEP=$($PKG_PY - <<'EOF'
import torch
print(torch.load("checkpoints/last.pt", map_location="cpu").get("step", "unknown"))
EOF
)
TAG=${WATCH_TAG:-vong1-step$STEP}
echo "[watch] step=$STEP -> tag=$TAG"

# 1) fp16 model-only (nhẹ ~1/5 bản full, đủ để convert/deploy/eval)
$PKG_PY - <<EOF
import torch
ck = torch.load("checkpoints/last.pt", map_location="cpu")
sd = {k: v.half() for k, v in ck["model"].items()}
torch.save({"model": sd, "step": ck.get("step"), "cfg": ck.get("cfg")}, "dist/ckpt_${STEP}_fp16.pt")
print("wrote dist/ckpt_${STEP}_fp16.pt")
EOF

# 2) GGUF F16 (về local: llama-quantize <f16> <i2s> I2_S 1 — nhớ số 1!)
$PKG_PY scripts/convert_to_gguf.py --ckpt checkpoints/last.pt --out "dist/vija_${STEP}_f16.gguf" --f16

# 3) data back-translation (tốn GPU mới sinh được -> giữ)
tar czf "dist/bt_vong1.tar.gz" data/synthetic/bt.ja data/synthetic/bt.vi 2>/dev/null || true

# 4) bản full để RESUME vòng 2 (fp32 + optimizer)
cp -f checkpoints/last.pt "dist/last_final_${STEP}.pt"
cp -f checkpoints/train.log "dist/train_vong1_${STEP}.log" 2>/dev/null || true

# 5) release: tạo nếu chưa có, upload --clobber (chạy lại an toàn)
say "Upload GitHub Release $TAG"
gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1 || \
  gh release create "$TAG" --repo "$REPO" --title "Vòng 1 — step $STEP" \
    --notes "Train Vòng 1 (PLAN_BUOC5 §2): glossary+IT docs+BT 86k (LaBSE 0.8), resume 14000 -> $STEP.
Assets: last_final (fp32+optimizer, resume vòng 2) | ckpt fp16 (convert/deploy) | GGUF F16 (quantize I2_S 1 ở local) | bt data | train.log"
gh release upload "$TAG" --repo "$REPO" --clobber \
  "dist/last_final_${STEP}.pt" \
  "dist/ckpt_${STEP}_fp16.pt" \
  "dist/vija_${STEP}_f16.gguf" \
  "dist/bt_vong1.tar.gz" \
  "dist/train_vong1_${STEP}.log"

say "XONG — https://github.com/$REPO/releases/tag/$TAG"
echo "Về máy local:"
echo "  gh release download $TAG --repo $REPO --pattern 'vija_${STEP}_f16.gguf' --dir dist/"
echo "  ~/BitNet/build/bin/llama-quantize dist/vija_${STEP}_f16.gguf dist/vija_${STEP}_i2s.gguf I2_S 1"
echo "  (rồi chạy gate Vòng 1: probe64 + chrF — PLAN_BUOC5 §2.5/§2.6)"
