#!/usr/bin/env bash
# Vòng 2 — đợt 2 (chạy SAU khi quota Gemini reset ~14:00 VN / 07:00 UTC).
# Chạy: bash scripts/run_gen_r2.sh   (từ WSL, tại repo root)
# - 2 key song song, todo ưu tiên pk/g2/id trước (vong2_r2_key{1,2}.json)
# - GEN_REASONING=medium (thinking) — chất lượng cao cho phủ định kép/idiom
# - Xong task chính -> harvest ~1000 idiom gloss (key1)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-$HOME/vija-venv/bin/python}
K1=$(grep "^gemini_key_1=" .env | cut -d= -f2 | tr -d "\r\n ")
K2=$(grep "^gemini_key_2=" .env | cut -d= -f2 | tr -d "\r\n ")
MODEL=${GEN_MODEL:-gemini-flash-lite-latest}   # 3.1 sau reset; đổi nếu cần

echo "[r2] bắt đầu $(date -u +%H:%M) UTC, model=$MODEL"
OPENAI_API_KEY=$K1 GEN_MODEL=$MODEL GEN_TODO=vong2_r2_key1.json GEN_RPM=14 GEN_REASONING=medium \
  $PY -u scripts/gen_vong2.py > /tmp/gen_r2_k1.log 2>&1 &
P1=$!
OPENAI_API_KEY=$K2 GEN_MODEL=$MODEL GEN_TODO=vong2_r2_key2.json GEN_RPM=14 GEN_REASONING=medium \
  $PY -u scripts/gen_vong2.py > /tmp/gen_r2_k2.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[r2] task chính xong $(date -u +%H:%M) UTC -> harvest idiom (key1)"
OPENAI_API_KEY=$K1 GEN_MODEL=$MODEL GEN_RPM=14 GEN_REASONING=medium HARVEST_ROUNDS=3 \
  $PY -u scripts/gen_idiom_harvest.py > /tmp/gen_r2_harvest.log 2>&1
echo "[r2] XONG TẤT CẢ $(date -u +%H:%M) UTC"
echo "--- tổng kết ---"
ls data/synthetic/vong2/ | wc -l
cat data/synthetic/vong2/out_*.jsonl | wc -l
wc -l data/synthetic/gen/idiom_glosses.jsonl 2>/dev/null
