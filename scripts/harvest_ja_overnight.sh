#!/usr/bin/env bash
# Chạy harvest_ja_indomain nhiều vòng qua đêm (Qiita unauth 60 req/h -> mỗi vòng
# 12 tag × 5 trang = 60 req, nghỉ 61' chờ reset). Mỗi vòng hút DẢI TRANG MỚI.
#   nohup bash scripts/harvest_ja_overnight.sh > /tmp/harvest_ja.log 2>&1 &
# Env: ROUNDS (mặc định 10), PAGES_PER_ROUND (5)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROUNDS=${ROUNDS:-10}
PAGES=${PAGES_PER_ROUND:-5}
export GITHUB_TOKEN=${GITHUB_TOKEN:-$(gh auth token 2>/dev/null || true)}
PY=${PY:-.venv/bin/python}

for r in $(seq 1 "$ROUNDS"); do
  start=$(( (r - 1) * PAGES + 1 ))
  echo ">>> vòng $r/$ROUNDS — trang $start..$((start + PAGES - 1)) $(date +%H:%M)"
  QIITA_PAGES=$PAGES QIITA_PAGE_START=$start $PY -u scripts/harvest_ja_indomain.py
  n=$(wc -l < data/synthetic/ja_indomain_bt.ja)
  echo ">>> sau vòng $r: $n câu $(date +%H:%M)"
  [ "$r" -lt "$ROUNDS" ] && sleep 3660   # 61' chờ reset rate limit Qiita
done
echo ">>> HARVEST ĐÊM XONG: $(wc -l < data/synthetic/ja_indomain_bt.ja) câu"
