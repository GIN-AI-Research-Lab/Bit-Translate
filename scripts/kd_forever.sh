#!/bin/bash
# Giữ KD chạy LIÊN TỤC, tự nối sang nguồn kế tiếp khi hết danh sách hiện tại.
# KHÔNG train, KHÔNG dừng theo giờ — user chọn ưu tiên DATA trước.
#
# Vì sao cần: run_kd_batch.py thoát khi dịch hết input. Nếu không ai trực, KD sẽ
# nằm không hàng giờ. Script này canh tiến trình, hết nguồn thì dựng nguồn kế tiếp
# từ phần CHƯA chọn (Quốc hội + CC-100 còn lại) và chạy tiếp.
#
# Thứ tự nguồn (theo giá trị đã đo, cao xuống thấp):
#   1. v5_todo2.txt   — 3,18M câu ĐÃ CHỌN cho vòng 5 (4 nhóm cân bằng, đã xáo)
#   2. v5_phase2.txt  — phần còn lại của Quốc hội + CC-100 khẩu ngữ (dựng khi cần)
#
#   nohup bash scripts/kd_forever.sh > logs/kd_forever.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
D=D:/Bit-Translate-data
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

alive() {
  powershell -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_kd_batch*' } | Measure-Object).Count" 2>/dev/null | tr -d ' \r\n'
}

fix_jsonl() {
  python - "$1" <<'PY'
import json, sys
p = sys.argv[1]; rows = []; bad = 0
try:
    for raw in open(p, "rb"):
        try:
            o = json.loads(raw.decode("utf-8")); assert "ja" in o and "vi" in o
            rows.append(raw)
        except Exception: bad += 1
    open(p, "wb").writelines(r if r.endswith(b"\n") else r + b"\n" for r in rows)
    print(f"  {p.split('/')[-1]}: {len(rows):,} dòng (bỏ {bad} hỏng)")
except FileNotFoundError:
    pass
PY
}

start_kd() {  # $1 = input, $2 = output
  say "khởi động KD: $(basename "$1") -> $(basename "$2")"
  KD_INPUT="$1" KD_OUTPUT="$2" WORKERS_PER_KEY=10 BATCH_SIZE=2 MAX_HANDSHAKE=16 \
    nohup python -u scripts/run_kd_batch.py > "logs/kd_$(basename "$2" .jsonl).log" 2>&1 &
  sleep 30
}

build_phase2() {
  say "dựng nguồn giai đoạn 2 (phần CHƯA chọn)"
  python - <<'PY'
import json, random
from pathlib import Path
D = Path("D:/Bit-Translate-data")
used = set()
for p in [D/"raw"/"v5_select.txt"]:
    for l in p.open(encoding="utf-8"): used.add(l.strip())
for p in [D/"raw"/"kd_v5.jsonl", D/"raw"/"kd_v5b.jsonl", D/"raw"/"kd_cc100.jsonl"]:
    if p.exists():
        for l in p.open(encoding="utf-8"):
            try: used.add(json.loads(l)["ja"])
            except Exception: pass
out = []
for p, lo, hi in [(D/"raw"/"kokkai_ja.txt", 40, 220),
                  (D/"raw"/"cc100_colloq.txt", 15, 60)]:
    if not p.exists(): continue
    for l in p.open(encoding="utf-8", errors="ignore"):
        s = l.strip()
        if lo <= len(s) < hi and s not in used:
            out.append(s); used.add(s)
random.Random(99).shuffle(out)
(D/"raw"/"v5_phase2.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"  v5_phase2.txt: {len(out):,} câu")
PY
}

say "=== GIỮ KD CHẠY LIÊN TỤC (không train) ==="
while true; do
  sleep 300
  if [ "$(alive)" != "0" ]; then continue; fi

  say "KD đã dừng — kiểm tra nguồn kế tiếp"
  fix_jsonl "$D/raw/kd_v5b.jsonl"
  fix_jsonl "$D/raw/kd_v5c.jsonl"

  if [ ! -f "$D/raw/v5_phase2.txt" ]; then
    build_phase2
  fi
  REM=$(python -c "
import json,sys
from pathlib import Path
D=Path('D:/Bit-Translate-data')
done=set()
for p in [D/'raw'/'kd_v5c.jsonl']:
    if p.exists():
        for l in p.open(encoding='utf-8'):
            try: done.add(json.loads(l)['ja'])
            except Exception: pass
n=sum(1 for l in (D/'raw'/'v5_phase2.txt').open(encoding='utf-8') if l.strip() and l.strip() not in done)
print(n)")
  say "giai đoạn 2 còn $REM câu"
  if [ "$REM" -lt 1000 ]; then
    say "HẾT NGUỒN — dừng vòng lặp"; break
  fi
  start_kd "$D/raw/v5_phase2.txt" "$D/raw/kd_v5c.jsonl"
done
say "=== KẾT THÚC ==="
