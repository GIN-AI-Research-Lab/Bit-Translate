#!/bin/bash
# Chạy trọn chuỗi vòng 5 KHÔNG CẦN NGƯỜI TRỰC, canh để xong trước 8h sáng.
#
# Ngân sách thời gian (đo/ước từ các vòng trước):
#   KD chạy thêm      : tham số $1 phút (mặc định 140)
#   gộp + prep + binar: ~90 phút
#   upload Modal      : ~40 phút
#   train             : tham số $2 step (14000 = 5,1h = ~$14 trong ngân sách $30)
#   dự phòng          : 30 phút
#
# Vì sao dừng KD sớm thay vì dịch hết 3,18M câu: dịch hết cần 17 giờ, mà chỉ có 8,3
# giờ tới 8h sáng. v5_todo2.txt ĐÃ XÁO TRỘN nên dừng lúc nào mẫu vẫn cân cả 4 nhóm.
#
#   nohup bash scripts/overnight_v5.sh 140 > logs/overnight_v5.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
KD_MIN="${1:-140}"
STEPS="${2:-14000}"
D=D:/Bit-Translate-data
say() { echo "[$(date +%H:%M:%S)] $*"; }

kill_py() {  # dừng theo tên script, KHÔNG kill bừa python
  powershell -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*$1*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  sleep 6
}

fix_jsonl() {  # kill giữa lúc 60 luồng đang ghi -> dòng cụt; đã vấp 2 lần
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
    print(f"  {p.split('/')[-1]}: {len(rows):,} dòng (bỏ {bad})")
except FileNotFoundError:
    print(f"  {p}: chưa có")
PY
}

say "=== BẮT ĐẦU. KD chạy thêm ${KD_MIN} phút ==="
sleep $((KD_MIN * 60))

say "--- dừng KD + sinh quán ngữ ---"
kill_py run_kd_batch
kill_py gen_idiom_gap
fix_jsonl "$D/raw/kd_v5.jsonl"
fix_jsonl "$D/raw/kd_v5b.jsonl"
fix_jsonl "$D/gen_niche/idiomgap.jsonl"

say "--- gộp corpus v5 ---"
python -u scripts/merge_v5.py 2>&1 | tail -25 || { say "GỘP LỖI — dừng"; exit 1; }

say "--- tách train/dev ---"
python -u scripts/prep_clean_split.py "$D/kd_v5_merged.jsonl" "$D/clean_v5" 2>&1 | tail -3

say "--- binarize ---"
python -u scripts/binarize_ja2vi.py --clean "$D/clean_v5" --out "$D/bin_v5" 2>&1 | tail -3

say "--- checkpoint khởi đầu từ v4_avg5 ---"
python -u scripts/init_round.py "$D/checkpoints_v4/v4_avg5.pt" -o "$D/checkpoints_v5/last.pt" 2>&1 | tail -2

say "--- upload Modal (account tritue12) ---"
modal volume put vija-100m-kd-vol "$D/bin_v5" bin_v5 >/dev/null 2>&1
modal volume put vija-100m-kd-vol "$D/checkpoints_v5" checkpoints_v5 >/dev/null 2>&1
modal volume ls vija-100m-kd-vol 2>&1 | tail -4

say "--- backup checkpoint mỗi 5 phút (nền) ---"
nohup bash scripts/backup_ckpt_loop.sh checkpoints_v5 300 > logs/backup_v5.log 2>&1 &

say "--- TRAIN vòng 5 ---"
modal run --detach cloud/modal_train_100m_kd.py::train_v5 --steps "$STEPS" > logs/train_v5.log 2>&1

say "=== XONG chuỗi. Xem logs/train_v5.log + logs/backup_v5.log ==="
