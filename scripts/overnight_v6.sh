#!/bin/bash
# Chuỗi vòng 6 chạy KHÔNG CẦN NGƯỜI TRỰC: chờ KD xong -> gộp -> binarize -> upload
# -> train (có backup) -> tự đánh giá.
#
# Khác vòng 5: KHÔNG cắt KD theo đồng hồ. Vòng 5 phải dừng sớm vì deadline 8h sáng;
# lần này chờ dịch HẾT 2,20M câu (đo được 98 câu/s -> ~6,2 giờ) rồi mới đi tiếp,
# vì mỗi mặt trận lỗi cần đủ hạn mức mới có tác dụng — dừng non thì mặt trận nào
# cũng dở dang.
#
#   nohup bash scripts/overnight_v6.sh > logs/overnight_v6.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
D=D:/Bit-Translate-data
STEPS="${1:-16400}"
TODO=$(grep -c "" "$D/raw/v6_todo.txt")
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

kill_py() {
  powershell -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*$1*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  sleep 6
}

fix_jsonl() {  # 120 luồng ghi song song -> kill/crash để lại dòng cụt. Đã vấp 3 lần.
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
    print(f"  {p}: chưa có")
PY
}

alive() {
  powershell -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*run_kd_batch*' } | Measure-Object).Count" 2>/dev/null | tr -d ' \r\n'
}

say "=== VÒNG 6 — chờ KD dịch hết $TODO câu ==="
LAST=0; STALL=0
while true; do
  sleep 300
  N=$(grep -c "" "$D/raw/kd_v6.jsonl" 2>/dev/null || echo 0)
  PCT=$((N * 100 / TODO))
  say "KD $N/$TODO ($PCT%)"
  if [ "$(alive)" = "0" ]; then say "tiến trình KD đã thoát"; break; fi
  if [ "$N" -ge "$((TODO * 98 / 100))" ]; then say "đã đạt 98% — đủ, đi tiếp"; break; fi
  # canh treo: 3 lần liên tiếp không tăng (25 phút) -> coi như hỏng, đi tiếp với data có
  if [ "$N" -le "$LAST" ]; then STALL=$((STALL + 1)); else STALL=0; fi
  if [ "$STALL" -ge 5 ]; then say "KD ĐỨNG YÊN 25 phút — đi tiếp với $N câu"; break; fi
  LAST=$N
done

say "--- dừng KD, vá file ---"
kill_py run_kd_batch
fix_jsonl "$D/raw/kd_v6.jsonl"

say "--- gộp corpus v6 ---"
python -u scripts/merge_v6.py 2>&1 | tail -12 || { say "GỘP LỖI — DỪNG"; exit 1; }

say "--- tách train/dev ---"
python -u scripts/prep_clean_split.py "$D/kd_v6_merged.jsonl" "$D/clean_v6" 2>&1 | tail -3

say "--- binarize ---"
python -u scripts/binarize_ja2vi.py --clean "$D/clean_v6" --out "$D/bin_v6" 2>&1 | tail -4
[ -f "$D/bin_v6/train.index.npy" ] || { say "BINARIZE LỖI — DỪNG"; exit 1; }

say "--- checkpoint khởi đầu từ v5_avg ---"
mkdir -p "$D/checkpoints_v6"
python -u scripts/init_round.py "$D/checkpoints_v5/v5_avg.pt" -o "$D/checkpoints_v6/last.pt" 2>&1 | tail -2

say "--- upload Modal ---"
modal volume put vija-100m-kd-vol "$D/bin_v6" bin_v6 >/dev/null 2>&1
modal volume put vija-100m-kd-vol "$D/checkpoints_v6" checkpoints_v6 >/dev/null 2>&1
modal volume ls vija-100m-kd-vol 2>&1 | tail -5

say "--- BẬT BACKUP checkpoint mỗi 5 phút về ổ D (phòng hết credit) ---"
nohup bash scripts/backup_ckpt_loop.sh checkpoints_v6 300 > logs/backup_v6.log 2>&1 &

say "--- TRAIN vòng 6: $STEPS step ---"
modal run --detach cloud/modal_train_100m_kd.py::train_v6 --steps "$STEPS" > logs/train_v6.log 2>&1
say "đã phát lệnh train"

say "--- bật theo dõi tốc độ + tự đánh giá khi xong ---"
nohup bash scripts/auto_eval_v6.sh > logs/auto_eval_v6.log 2>&1 &

say "=== CHUỖI ĐÃ CHẠY HẾT. Xem logs/train_v6.log, logs/backup_v6.log ==="
