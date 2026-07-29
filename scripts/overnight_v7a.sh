#!/bin/bash
# CHUỖI ĐÊM V7A-GATE: chờ KD xong -> merge -> binarize (dev SẠCH) -> upload tritue12
# -> train GATE 4.000 step (--dev-every 500) -> backup -> avg mốc cuối -> dịch 2 bench.
# Judge (chấm mù) để NGƯỜI làm sáng hôm sau — quota Gemini đêm đang siết.
#
#   nohup bash scripts/overnight_v7a.sh > logs/overnight_v7a.log 2>&1 &
#
# Bài học nhúng sẵn: poll bằng FILE không bằng ps (ps không thấy tiến trình khác
# session); MODAL_PROFILE=tritue12 từng lệnh (không activate toàn cục); backup lọc
# danh sách tải (KEEP), anchor có kiểm torch.load; đường dẫn D:/ trong python.
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
TODO=623007
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------- 1. chờ KD ----------
say "=== V7A: chờ KD dịch $TODO câu ==="
LAST=0; STALL=0
while true; do
  N=$(grep -c "" "$V/v7a/kd_v7a.jsonl" 2>/dev/null || echo 0)
  say "KD $N/$TODO ($((N*100/TODO))%)"
  [ "$N" -ge "$((TODO*98/100))" ] && { say "đạt 98% — đủ"; break; }
  if [ "$N" -le "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
  # 8 chu kỳ (40') không nhích -> coi như KD chết/hết quota, đi tiếp với phần đã có
  [ "$STALL" -ge 8 ] && { say "KD đứng $((STALL*5)) phút — đi tiếp với $N câu"; break; }
  LAST=$N; sleep 300
done

# ---------- 2. merge (1x, KHÔNG oversample ở gate — tín hiệu sạch) ----------
say "--- merge corpus v6 + kd_v7a ---"
python -u scripts/merge_v6.py --base "$D/kd_v6_merged.jsonl" \
  --new "$D/v7a/kd_v7a.jsonl" --out "$D/kd_v7a_merged.jsonl" 2>&1 | tail -5

# ---------- 3. jsonl -> ja/vi + dev SẠCH -> binarize ----------
say "--- dựng clean_v7g (train=merged, dev=dev SẠCH 3.886) ---"
python - <<'PY'
import json, os
D="D:/Bit-Translate-data"
os.makedirs(f"{D}/clean_v7g", exist_ok=True)
n=0
with open(f"{D}/kd_v7a_merged.jsonl",encoding="utf-8") as f, \
     open(f"{D}/clean_v7g/train.ja","w",encoding="utf-8") as fj, \
     open(f"{D}/clean_v7g/train.vi","w",encoding="utf-8") as fv:
    for line in f:
        try: o=json.loads(line)
        except Exception: continue
        ja,vi=o.get("ja","").strip(),o.get("vi","").strip()
        if ja and vi:
            fj.write(ja+"\n"); fv.write(vi+"\n"); n+=1
print(f"train: {n:,} cap")
for s in ("dev.ja","dev.vi"):
    with open(f"{D}/clean_v7a/{s}",encoding="utf-8") as a, \
         open(f"{D}/clean_v7g/{s}","w",encoding="utf-8") as b:
        b.write(a.read())
print("dev sach: copied")
PY
say "--- binarize -> bin_v7g ---"
python -u scripts/binarize_ja2vi.py --clean "$D/clean_v7g" --out "$D/bin_v7g" 2>&1 | tail -3
[ -f "$V/bin_v7g/train.index.npy" ] || { say "BINARIZE LỖI — DỪNG"; exit 1; }

# ---------- 4. init checkpoint từ v6_avg ----------
say "--- init từ v6_avg (step=0, không optimizer) ---"
mkdir -p "$V/checkpoints_v7a"
python -u scripts/init_round.py "$D/checkpoints_v6/v6_avg.pt" \
  -o "$D/checkpoints_v7a/last.pt" 2>&1 | tail -2

# ---------- 5. upload tritue12 ----------
say "--- upload bin_v7g + checkpoints_v7a lên tritue12 ---"
MODAL_PROFILE=tritue12 modal volume put vija-100m-kd-vol "$D/bin_v7g" bin_v7g --force >/dev/null 2>&1
MODAL_PROFILE=tritue12 modal volume put vija-100m-kd-vol "$D/checkpoints_v7a" checkpoints_v7a --force >/dev/null 2>&1
MODAL_PROFILE=tritue12 modal volume ls vija-100m-kd-vol 2>&1 | tail -4

# ---------- 6. train GATE 4000 step ----------
say "--- TRAIN GATE: 4000 step, lr 5e-5, dev-every 500 ---"
MODAL_PROFILE=tritue12 modal run --detach cloud/modal_train_100m_kd.py::train_v6 \
  --steps 4000 --lr 5e-05 --bin-dir bin_v7g --ckpt-dir checkpoints_v7a \
  > logs/train_v7a_gate.log 2>&1
say "đã phát lệnh train"

# ---------- 7. backup + anchor (profile tritue12) ----------
say "--- bật backup KEEP=12 + anchor %1000 ---"
MODAL_PROFILE=tritue12 KEEP=12 nohup bash scripts/backup_ckpt_loop.sh checkpoints_v7a 300 \
  > logs/backup_v7a.log 2>&1 &
nohup bash scripts/curve_anchor_loop.sh checkpoints_v7a curve_v7a 1000 120 \
  > logs/curve_v7a.log 2>&1 &

# ---------- 8. chờ train xong (poll volume train.log) ----------
say "--- theo dõi train (đích 4000) ---"
while true; do
  sleep 300
  MODAL_PROFILE=tritue12 modal volume get vija-100m-kd-vol checkpoints_v7a/train.log \
    "$V/checkpoints_v7a/train.log" --force >/dev/null 2>&1
  CUR=$(grep -oE "step [0-9]+" "$V/checkpoints_v7a/train.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  DEV=$(grep -oE "dev_loss [0-9.]+" "$V/checkpoints_v7a/train.log" 2>/dev/null | tail -1)
  say "gate step ${CUR:-?}/4000 | ${DEV:-chưa có dev}"
  [ "${CUR:-0}" -ge 4000 ] && break
  grep -q "training loop exited" "$V/checkpoints_v7a/train.log" 2>/dev/null && break
done
say "TRAIN GATE XONG"

# ---------- 9. avg 4 mốc cuối + dịch 2 bench (PyTorch local, TUẦN TỰ) ----------
R=logs/KETQUA_V7A_GATE.txt
{ echo "===== GATE V7A ====="; date
  echo "--- dev_loss theo step (dev SẠCH 3.886) ---"
  grep -E "DEV step" "$V/checkpoints_v7a/train.log" 2>/dev/null
} > "$R"
FILES=""
for s in 3250 3500 3750 4000; do
  f="$V/checkpoints_v7a/step$s.pt"
  [ -f "$f" ] || MODAL_PROFILE=tritue12 modal volume get vija-100m-kd-vol "checkpoints_v7a/step$s.pt" "$f" >/dev/null 2>&1
  python -c "import torch;torch.load(r'$D/checkpoints_v7a/step$s.pt',map_location='cpu')" 2>/dev/null \
    && FILES="$FILES $D/checkpoints_v7a/step$s.pt"
done
python scripts/avg_ckpt.py $FILES -o "$D/checkpoints_v7a/gate_avg.pt" 2>&1 | tail -2 >> "$R"
say "--- dịch bench opus100 + rand bằng gate_avg (PyTorch, 5 luồng) ---"
OMP_NUM_THREADS=5 python scripts/translate_bench.py "$D/checkpoints_v7a/gate_avg.pt" \
  --label gateopus --probe eval/bench_opus100.jsonl >> logs/overnight_v7a.log 2>&1
OMP_NUM_THREADS=5 python scripts/translate_bench.py "$D/checkpoints_v7a/gate_avg.pt" \
  --label gaterand --probe eval/bench_rand.jsonl >> logs/overnight_v7a.log 2>&1
{ echo; echo "SÁNG DẬY CẦN LÀM (chấm mù cùng phiên với baseline):"
  echo "  panel: build_panels (bản _opus/_rand ở scratchpad) gate=eval/bench_gateopus.jsonl vs opuspt/opusgguf/google"
  echo "  LUẬT: long-OOD >=78 -> 18L tiếp | <75 + dev giảm -> thêm liều | <75 + dev phẳng -> GROW 24L"
} >> "$R"
say "=== CHUỖI V7A-GATE HOÀN TẤT — xem $R ==="
