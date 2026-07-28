#!/bin/bash
# Theo dõi TỐC ĐỘ STEP tới lúc ổn định (để biết bao giờ xong), rồi tự chấm khi kết thúc.
#
# Vì sao đo tốc độ: ~200 step đầu chậm bất thường (compile torch ~17 phút + warmup),
# lấy tốc độ lúc đó mà ngoại suy sẽ ra ETA sai gấp đôi. Phải chờ ổn định rồi mới tính.
#
# Vì sao chấm ngay khi xong: nếu hết credit giữa chừng, app dừng mà không báo — vẫn
# phải chấm trên checkpoint cuối đã backup về ổ D.
#
#   nohup bash scripts/auto_eval_v6.sh > logs/auto_eval_v6.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
R=logs/KETQUA_V6.txt
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

APP=""
for i in $(seq 1 60); do
  APP=$(grep -oE "ap-[A-Za-z0-9]+" logs/train_v6.log 2>/dev/null | head -1)
  [ -n "$APP" ] && break
  sleep 60
done
say "app=$APP"

# ---------- 1. THEO DÕI TỐC ĐỘ ----------
STEPS=$(grep -oE "\-\-steps [0-9]+" logs/train_v6.log 2>/dev/null | grep -oE "[0-9]+" | head -1)
STEPS=${STEPS:-16400}
say "theo dõi tốc độ tới khi ổn định (đích $STEPS step)"
PREV=0; PREV_T=0; STABLE=0
while true; do
  sleep 300
  modal volume get vija-100m-kd-vol checkpoints_v6/train.log "$V/train_v6.log" --force >/dev/null 2>&1
  CUR=$(grep -oE "step [0-9]+" "$V/train_v6.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  NOW=$(date +%s)
  [ -z "${CUR:-}" ] && { say "chưa có step nào (đang compile)"; continue; }
  if [ "$PREV" -gt 0 ]; then
    DS=$((CUR - PREV)); DT=$((NOW - PREV_T))
    if [ "$DS" -gt 0 ]; then
      SPS=$(python -c "print(f'{$DS/$DT:.3f}')")
      LEFT=$(python -c "
r=$DS/$DT
s=int(($STEPS-$CUR)/r) if r>0 else 0
print(f'{s//3600}g{(s%3600)//60}p')")
      LOSS=$(grep -oE "loss [0-9.]+" "$V/train_v6.log" | tail -1)
      say "step $CUR/$STEPS | $SPS step/s | còn ~$LEFT | $LOSS"
      # ổn định = 3 lần đo liên tiếp đều có tiến triển
      STABLE=$((STABLE + 1))
      [ "$STABLE" = "3" ] && say ">>> TỐC ĐỘ ĐÃ ỔN ĐỊNH: $SPS step/s, dự kiến còn $LEFT <<<"
    fi
  fi
  PREV=$CUR; PREV_T=$NOW
  [ "$CUR" -ge "$STEPS" ] && { say "đạt đích"; break; }
  if modal app logs "$APP" 2>&1 | grep -qE "VÒNG 6 kết thúc|Traceback"; then
    say "app báo kết thúc"; break
  fi
  ST=$(modal app list 2>/dev/null | grep -c "$APP.*\(ephemeral\|running\)" || true)
  [ "$ST" = "0" ] && { say "app đã dừng (hết credit?)"; break; }
done

# ---------- 2. TRUNG BÌNH CHECKPOINT ----------
say "=== TRAIN KẾT THÚC — bắt đầu chấm ==="
{
echo "============ KẾT QUẢ VÒNG 6 ============"
date
echo "--- 10 dòng cuối train.log ---"
modal volume get vija-100m-kd-vol checkpoints_v6/train.log "$V/train_v6.log" --force >/dev/null 2>&1
tail -10 "$V/train_v6.log" 2>/dev/null
} > "$R"

MS=$(modal volume ls vija-100m-kd-vol checkpoints_v6 2>/dev/null \
     | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -7)
[ -z "$MS" ] && MS=$(ls "$V/checkpoints_v6"/step*.pt 2>/dev/null \
     | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -7)
say "average 7 mốc: $MS"
FILES=""
for s in $MS; do
  f="$V/checkpoints_v6/step$s.pt"
  [ -f "$f" ] || modal volume get vija-100m-kd-vol "checkpoints_v6/step$s.pt" "$f" >/dev/null 2>&1
  python -c "
import torch,sys
try: torch.load(r'$D/checkpoints_v6/step$s.pt',map_location='cpu')
except Exception: sys.exit(1)" 2>/dev/null && FILES="$FILES $D/checkpoints_v6/step$s.pt"
done
[ -z "$FILES" ] && { say "KHÔNG có checkpoint hợp lệ — dừng"; exit 1; }
python scripts/avg_ckpt.py $FILES -o "$D/checkpoints_v6/v6_avg.pt" 2>&1 | tail -3 >> "$R"

# ---------- 3. CHẤM ----------
{
echo
echo "===== THƯỚC ĐO 1: dò ngữ pháp 80 phép thử ====="
python scripts/grammar_probe.py "$D/checkpoints_v6/v6_avg.pt" --label v6 2>&1 | tail -14
echo
echo "===== THƯỚC ĐO 2: dịch bench 200 câu THẬT (TED) ====="
python scripts/translate_bench.py "$D/checkpoints_v6/v6_avg.pt" --label v6 2>&1 | tail -3
echo
echo "===== THƯỚC ĐO 3: hardbench (canh hồi quy) ====="
python scripts/hardbench_ckpt.py "$D/checkpoints_v6/v6_avg.pt" --out eval/hb_v6_avg.jsonl 2>&1 | tail -14
} >> "$R" 2>&1

# ---------- 4. BENCH TRONG MIỀN (bài học vòng 5) ----------
say "bench TRONG MIỀN vừa bơm — bench TED một mình sẽ giấu mất cải thiện"
for f in cntt_phan_mem y_te_suc_khoe nong_lam_ngu; do
  [ -f "$D/raw/probe_$f.txt" ] && python scripts/translate_txt.py \
     "$D/checkpoints_v6/v6_avg.pt" "$D/raw/probe_$f.txt" "eval/dom6_$f.jsonl" --label "v6-$f" \
     >> logs/auto_eval_v6.log 2>&1
done
[ -f "$D/raw/heldout_kokkai.txt" ] && python scripts/translate_txt.py \
   "$D/checkpoints_v6/v6_avg.pt" "$D/raw/heldout_kokkai.txt" "eval/bench_held_v6.jsonl" \
   --label v6-held >> logs/auto_eval_v6.log 2>&1

{
echo
echo "CÒN LẠI CẦN NGƯỜI LÀM:"
echo "  1. panel mù v5/v6/google trên 200 câu:"
echo "     python eval/build_panels_bench.py eval/judge_v6 v5=eval/bench_v5.jsonl \\"
echo "        v6=eval/bench_v6.jsonl google=eval/bench_google.jsonl"
echo "  2. so miền: eval/dom6_*.jsonl vs eval/dom_*.jsonl (v5)"
echo "  3. so Quốc hội held-out: eval/bench_held_v6.jsonl vs bench_held_v5.jsonl (v5 đạt 95%)"
echo
echo "MỐC PHẢI VƯỢT (n=200, chấm cùng phiên): v5 66% | ngắn 77% | dài 54% | Google 87%"
echo "MỐC THEO ĐẶC TRƯNG: mệnh đề 50% | slang 45% | zeropron 65% | phủ định 65% | katakana 75%"
} >> "$R"

say "XONG -> $R"
