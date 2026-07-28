#!/bin/bash
# Tự động chạy khi train vòng 3 kết thúc (hoặc dừng vì hết credit).
# Không cần người trực: chờ -> average milestone cuối -> chrF -> đo layer mới.
# Kết quả gom vào logs/KETQUA_V3.txt để đọc một lần.
#
#   nohup bash scripts/auto_eval_v3.sh ap-XXXX > logs/auto_eval_v3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
APP="${1:?can app id}"
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
R=logs/KETQUA_V3.txt
export PYTHONUTF8=1

echo "[auto] chờ train kết thúc..."
# Chờ tới khi app hết task đang chạy (kết thúc bình thường HOẶC hết credit).
while true; do
  if modal app logs "$APP" 2>&1 | grep -qE "VÒNG 3 kết thúc|Traceback"; then break; fi
  # phòng khi app bị kill đột ngột: kiểm tra còn task không
  ST=$(modal app list 2>/dev/null | grep "$APP" | grep -cE "ephemeral|running" || true)
  if [ "$ST" = "0" ]; then echo "[auto] app không còn chạy (có thể hết credit)"; break; fi
  sleep 300
done

{
echo "==================== KẾT QUẢ VÒNG 3 (grow 18L) ===================="
date
echo
echo "--- 8 dòng cuối train.log ---"
modal volume get vija-100m-kd-vol checkpoints_v3/train.log "$V/train_v3.log" --force >/dev/null 2>&1
tail -8 "$V/train_v3.log" 2>/dev/null
echo
} > "$R"

# Lấy 3 milestone CAO NHẤT hiện có (linh hoạt: nếu hết credit sớm vẫn dùng được)
MS=$(modal volume ls vija-100m-kd-vol checkpoints_v3 2>/dev/null \
     | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -3)
echo "[auto] milestone dùng để average: $MS"
FILES=""
for s in $MS; do
  f="$V/checkpoints_v3/step$s.pt"
  [ -f "$f" ] || modal volume get vija-100m-kd-vol "checkpoints_v3/step$s.pt" "$f" >/dev/null 2>&1
  python -c "
import torch,sys
try: torch.load(r'$D/checkpoints_v3/step$s.pt',map_location='cpu')
except Exception: sys.exit(1)" 2>/dev/null && FILES="$FILES $D/checkpoints_v3/step$s.pt"
done
LAST=$(echo $MS | awk '{print $NF}')

{
echo "--- AVERAGE $MS ---"
python scripts/avg_ckpt.py $FILES -o "$D/checkpoints_v3/v3_avg.pt" 2>&1 | tail -3
echo
echo "--- chrF: step$LAST đơn lẻ ---"
python scripts/hardbench_ckpt.py "$D/checkpoints_v3/step$LAST.pt" --out eval/hb_v3_last.jsonl 2>&1 | tail -14
echo
echo "--- chrF: v3_avg ---"
python scripts/hardbench_ckpt.py "$D/checkpoints_v3/v3_avg.pt" --out eval/hb_v3_avg.jsonl 2>&1 | tail -14
echo
echo "--- CHỈ SỐ QUYẾT ĐỊNH: 6 block mới có học không? ---"
python scripts/check_new_layers.py "$D/checkpoints_v3/v3_avg.pt" --new 2,5,8,11,14,17 2>&1 | tail -26
echo
echo "--- So sánh chrF theo domain: v2_avg3 -> v3_avg ---"
python -c "
import json
from collections import defaultdict
def dom(p):
    d=defaultdict(list)
    for l in open(p,encoding='utf-8'):
        r=json.loads(l); d[r['domain']].append(r['chrf'])
    t=sum(sum(v) for v in d.values())/sum(len(v) for v in d.values())
    return {k:sum(v)/len(v) for k,v in d.items()}, t
a,at=dom('eval/hb_v2_avg3.jsonl'); b,bt=dom('eval/hb_v3_avg.jsonl')
BUM={'thanhngu','slang','keigo','caudai','hoithoai','hop','zeropronoun'}
print(f'{\"domain\":12} {\"v2\":>6} {\"v3\":>6} {\"delta\":>7}  bơm data vòng 3?')
for k in sorted(a,key=lambda x:-(b[x]-a[x])):
    print(f'{k:12} {a[k]:6.1f} {b[k]:6.1f} {b[k]-a[k]:+7.1f}  {\"CÓ\" if k in BUM else \"-\"}')
print(f'{\"TỔNG\":12} {at:6.1f} {bt:6.1f} {bt-at:+7.1f}')
" 2>&1
echo
echo "--- Dựng panel mù cho judge (chấm khi user quay lại) ---"
python eval/build_panels_ja2vi.py eval/hb_v3_avg.jsonl eval/judge_v3 2>&1 | tail -2
echo
echo "CÒN LẠI CẦN NGƯỜI LÀM: judge mù 100 câu (eval/judge_v3/panels/),"
echo "so với mốc vòng 2 = acc 3,82 / 65% dùng được (eval/judge_v2/RESULT_v2.json)."
} >> "$R" 2>&1

echo "[auto] XONG. Kết quả ở $R"
