#!/bin/bash
# Tự chấm khi train vòng 5 kết thúc (hoặc dừng vì hết credit).
#
# ĐO CÁI GÌ — thứ tự ưu tiên rút từ bài học vòng 3-4:
#   1. bench câu THẬT (eval/bench_new.jsonl, 200 câu TED) — thước đo quyết định
#   2. DÒ NGỮ PHÁP 80 phép thử — tự động, so thẳng v3/v4/v5
#   3. hardbench — CHỈ canh hồi quy, KHÔNG dùng kết luận (vòng 3 +18 điểm ở đây
#      mà bench thật đứng yên, vì hardbench do chính pipeline này sinh ra)
#
# Average 7 mốc thay vì 3: vòng 4 đo được avg5 > avg3 trên CẢ hai thước đo, và
# vòng 5 lưu milestone mỗi 250 step nên có ~56 mốc để chọn.
#
#   nohup bash scripts/auto_eval_v5.sh ap-XXXX > logs/auto_eval_v5.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
APP="${1:?can app id}"
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
R=logs/KETQUA_V5.txt
export PYTHONUTF8=1

echo "[auto] chờ train vòng 5 kết thúc..."
while true; do
  if modal app logs "$APP" 2>&1 | grep -qE "VÒNG 5 kết thúc|Traceback"; then break; fi
  ST=$(modal app list 2>/dev/null | grep "$APP" | grep -cE "ephemeral|running" || true)
  if [ "$ST" = "0" ]; then echo "[auto] app dừng (hết credit?)"; break; fi
  sleep 300
done

{
echo "============ KẾT QUẢ VÒNG 5 (data thật miền mới) ============"
date
echo
echo "--- 8 dòng cuối train.log ---"
modal volume get vija-100m-kd-vol checkpoints_v5/train.log "$V/train_v5.log" --force >/dev/null 2>&1
tail -8 "$V/train_v5.log" 2>/dev/null
echo
} > "$R"

# 7 mốc CAO NHẤT; rơi về file đã backup nếu volume không đọc được (hết credit)
MS=$(modal volume ls vija-100m-kd-vol checkpoints_v5 2>/dev/null \
     | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -7)
if [ -z "$MS" ]; then
  echo "[auto] không đọc được volume -> dùng milestone đã backup"
  MS=$(ls "$V/checkpoints_v5"/step*.pt 2>/dev/null \
       | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -7)
fi
echo "[auto] average: $MS"
[ -z "$MS" ] && { echo "[auto] KHÔNG có milestone — dừng"; exit 1; }
FILES=""
for s in $MS; do
  f="$V/checkpoints_v5/step$s.pt"
  [ -f "$f" ] || modal volume get vija-100m-kd-vol "checkpoints_v5/step$s.pt" "$f" >/dev/null 2>&1
  python -c "
import torch,sys
try: torch.load(r'$D/checkpoints_v5/step$s.pt',map_location='cpu')
except Exception: sys.exit(1)" 2>/dev/null && FILES="$FILES $D/checkpoints_v5/step$s.pt"
done

{
echo "--- AVERAGE $MS ---"
python scripts/avg_ckpt.py $FILES -o "$D/checkpoints_v5/v5_avg.pt" 2>&1 | tail -3
echo
echo "===== THƯỚC ĐO 1: DÒ NGỮ PHÁP (tự động) ====="
python scripts/grammar_probe.py "$D/checkpoints_v5/v5_avg.pt" --label v5 2>&1 | tail -40
echo
echo "--- so v3 / v4 / v5 theo ĐỘ DÀI (cùng bộ regex hiện tại) ---"
python -c "
import json, re
from collections import defaultdict
src=open('scripts/grammar_probe.py',encoding='utf-8').read()
src=src.split('# ---------------------------------------------------------------------------\ndef score')[0]
g={'__file__':'scripts/grammar_probe.py','__name__':'gp'}; exec(compile(src,'gp.py','exec'),g)
P={p['ja']:p for p in g['P']}
def t(path):
    d=defaultdict(lambda:[0,0])
    for l in open(path,encoding='utf-8'):
        r=json.loads(l); p=P.get(r['ja'])
        if not p: continue
        ok=(not p['must'] or bool(re.search(p['must'],r['hyp'],re.I))) and \
           (not p['must_not'] or not re.search(p['must_not'],r['hyp'],re.I))
        gr=p['group']
        k=('2.vua' if gr.startswith('mid_') else '3.dai' if gr.startswith('long_')
           else '4.ratdai' if gr.startswith('xlong_') else '1.ngan')
        d[k][0]+=ok; d[k][1]+=1
    return d
print(f\"{'':10}{'ngan':>12}{'KHONG ngan':>14}{'TONG':>12}\")
for n,p in [('v3','eval/probe_v3.jsonl'),('v4','eval/probe_v4.jsonl'),
            ('v4 avg5','eval/probe_v4a5.jsonl'),('v5','eval/probe_v5.jsonl')]:
    try: d=t(p)
    except FileNotFoundError: continue
    s,sn=d['1.ngan']
    ns=sum(v[0] for k,v in d.items() if k!='1.ngan'); nn=sum(v[1] for k,v in d.items() if k!='1.ngan')
    print(f'{n:10}{s:>5}/{sn:<6}{ns:>6}/{nn:<7}{s+ns:>5}/{sn+nn:<5} = {100*(s+ns)/(sn+nn):.0f}%')
" 2>&1
echo
echo "===== THƯỚC ĐO 2: dịch bench 200 câu THẬT (chờ judge) ====="
python scripts/translate_bench.py "$D/checkpoints_v5/v5_avg.pt" --label v5 2>&1 | tail -3
echo
echo "--- canh hồi quy: hardbench (KHÔNG dùng kết luận) ---"
python scripts/hardbench_ckpt.py "$D/checkpoints_v5/v5_avg.pt" --out eval/hb_v5_avg.jsonl 2>&1 | tail -14
echo
echo "CÒN LẠI CẦN LÀM: dựng panel mù v4/v5/google trên CẢ 200 câu rồi chấm"
echo "  python eval/build_panels_bench.py eval/judge_v5 v4=eval/bench_v4.jsonl \\"
echo "      v5=eval/bench_v5.jsonl google=eval/bench_google.jsonl"
echo "  (chấm xong: python eval/aggregate_bench_judge.py eval/judge_v5)"
echo "MỐC PHẢI VƯỢT (n=200, chấm cùng phiên): v4 64% | ngắn 75% | dài 52% | Google 86%"
} >> "$R" 2>&1

echo "[auto] XONG -> $R"
