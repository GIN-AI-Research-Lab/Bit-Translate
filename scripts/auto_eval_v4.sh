#!/bin/bash
# Tự động chạy khi train vòng 4 kết thúc (hoặc dừng vì hết credit).
#
# KHÁC auto_eval_v3.sh ở chỗ ĐO CÁI GÌ. Vòng 3 đo bằng hardbench và thấy +18 điểm,
# nhưng hardbench do chính pipeline này sinh ra nên nó luôn đẹp lên — bench câu THẬT
# thì đứng yên (v3 66% = v2 66%). Vòng 4 vì vậy lấy hai thước đo ĐỘC LẬP làm chính:
#
#   1. bench câu THẬT (eval/bench_new.jsonl, 200 câu TED chưa từng vào train)
#      -> chỉ sinh bản dịch, judge mù chấm sau (cùng phiên với v3 để không lệch thang)
#   2. DÒ NGỮ PHÁP (scripts/grammar_probe.py, 54 cặp tối thiểu)
#      -> chấm bằng regex, tự động, so thẳng được với mốc v3
#
# hardbench vẫn chạy nhưng chỉ để canh hồi quy, KHÔNG dùng để kết luận.
#
#   nohup bash scripts/auto_eval_v4.sh ap-XXXX > logs/auto_eval_v4.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
APP="${1:?can app id}"
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
R=logs/KETQUA_V4.txt
export PYTHONUTF8=1

echo "[auto] chờ train vòng 4 kết thúc..."
while true; do
  if modal app logs "$APP" 2>&1 | grep -qE "VÒNG 4 kết thúc|Traceback"; then break; fi
  ST=$(modal app list 2>/dev/null | grep "$APP" | grep -cE "ephemeral|running" || true)
  if [ "$ST" = "0" ]; then echo "[auto] app không còn chạy (có thể hết credit)"; break; fi
  sleep 300
done

{
echo "==================== KẾT QUẢ VÒNG 4 (data theo danh sách phủ) ===================="
date
echo
echo "--- 8 dòng cuối train.log ---"
modal volume get vija-100m-kd-vol checkpoints_v4/train.log "$V/train_v4.log" --force >/dev/null 2>&1
tail -8 "$V/train_v4.log" 2>/dev/null
echo
} > "$R"

# Lấy milestone từ VOLUME; nếu hết credit thì `modal volume ls` cũng chết theo, nên
# rơi về danh sách file ĐÃ BACKUP dưới ổ D (backup_ckpt_loop.sh kéo mỗi 5 phút).
MS=$(modal volume ls vija-100m-kd-vol checkpoints_v4 2>/dev/null \
     | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -3)
if [ -z "$MS" ]; then
  echo "[auto] không đọc được volume (hết credit?) -> dùng milestone đã backup ở $V"
  MS=$(ls "$V/checkpoints_v4"/step*.pt 2>/dev/null \
       | grep -oE 'step[0-9]+\.pt' | sed 's/step//;s/\.pt//' | sort -n | tail -3)
fi
echo "[auto] milestone dùng để average: $MS"
[ -z "$MS" ] && { echo "[auto] KHÔNG có milestone nào — dừng."; exit 1; }
FILES=""
for s in $MS; do
  f="$V/checkpoints_v4/step$s.pt"
  [ -f "$f" ] || modal volume get vija-100m-kd-vol "checkpoints_v4/step$s.pt" "$f" >/dev/null 2>&1
  python -c "
import torch,sys
try: torch.load(r'$D/checkpoints_v4/step$s.pt',map_location='cpu')
except Exception: sys.exit(1)" 2>/dev/null && FILES="$FILES $D/checkpoints_v4/step$s.pt"
done

{
echo "--- AVERAGE $MS ---"
python scripts/avg_ckpt.py $FILES -o "$D/checkpoints_v4/v4_avg.pt" 2>&1 | tail -3
echo
echo "=========== THƯỚC ĐO 1: DÒ NGỮ PHÁP (tự động, so thẳng với v3) ==========="
python scripts/grammar_probe.py "$D/checkpoints_v4/v4_avg.pt" --label v4 2>&1 | tail -60
echo
echo "--- so nhóm ngữ pháp v3 -> v4 ---"
python -c "
import json
from collections import defaultdict
def load(p):
    d=defaultdict(lambda:[0,0])
    for l in open(p,encoding='utf-8'):
        r=json.loads(l); d[r['group']][0]+=r['ok']; d[r['group']][1]+=1
    return d
a,b=load('eval/probe_v3.jsonl'),load('eval/probe_v4.jsonl')
print(f'{\"nhóm\":14} {\"v3\":>7} {\"v4\":>7} {\"delta\":>7}')
ta=tb=n=0
for k in sorted(a,key=lambda x:(b[x][0]/b[x][1])-(a[x][0]/a[x][1])):
    ta+=a[k][0]; tb+=b[k][0]; n+=a[k][1]
    print(f'{k:14} {a[k][0]:3}/{a[k][1]:<3} {b[k][0]:3}/{b[k][1]:<3} {b[k][0]-a[k][0]:+7}')
print(f'{\"TỔNG\":14} {ta:3}/{n:<3} {tb:3}/{n:<3} {tb-ta:+7}')
" 2>&1
echo
echo "=========== THƯỚC ĐO 2: bench câu THẬT (sinh bản dịch, chờ judge) ==========="
python scripts/translate_bench.py "$D/checkpoints_v4/v4_avg.pt" --label v4 2>&1 | tail -4
echo
echo "--- canh hồi quy: hardbench (KHÔNG dùng để kết luận) ---"
python scripts/hardbench_ckpt.py "$D/checkpoints_v4/v4_avg.pt" --out eval/hb_v4_avg.jsonl 2>&1 | tail -14
echo
echo "--- so hardbench theo domain: v3_avg -> v4_avg ---"
python -c "
import json
from collections import defaultdict
def dom(p):
    d=defaultdict(list)
    for l in open(p,encoding='utf-8'):
        r=json.loads(l); d[r['domain']].append(r['chrf'])
    t=sum(sum(v) for v in d.values())/sum(len(v) for v in d.values())
    return {k:sum(v)/len(v) for k,v in d.items()}, t
a,at=dom('eval/hb_v3_avg.jsonl'); b,bt=dom('eval/hb_v4_avg.jsonl')
print(f'{\"domain\":12} {\"v3\":>6} {\"v4\":>6} {\"delta\":>7}')
for k in sorted(a,key=lambda x:-(b[x]-a[x])):
    print(f'{k:12} {a[k]:6.1f} {b[k]:6.1f} {b[k]-a[k]:+7.1f}')
print(f'{\"TỔNG\":12} {at:6.1f} {bt:6.1f} {bt-at:+7.1f}')
" 2>&1
echo
echo "CÒN LẠI CẦN LÀM: judge mù v3 vs v4 vs google trên CẢ 200 câu bench_new.jsonl,"
echo "chấm trong CÙNG một phiên (n=50 cũ quá ít: sai số ±13% nên +8 điểm không phân biệt"
echo "được với nhiễu). Mốc v3 hiện tại: 66% đúng nghĩa (short 80% / long 57%)."
} >> "$R" 2>&1

echo "[auto] XONG. Kết quả ở $R"
