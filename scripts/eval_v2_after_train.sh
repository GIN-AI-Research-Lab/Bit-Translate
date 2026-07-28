#!/bin/bash
# Chuỗi eval SAU khi train_v2 xong (vòng enrich niche 8,39%).
# Chạy: bash scripts/eval_v2_after_train.sh
#
# Vì sao average lại: avg5 của vòng 1 được +0,7 chrF miễn phí (44,4 vs 43,7).
# Vòng 2 có milestone mỗi 500 step -> average 3 mốc cuối (3000/3500/4000).
# Vì sao KHÔNG average cả dải: avg8 (kéo về lúc LR còn cao) TỆ hơn ckpt đơn.
set -e
cd "$(dirname "$0")/.."
D=D:/Bit-Translate-data
V=/d/Bit-Translate-data
export PYTHONUTF8=1

echo "=== 1) Tải milestone vòng 2 từ Modal ==="
mkdir -p "$V/checkpoints_v2"
for s in 3000 3500 4000; do
  f="$V/checkpoints_v2/v2_step$s.pt"
  if [ ! -f "$f" ]; then
    modal volume get vija-100m-kd-vol "checkpoints_v2/step$s.pt" "$f" 2>&1 | tail -1
  fi
  # modal volume get có thể ra file corrupt IM LẶNG -> luôn kiểm tra
  python -c "
import torch,sys
try:
    ck=torch.load(r'$D/checkpoints_v2/v2_step$s.pt',map_location='cpu')
    print('  OK step', ck.get('step'))
except Exception as e:
    sys.exit(f'  CORRUPT: {str(e)[:80]} -> xoá file và tải lại')
"
done

echo "=== 2) Average 3 milestone cuối ==="
python scripts/avg_ckpt.py \
  "$D/checkpoints_v2/v2_step3000.pt" \
  "$D/checkpoints_v2/v2_step3500.pt" \
  "$D/checkpoints_v2/v2_step4000.pt" \
  -o "$D/checkpoints_v2/v2_avg3.pt"

echo "=== 3) chrF hardbench: step4000 đơn lẻ vs avg3 ==="
python scripts/hardbench_ckpt.py "$D/checkpoints_v2/v2_step4000.pt" --out eval/hb_v2_step4000.jsonl 2>&1 | tail -15
python scripts/hardbench_ckpt.py "$D/checkpoints_v2/v2_avg3.pt"     --out eval/hb_v2_avg3.jsonl    2>&1 | tail -15

echo "=== 4) Dựng panel mù cho bản tốt hơn (sửa tay nếu avg3 thắng) ==="
echo "  python eval/build_panels_ja2vi.py eval/hb_v2_avg3.jsonl eval/judge_v2"
echo "  -> rồi CHẤM LẠI CẢ avg5 VÀ v2 TRONG CÙNG PHIÊN (thang judge không hiệu chuẩn"
echo "     giữa các phiên; mốc cũ ở eval/judge_avg5/BASELINE_avg5.json)"
echo
echo "So sánh cần nhìn: 4 domain đã bơm data (thanhngu 1,8 / caudai 2,0 / slang 2,4 /"
echo "keigo 2,7) có lên không, và 5 domain đang mạnh (solieu 4,8 / nguphap 3,8 /"
echo "it_deep 3,4 / zeropronoun 3,3 / hoithoai 3,0) có tụt không."
