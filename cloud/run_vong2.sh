#!/usr/bin/env bash
# Vòng 2 — train TIẾP từ step 19000 với data ĐÃ TRỘN (mix_and_binarize) + RESTART LR.
# Chạy nền:  nohup bash cloud/run_vong2.sh > checkpoints/vong2.log 2>&1 &
# Tự resume nếu crash; dừng khi đạt max-steps hoặc gặp checkpoints/PAUSED.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Python (torch+cuda). Shell nền không source profile nên tự thêm PATH conda.
export PATH=/opt/conda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume ở 19000; +5000 -> 23000. LR RESTART 1e-4 -> cosine 1e-5 (anchor=19000)
# nhờ --lr-anchor (nếu không có, schedule tuyệt đối sẽ cho LR ~min => không học).
# max-tokens 4096 * grad-accum 32 = 131072 token/step (giống hệt cấu hình gốc).
# Node 16GB muốn nhanh hơn: VONG2_MT=8192 VONG2_GA=16.
STEPS=${VONG2_STEPS:-23000}
ANCHOR=${VONG2_ANCHOR:-19000}
MT=${VONG2_MT:-4096}
GA=${VONG2_GA:-32}
# torch.compile: mặc định bật. Trên GPU Blackwell (RTX 50) nếu Triton lỗi thì
# chạy với VONG2_COMPILE=0 để tắt (chậm ~30% nhưng chắc chạy).
COMPILE=""; [ "${VONG2_COMPILE:-1}" = "1" ] && COMPILE="--compile"
ARGS="--max-tokens $MT --grad-accum $GA --max-steps $STEPS --lr-anchor $ANCHOR \
--lr 1e-4 --min-lr 1e-5 --warmup 200 --save-every 100 --milestone-every 1000 --log-every 5 $COMPILE"

mkdir -p checkpoints
if [ ! -f checkpoints/last.pt ]; then echo "THIẾU checkpoints/last.pt — chạy prep_vong2.sh trước"; exit 1; fi

# Chặn 2 launcher/train.py cùng chạy (tranh 1 GPU -> cả hai kẹt không tiến,
# bài học từ lần train 19000). pgrep -c tự match cả tiến trình shell hiện tại
# nên KHÔNG dùng; đếm đúng bằng /proc/<pid>/cmdline.
for pid in $(pgrep -f 'scripts/train.py' 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'scripts/train.py'; then
    echo "!! đã có train.py chạy (PID $pid) -> không launch trùng. Theo dõi: tail -f checkpoints/train.log"
    exit 1
  fi
done

# Log của lần train base (đã chứa 'training loop exited' @19000) sẽ làm vòng lặp
# hiểu nhầm là XONG ngay. Lưu nó sang bên 1 lần, để train.log của Vòng 2 sạch.
if [ ! -f checkpoints/train.prevong2.log ] && [ -f checkpoints/train.log ]; then
  mv checkpoints/train.log checkpoints/train.prevong2.log
fi
rm -f checkpoints/PAUSED
echo "resume từ checkpoints/last.pt -> train tới step $STEPS (anchor $ANCHOR)"

for i in $(seq 1 500); do
  echo ">>> vong2 launch $i $(date -u +%H:%M:%S)" >> checkpoints/train.log
  python3 scripts/train.py $ARGS
  if [ -f checkpoints/PAUSED ]; then echo ">>> PAUSED, dừng launcher" >> checkpoints/train.log; break; fi
  if grep -q "training loop exited" checkpoints/train.log; then echo ">>> VONG1 COMPLETE" >> checkpoints/train.log; break; fi
  echo ">>> crashed lần $i, resume sau 10s" >> checkpoints/train.log
  sleep 10
done
