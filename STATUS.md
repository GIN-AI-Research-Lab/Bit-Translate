# Trạng thái dự án — cập nhật 2026-07-12

## Đang chạy: TRAINING (Bước 4)

Model **BitNet b1.58 decoder-only, 109.6M params** đang train from-scratch trên GPU.
- Launcher tự-restart: `scripts/run_training.sh` (nohup, resume từ `checkpoints/last.pt` nếu crash).
- Cấu hình: max_tokens 4096, grad_accum 32, max_steps **14000**, cosine LR 3e-4→3e-5, warmup 1000, **--compile (dynamic=True)**.
- Tốc độ với compile: **~6s/step** (eager ~24-35s) → **~1.5 ngày** (nhanh ~3-4x). Log: `checkpoints/train.log`. Checkpoint: `last.pt` (mỗi 100 step) + `stepN.pt` (mỗi 2000) + lưu-ngay-khi-pause.
- torch.compile: cần `linux-libc-dev` (đã cài) để Triton build; dùng `dynamic=True` để tránh recompile theo shape; ~1-3 phút warmup mỗi lần (re)start; đôi khi 1 step recompile lẻ ~24s — bình thường.
- LƯU Ý: KHÔNG bật `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` (crash "device not ready" trong WSL2). KHÔNG dùng bitsandbytes (cùng lỗi). Dùng AdamW thường.

### ĐIỀU KHIỂN buổi tối / chơi game (3 lệnh chính)
```
bash pause_training.sh     # tạm dừng, lưu checkpoint, giải phóng GPU (chơi game mượt nhất)
bash resume_training.sh    # chạy tiếp full tốc từ checkpoint (khi không chơi)
bash game_mode.sh          # vừa train (~50% GPU, ~3.6GB VRAM) vừa chơi LoL (setting thấp/TB)
```
Checkpoint lưu mỗi 100 step + ngay khi pause → tắt/pause lúc nào cũng KHÔNG mất tiến độ.
Tắt máy hẳn cũng được; tối sau chỉ cần `bash resume_training.sh`.

### Kiểm tra tiến độ
```
tail -5 checkpoints/train.log
grep -E "^step " checkpoints/train.log | tail -3
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# đếm ĐÚNG số python train (pgrep -c tự match shell nên sai):
for p in $(pgrep -x python3); do tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -q scripts/train.py && echo $p; done
```
LƯU Ý: chỉ được có ĐÚNG 1 python train.py. Nếu >1 → chúng tranh GPU, kẹt; kill hết rồi `resume_training.sh`.

### Nếu training dừng hẳn (launcher chết): khởi động lại (tự resume)
```
cd ~/Train-model-translate && nohup bash scripts/run_training.sh > checkpoints/launcher.log 2>&1 &
```

## Khi training xong (log có "TRAINING COMPLETE") — HOẶC muốn thử model giữa chừng

1. **Eval chrF** (nếu training đang chạy, dùng --device cpu để khỏi tranh GPU):
```
source .venv/bin/activate
python3 scripts/evaluate.py --ckpt checkpoints/last.pt --n 300 --device cpu
# hoặc khi GPU rảnh: --device cuda --n 1000
cat eval/history.tsv
```

2. **Đóng gói** model 1.58-bit + chạy thử trên CPU:
```
python3 scripts/export_packed.py --ckpt checkpoints/last.pt   # -> dist/model_1p58.npz (~40-50MB)
python3 scripts/cpu_translate.py --to jpn --text "Xin chào, hôm nay bạn thế nào?"
python3 scripts/cpu_translate.py --to vie --text "おはようございます。今日はいい天気ですね。"
```

## Đã xong
- Bước 1-2: data 7.98M thô → **5.40M sạch** (`data/clean/{train,dev}.{ja,vi}`), FLORES test.
- Bước 3: tokenizer `tokenizer/spm_vija_32k.model` (SPM unigram 32k, thẻ >>vie<</>>jpn<<).
- Bước 4b: binarize `data/bin/` (10.79M chuỗi 2 chiều, 391M token).
- Code: `src/bitnet.py`, `scripts/{train,evaluate,export_packed,cpu_translate}.py`.

## Ghi chú kiến trúc
Decoder-only (không encoder-decoder) để đóng gói được — xem memory `project-step4-decisions`.
Định dạng chuỗi: `[BOS] >>tgt<< nguồn [EOS] đích [EOS]`, loss chỉ trên phần đích.
Đóng gói dùng packed-ternary + CPU runtime tự viết (bitnet.cpp không hỗ trợ arch này).

## Việc còn lại của lộ trình
- Bước 5 vòng lặp data: đọc lỗi bản dịch → bổ sung data nhắm điểm yếu → train lại (3-5 vòng). Cần con người đánh giá.
- Back-translation (Bước 1 mục 4) sau khi có model sơ bộ.
- Ổ C: nếu chật, nén vhdx — xem memory `wsl-disk-reclaim`.
