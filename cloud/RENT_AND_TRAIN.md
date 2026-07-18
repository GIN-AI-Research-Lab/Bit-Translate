# Thuê node cloud → train tiếp 292M (checklist đã xác minh 2026-07-18)

> Local 3060 Ti 8GB quá chật cho 292M (static 4.67GB + compile/phân mảnh chạm trần
> → treo/thrash, chỉ chạy nổi eager ~27s/step). Cloud 12GB+ chạy compile ~4.78s/step
> (nhanh ~5.6×). Đây là hướng đúng cho model này.

## Trạng thái đã sẵn sàng (không cần làm gì thêm ở nhà)
- Checkpoint mới nhất **step 11425** đã nằm trên release `autosave-scale300m` (last_a, 3.5GB, đã validate). Cloud tự tải + resume từ đây.
- Data premix (base+vòng1+vòng2, 451MB) trên `train-assets-vong2`. Cloud tự tải.
- Guard 110M (`last_final_23000.pt`) có trên `vong2-step23000` → run_300m.sh không bị chặn.
- gh ở máy này đang đăng nhập `trituenguyen97` → lấy token được ngay.

## Bước 1 — Thuê node
- Template/image: **có sẵn PyTorch + CUDA** (bắt buộc — torch ~5GB không đóng gói được).
- VRAM: **≥12GB** (đã kiểm chứng 4070 Ti 12GB @ MT4096; 16GB A4000 càng thoải mái).
- torch trong khoảng 2.4–2.6 là chuẩn; ngoài dải vẫn thử được (bootstrap sẽ cảnh báo).
- GPU Blackwell/RTX 50xx: thêm env `M300_COMPILE=0` (Triton hay lỗi trên kiến trúc mới).

## Bước 2 — Lấy token GitHub (chạy Ở MÁY NÀY)
```bash
gh auth token          # copy chuỗi in ra (KHÔNG chia sẻ công khai)
```

## Bước 3 — Trên node mới, dán 5 dòng (thay <token>)
```bash
export GH_TOKEN=<token>
curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz | tar xz -C /tmp
/tmp/gh_2.63.0_linux_amd64/bin/gh repo clone trituenguyen97/Bit-Translate bt && cd bt
bash cloud/bootstrap.sh
```
`bootstrap.sh` tự làm hết (idempotent): cài gh+gcc+zstd, kiểm torch/GPU, tải data premix,
**restore checkpoint step 11425**, rồi launch: train + backup daemon (autosave 30'/lần lên
`autosave-scale300m`) + watcher (đóng gói khi đạt 25000).

Env tuỳ chọn trước lệnh cuối: `M300_MT=4096 M300_GA=32` (mặc định, cho 12GB) — node 16GB
có thể thử `M300_MT=6144 M300_GA=22` (giữ ~131k token/step) cho nhanh hơn chút.

## Bước 4 — Theo dõi
```bash
tail -f checkpoints/train.log          # compile warmup ~10-20' rồi mới có step đầu
bash watch.sh                          # bảng tiến độ (đích 25000)
```
Kỳ vọng: sau warmup ~**4.78s/step**; còn ~13.575 step (11425→25000) ≈ **~18h**.

## Bước 5 — Lấy kết quả về (khi log có `training loop exited`, hoặc bất cứ lúc nào)
```bash
python3 scripts/convert_to_gguf.py --ckpt checkpoints/last.pt --out dist/x_f32.gguf
# (converter tự đọc dims 292M từ ckpt)
```
Tải về: `checkpoints/last.pt` (train tiếp) hoặc gguf đã đóng gói (chạy CPU).

## Lưu ý sống còn
- **Tính tiền theo phút, hết số dư = node bị xoá.** backup_300m.sh đã autosave mỗi 30'
  lên `autosave-scale300m` (2 bộ a/b luân phiên) → node chết vẫn không mất quá 30' tiến độ;
  node mới lặp lại Bước 3 là tự resume. Nhưng vẫn nên giữ đủ tiền + để chạy tới xong.
- **Đừng dùng cấu hình LOCAL trên cloud** (eager/MT2048) — cloud dùng run_300m.sh mặc định
  (compile + MT4096) là đúng và nhanh.
- Pause nghỉ: `touch checkpoints/PAUSED` → launcher dừng sau step hiện tại; chạy lại
  `bash cloud/run_300m.sh` để tiếp.

## (Tuỳ chọn) tối ưu tốc độ đã viết ở local — CHƯA đưa lên cloud
`src/bitnet.py`+`scripts/train.py` local có 4 opt (STE Function, cache ternary weight,
gộp GEMM, masked-CE) đã chứng minh tương đương toán học (parity Δloss=0), bật qua env
`BITNET_OPT`. Trên compile chúng chỉ +~4% và maskce (nonzero) có rủi ro recompile — nên
**mặc định KHÔNG dùng trên cloud**; giữ config gốc đã kiểm chứng 4.78s/step. Nếu muốn thử
sau: push code local lên rồi đặt `BITNET_OPT=ste,fusedproj` (tránh maskce/wqcache) + đo lại.
