# STATUS — BitNet 1.58-bit VI↔JA (cập nhật 2026-07-13)

## TL;DR
- **Model TRAIN ĐÚNG và DỊCH TỐT** (đã chứng minh trong PyTorch): `猫が好きです。→Tôi thích mèo.` hoàn hảo; step 1200/loss 3.37 đúng 5/6 câu. Tốc độ CPU đo được **465 tok/s** (5600X, 6 luồng, i2_s).
- **NÚT THẮT DUY NHẤT chưa giải:** đóng gói sang `bitnet.cpp` thì model **CÂM** (nhả EOS ngay) với MỌI quant (i2_s / Q8_0 / F16). Model 2B-4T chính chủ chạy tốt trên cùng build ⇒ runtime OK, lỗi ở tương tác model-tự-train ↔ bitnet.cpp.

## Đã làm xong
1. **Data (Bước 1-3):** 5.40M cặp sạch ở `data/clean`, tokenizer SPM 32k, binarize → `data/bin`.
2. **Arch (Bước 4):** decoder-only BitNet b1.58, **FFN squared-ReLU** `down(relu(gate(x))²·up(x))` (KHÔNG SwiGLU — để khớp `build_bitnet_158` của bitnet.cpp), RoPE NEOX, RMSNorm SubLN, tied embed, d_model 768/12 layer/12 head/d_ff 2048 = 109.6M params.
3. **Converter** `scripts/convert_to_gguf.py`: arch `"bitnet-b1.58"`, áp `weight_quant` (ternary {-m,0,+m}) trước khi lưu, `add_rope_dimension_count`, cờ `--f16` (tải nhẹ), `--d-model/--d-ff/...` (test dims).
4. **Train lại relu² từ đầu trên cloud** (A4000). Xem "Cloud" bên dưới.

## Bug deploy — ĐÃ LOẠI TRỪ (đừng dò lại)
- Template: prompt đúng `<s>>>jpn<<{src}</s>` KHÔNG cách sau thẻ → token `[2,5,...,3]` chuẩn. Chạy `llama-cli --special`.
- Weight scheme: đã thêm weight_quant vào converter.
- Packing i2_s: `llama-quantize` của build này tạo file **byte-identical** với official (cùng md5 khi quantize model `large`).
- Kích thước: untrained ở dims 2B qua converter TÔI cũng câm ⇒ không phải dims.
- Activation outlier: attn_norm absmax 2.87, ratio 3.5, 765/768 phần tử ≠0 sau int8 → không crush.
- Embedding lookup: giá trị `inp_embd` khớp `embed[3189]` cả đầu lẫn cuối.
- i2_s: mọi matmul ternary ra **0** (kernel MAD). Q8_0: matmul ra nonzero nhưng vẫn câm.
- **Lưu ý:** `llama-eval-callback` báo "sum" KHÔNG đáng tin (0.0776 cho vector sum thật 0.4165) → không dùng "sum" để so lớp. Muốn so phải dump giá trị element hoặc viết harness khác.

## Nghi ngờ còn lại
Lệch tinh vi giữa **forward PyTorch (`src/bitnet.py`)** và **`build_bitnet_158()`** trong `~/BitNet/3rdparty/llama.cpp/src/llama.cpp` (~dòng 15389): kiểm RMSNorm eps, **RoPE NEOX convention** (rotate_half vs interleaved), thứ tự op, cách áp scale.

## HƯỚNG TIẾP (mai)
- **(A) Ưu tiên:** route model qua converter CHÍNH CHỦ `~/BitNet/utils/convert-hf-to-gguf-bitnet.py` (chính công cụ tạo 2B chạy tốt). Export model → HF-BitNet format (config.json + safetensors) → convert. Loại bỏ converter tự viết = xác suất fix cao nhất.
- **(B) Fallback có sản phẩm ngay:** đóng gói chạy PyTorch CPU (`generate_cached`, ~24 tok/s) — chậm hơn nhưng DÙNG ĐƯỢC.
- **(C)** Dò tiếp lệch graph (cần harness so element-wise, KHÔNG dùng eval-callback sum).

## Scripts debug (đã có trong repo)
- `scripts/gguf_pytorch_check.py <gguf>`: nạp trọng số GGUF → PyTorch → generate (chứng minh model đúng).
- `scripts/test_ckpt_translate.py <ckpt.pt>`: test dịch nhanh 1 checkpoint (frozen = i2_s), chạy CPU.
- `scripts/compare_logits.py <gguf>`: top-8 token kế tiếp từ PyTorch.
- `scripts/trace_compare.py`: dump sum từng lớp layer-0 (LƯU Ý eval-callback sum không đáng tin).

## Cloud (A4000, ckey.vn)
- SSH `root@n2.ckey.vn -p 2430` (mật khẩu user cấp riêng). python = `/opt/conda/bin/python3` (torch 2.5.1+cu124).
- Training relu² đang chạy `bash cloud/run_cloud.sh` (nohup, tự resume/restart, tới step 14000). Checkpoint mới nhất: `~/Train-model-translate/checkpoints/last.pt` (lưu mỗi 100 step). Backup run SiLU cũ ở `checkpoints/_silu_backup/`.
- Convert TRÊN cloud rồi tải i2_s (67MB) / F16 (269MB, `--f16`) về. Egress cloud CHẬM (~100-350KB/s) → resume tải bằng `tail -c +$((offset+1)) file >> local`.
- Đẩy file lên cloud khi scp fail: `base64 -w0 file` rồi `echo <b64> | base64 -d > remote_file`.

## Lệnh nhanh (local, venv ~/BitNet/.venv_bitnet)
```
python scripts/convert_to_gguf.py --ckpt <ckpt> --out dist/x_f32.gguf   # bỏ --ckpt = untrained
~/BitNet/build/bin/llama-quantize dist/x_f32.gguf dist/x_i2s.gguf i2_s
~/BitNet/build/bin/llama-cli -m dist/x_i2s.gguf --special --no-display-prompt \
  -p "<s>>>jpn<<Tôi thích mèo.</s>" -n 32 -t 6 --temp 0 --no-warmup
~/BitNet/build/bin/llama-bench -m dist/x_i2s.gguf -t 6 -p 128 -n 128
```
