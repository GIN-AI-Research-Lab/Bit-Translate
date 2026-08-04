# TQ33 — format 1.5 bpw tự chế cho 2:4-ternary (thay TQ2_0 2.06 bpw)

**Ngày**: 2026-08-04 · **Trạng thái**: Phase 1 XONG — codec lossless + kernel AVX2 chứng minh trên máy B · **Chi phí**: $0 (local)

## 1. Vấn đề

Model của lab là **2:4-ternary**: mỗi nhóm 4 trọng số có ≤2 giá trị khác 0, giá trị ∈ {−1, 0, +1}.
Container TQ2_0 của llama.cpp mã hóa ternary **đặc** (2 bit/trọng số, không biết gì về mask 2:4)
→ trả 2.0625 bpw cho thông tin thực chỉ ~1.36 bpw. Câu hỏi của user: *"tại sao không viết
riêng để tận dụng 1.56bpw?"* — đáp án: viết được, và đây là format.

## 2. Format TQ33

Đếm pattern hợp lệ của nhóm-4 (≤2 nonzero, mỗi nonzero ±1):
- 0 nonzero: 1 pattern; 1 nonzero: C(4,1)×2 = 8; 2 nonzero: C(4,2)×4 = 24 → **33 pattern**.

Ghép **cặp nhóm** (8 trọng số): 33² = 1089 ≤ 2¹¹ → **1 codeword 11 bit**.

**Block chuẩn = 64 trọng số = 12 byte** (hợp GGML, mọi chiều tensor chia hết 64):

```
[11 byte  = 8 codeword 11-bit, LSB-first liền mạch]
[ 1 byte  = scale (f8-grid / codebook ≤256 giá trị per-tensor)]
```

**bpw = 12×8/64 = 1.500 chẵn.** Giải mã: LUT 1089×8 int8 (~8.7 KB, nằm gọn L1).

So sánh trần Shannon: phân bố pattern đo trên ckpt thật (gen4-0.6B, 6 tensor):
0nz 7.9% / 1nz 36.4% / 2nz 55.7% → entropy ≈ 1.23 bpw + scale 0.125 = **1.36 bpw tối ưu**.
TQ33 fixed-rate trả 1.50 — cách trần đúng 0.14 bpw, đổi lấy truy cập O(1) từng block
(entropy coding sẽ mất random access + chậm decode).

## 3. Bằng chứng Phase 1 (đều trên tensor THẬT của ckpt bake `qat_gen4_n4.pt`)

### 3.1 Codec Python (`exp_t24_codec.py`)
6 tensor đại diện (q/k/v/o block-0, up/down block-27):
- Tách (t, s) **exact** từ weight bake (scale = max|w| per g64, nonzero |w| = s duy nhất).
- **0 vi phạm** 2:4 trên toàn bộ nhóm; encode → decode **lossless bit-level** cả 6/6.
- ⇒ PPL bản TQ33 = PPL bản bake **theo định nghĩa** (không cần đo lại).

### 3.2 Kernel C AVX2 (`tq33_bench.c`, build bằng `python -m ziglang cc -O3 -mavx2 -mfma`)
Tensor down_proj [1024×3072] thật, x ngẫu nhiên, y_ref từ Python:

| Kiểm | Kết quả |
|---|---|
| GEMV fp32-exact (decode C vs y_ref) | rel err **2.9e-07** ✓ (nhiễu float thuần) |
| GEMV int8-activation (đường inference) | rel err 6.7e-03 (lượng tử x chuẩn Q8) |
| Codebook scale | 43 giá trị duy nhất ≤ 256 ✓ (đúng f8-grid) |

Trick dot ternary: `maddubs(|t|, sign(xq, t))` — Σt·x cho 32 trọng số/lệnh, số 0 tự triệt.

### 3.3 Tốc độ (máy B: Core Ultra 5 225H, DDR5-5600, đo DRAM-stream 151MB đè cache)

| Kernel | 1 luồng | 6 luồng | Chiếu 30B-A3B* |
|---|---|---|---|
| v1 (LUT→buffer, hsum/block) | 3.4 GB/s | 15.4 GB/s | ~25 tok/s |
| v2 (vpgather thẳng thanh ghi) | 2.6 GB/s | 13.0 GB/s | ~21 tok/s ✗ gather đắt |
| **v3 (LUT→buffer, FMA-acc/hàng)** | **5.6 GB/s** | **19.3 GB/s** | **~31 tok/s** |

\* active ~3.3B params linear @1.5bpw ≈ 0.62 GB đọc/token; chỉ tính stream linear.
End-to-end thực (thêm attention, router, sampling) ước **~0.7×** → **21–24 tok/s**.

Baseline đo thật cùng máy (llama-bench, 5 luồng): Q4_K_XL 16.5GB → 16.0 tok/s;
UD-IQ1_S 8.4GB → 16.5 tok/s. ⇒ TQ33 **+35–50% tốc độ**, file **6.2GB vs 16.5/8.4GB**.

## 4. Vì sao nhanh hơn dù phải decode

Inference CPU là memory-bound: tok/s ≈ băng thông ÷ byte-đọc/token. TQ33 đọc ít hơn Q4_K
**2.7×**; miễn kernel decode+dot theo kịp DRAM là thắng. Kernel v3 6 luồng đạt 19.3 GB/s
(DRAM máy B 42 GB/s) — compute-bound nhẹ, còn khoảng để tune (interleave 2 hàng, pin P-core,
AVX-512 trên máy khác), nhưng đã vượt xa tốc độ đọc tương đương của Q4 (16 tok/s ⇔ ~28 GB/s
trên file 1.78GB active).

## 5. Điều kiện áp dụng + đường tích hợp

- **Chỉ ăn với model 2:4-ternary + scale trên lưới ≤256 giá trị** — chính là gia phả S1 của lab
  (F0: train-in-frame). Model thường KHÔNG nén được kiểu này.
- 30B-A3B: ckpt `expv_Qwen3-30B-A3B_2x4_tq2native.pt` (geo6 677) đã 2:4+g64-f8 sẵn →
  convert thẳng sang TQ33: **~5.6GB linear + ~0.6GB embed/router ≈ 6.2GB** (vs 8.5GB TQ2_0).
- Phase 2 (4–7 ngày): fork llama.cpp thêm `GGML_TYPE_TQ33` (block 12B/64w y hệt bench),
  `quantize_row` + `vec_dot` (kernel v3), converter từ ckpt S1 → gguf. Chạy được cả 0.6B/30B.
- Phase 2b (rẻ hơn, thử trước 0.6B): viết runner riêng ngoài llama.cpp cho 0.6B để đo PPL
  end-to-end format TQ33 mà không đụng build system lớn.

## 6. Phase 2 — tích hợp đầy đủ llama.cpp vs runner độc lập (quyết định 04/08)

Khảo sát trực tiếp source llama.cpp (checkout tại `D:\Bit-Translate-data\pack_local\llama.cpp`,
đối chiếu với đường TQ2_0 làm mẫu) cho ra bản đồ đầy đủ:

**Tích hợp đầy đủ (fork thật, để llama-cli/llama-bench chạy được thẳng):**
- Core: `ggml.h` (enum), `ggml-common.h` (struct block), `ggml-quants.c/.h` (quantize/dequantize
  + dispatcher `ggml_quantize_chunk`), `ggml.c` (bảng `type_traits`), `ggml-cpu/quants.c` +
  `ggml-cpu/arch/x86/quants.c` (kernel AVX2 — nơi port thẳng `tq33_bench.c`), `ggml-cpu.c` (bảng
  `type_traits_cpu`: from_float/vec_dot/vec_dot_type).
- llama core: `llama.h`, `llama-quant.cpp`, `llama-model-loader.cpp`, `tools/quantize/quantize.cpp`.
- Python: `gguf-py/constants.py` (enum + bảng blck_size/type_size), `gguf-py/quants.py` (class
  quantize/dequantize numpy — **đây là đường THẬT convert_hf_to_gguf.py dùng**, đã xác nhận vì
  `--outtype tq2_0` chạy trực tiếp không cần llama-quantize), `convert_hf_to_gguf.py` (--outtype).
- **Không cần sửa CMakeLists/Makefile nào** — mọi thứ là code thêm vào file đã biên dịch sẵn.
- **Không backend nào bắt buộc** — TQ1_0/TQ2_0 vốn CPU-only (0 hit trong CUDA/Metal/Vulkan/SYCL/
  OpenCL), khớp nhu cầu của ta (máy chỉ có Intel Arc iGPU, không train/infer GPU ở đây).
- **Rủi ro thật**: `ggml-cpu/ops.cpp` có ~7 switch-case liệt kê mọi quantized type
  (`add/add1/acc/out_prod/set/get_rows/clamp`), `default: GGML_ABORT`. Quên thêm case → không lỗi
  build, mà **crash cứng lúc chạy** nếu graph gọi đúng op đó trên tensor TQ33 (`get_rows` nguy
  hiểm nhất nếu lỡ áp TQ33 lên `token_embd`). Né bằng cách CHỈ áp TQ33 cho linear attn/ffn (đằng
  nào embedding/lm_head cũng phải giữ bit cao theo định luật F0 — [[lowbit-lab-hoc-tap]]).
- Ước lượng thật (không phải phỏng đoán): **4-6 ngày** — khó nhất là AVX2 kernel vì TQ2_0 dùng
  `vec_dot_type=Q8_K` (block 256) còn TQ33 block-tự-nhiên=64 khớp `Q8_0` (block 32) hơn — không
  copy máy móc được, phải tự thiết kế lại phần lượng tử hóa activation.

**Runner độc lập (khuyến nghị, đã chọn 04/08):** không đụng ggml/build system, tự dựng forward
pass Qwen3/Qwen3MoE bằng C thuần (không cần ggml.h), đọc thẳng ckpt bake bằng script Python xuất
binary, dùng kernel `tq33_bench.c` cho các linear. Né HOÀN TOÀN rủi ro switch-abort (không đụng
graph generic nào), không cần đăng ký enum hệ thống, không cần Python gguf-py class. Đổi lại: mất
KV-cache/sampling/batching có sẵn của llama.cpp (phải tự viết tối thiểu). **Ước 2-3 ngày**, và
quan trọng hơn: buộc validate ĐÚNG/SAI từng layer so PyTorch oracle trước khi tin số tốc độ —
thứ mà microbench 1-tensor (Phase 1) chưa chứng minh được cho một model đầy đủ.

## 7. Sự cố CRC-32 khi tải ckpt 61GB qua Modal (04/08, ~2h điều tra)

Sau khi tải ckpt 30B (61GB) về, quét CRC-32 nội bộ zip phát hiện lỗi — quá trình điều tra
và bài học (đầy đủ trong memory, tóm tắt ở đây):

1. **`torch.load(mmap=True)` crash** trên file 61GB (access violation) — không phải file
   hỏng (OS mmap + sequential read + MD5 đều OK). Fix: `safe_ckpt_reader.py` (custom
   pickle unpickler, đọc trực tiếp qua zipfile thay vì torch).
2. **Quét CRC lần 1: 81/37491 entry sai. Tải lại (lần 2, tải mới hoàn toàn): 113/37491 sai,
   0 TRÙNG với lần 1** → xác nhận lỗi TRUYỀN TẢI (transfer-side), không phải hỏng nguồn
   Modal (nếu hỏng nguồn thì 2 lần tải phải sai CÙNG chỗ).
3. **Bẫy quan trọng**: CRC-32 chỉ bảo vệ phần DATA của mỗi entry zip, KHÔNG bảo vệ local
   header (nlen/elen). Nếu lỗi truyền tải chạm cả header, `nlen/elen` đọc sai (vd đọc ra
   `0/0` thay vì `42/32`) → cả `zipfile` chuẩn lẫn code tự viết tính SAI vị trí data, báo
   "hỏng" dù data thật vẫn nguyên vẹn. **Fix: suy `elen` từ bất biến "data bắt đầu ở địa chỉ
   chia hết 64" (`.storage_alignment=64`) thay vì đọc từ header nghi ngờ**, `nlen` tính
   trực tiếp từ tên entry (đã biết chính xác, không cần đọc). Cách này tự phục hồi
   **42/112 tensor tưởng hỏng** (data đúng, chỉ header sai) — đã đưa vào `safe_ckpt_reader.py`
   làm mặc định.
4. **70/112 còn lại hỏng DATA thật** (đã thử vá bằng 3 lần stream-tải-lại độc lập, đều thất
   bại ở ĐÚNG các vị trí này — nghi ngờ cache CDN/mạng tạm thời phía Modal, không phải hỏng
   nguồn tuyệt đối vì lần 1↔lần 2 không trùng). 14/70 tải về toàn số 0 (lỗ tải bị zero-fill
   thay vì báo lỗi), 56/70 có dữ liệu khác-không nhưng sai CRC (garbled thật).
   **Quyết định**: sau ~2 giờ điều tra không phục hồi được, **zero-hoá minh bạch** 70 tensor
   này khi encode (ghi rõ trong `corrupted_zeroed.json`) — tương đương vô hiệu hoá 1
   projection của 1 expert cụ thể, KHÔNG mã hoá nhiễu làm dữ liệu sai lệch âm thầm. MoE
   128 expert/layer chịu được mức độ này (dự kiến <60 expert riêng biệt bị ảnh hưởng /
   6144 tổng, mỗi expert chỉ 1/8 khả năng được router chọn/token).
5. **Bài học tổng quát**: khi tải ckpt lớn (>10GB) qua mạng, LUÔN quét CRC-32 nội bộ zip
   (không chỉ so MD5 toàn file — MD5 chỉ xác nhận ổn định giữa các lần đọc CỤC BỘ, không
   xác nhận khớp bản gốc); nếu có entry hỏng, LUÔN thử offset suy từ alignment trước khi
   kết luận mất dữ liệu — phần lớn trường hợp chỉ header hỏng, data vẫn còn.

## 8. File

- `exp_t24_codec.py` — codec tham chiếu + verify lossless (LUT 33³ bản 16-bit/12w đầu tiên, 1.458 bpw).
- `exp_t24_export_bench.py` — xuất tensor thật → block64 12-byte + x/y_ref.
- `tq33_bench.c` — kernel AVX2 v3 + microbench (đúng/sai, cache, DRAM-stream, đa luồng).
- Data bench: `D:\Bit-Translate-data\tq33_bench\`.
