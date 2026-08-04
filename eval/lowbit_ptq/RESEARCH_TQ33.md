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

## 6. File

- `exp_t24_codec.py` — codec tham chiếu + verify lossless (LUT 33³ bản 16-bit/12w đầu tiên, 1.458 bpw).
- `exp_t24_export_bench.py` — xuất tensor thật → block64 12-byte + x/y_ref.
- `tq33_bench.c` — kernel AVX2 v3 + microbench (đúng/sai, cache, DRAM-stream, đa luồng).
- Data bench: `D:\Bit-Translate-data\tq33_bench\`.
