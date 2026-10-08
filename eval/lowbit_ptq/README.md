# Lab học Low-bit: convert model có sẵn xuống 1.58 / 2 / 2.xx bit

> **Bàn giao sang máy khác (Laguna S2.1):** đọc `HANDOFF_SUB1BIT_LAGUNA.md` — bản đồ kết quả,
> job đang chạy (exp_k/exp_l), và kế hoạch 2 phase cho Laguna 118B.
> **Đính chính kế toán bit (2026-08-01):** số bpw trong Bài 6–7 dưới đây THIẾU scale overhead
> (+0.5 f16/g32) và mask không cấu trúc bị tính thiếu — bảng bpw đúng nằm ở HANDOFF §1 và exp_k/l.

> Mục tiêu: hiểu bằng **số đo thật trên máy B (CPU, không GPU)** vì sao convert model có
> sẵn xuống ternary khó, chỗ nào là tường lý thuyết, chỗ nào còn dư địa. Mọi script ở đây
> chạy trên trọng số Qwen3-0.6B THẬT (`D:\Bit-Translate-data\hf_cache`), tự kiểm chứng,
> **không có kernel giả / trọng số randn** như lô đã bị bóc trong `native_engine/KETQUA_ENGINE.md`.

Chạy: `set OMP_NUM_THREADS=5 && python eval/lowbit_ptq/exp_X.py`

---

## Bài 0 — Định luật chặn: rate–distortion (không phá được bằng thuật toán)

Lượng tử hóa trọng số Gaussian là bài toán rate–distortion của Shannon: mỗi mức bit ứng với
một mức méo **tối thiểu cố định**. Đây là "định luật bảo toàn" — như không thể chế máy
chuyển động vĩnh cửu. `exp_b_weight_error_table.py` đo trên 35 ma trận thật:

| bpw | scheme | sai số năng lượng trọng số |
|---:|---|---:|
| 4.5 | int4-g32 | **9.9%** ← vùng dùng được |
| 3.5 | int3-g32 | 23.1% |
| 2.5 | int2-g32 (Lloyd) | 34.1% |
| 2.5 | ternary-g32 (Lloyd) | **43.0%** ← 1.58-bit rơi ở đây |
| 5.0 | ternary 2-plane (cascade) | 19.9% ← **tốn hơn int4 mà tệ hơn** |
| 2.86 | int2 + 1% outlier f16 | 31.0% |

Ba nguồn độc lập trùng khớp: đo của tôi, `KETQUA_ENGINE.md §5` (44–47%), và lý thuyết
Gaussian (~40–45%). **Kết luận: "cascade/residual xuống 2.45 BPW" của Antigravity là bất khả**
— muốn sai số như int4 thì phải tốn bit như int4.

## Bài 1 — Thang PTQ thật + imatrix (llama.cpp), đo PPL

`exp_a_ptq_ladder.py`: imatrix calibration vi+ja → quantize → PPL trên dev held-out.
IQ1_S (1.56 bpw — PTQ 1-bit tốt nhất llama.cpp có, đã gồm Hadamard + codebook + imatrix):

```
IQ1_S  198MB   PPL_vi = 20492   PPL_ja = 18390   → sụp hoàn toàn
```

Ngay cả kỹ thuật PTQ đỉnh nhất cũng không cứu được 1.56 bit trên model 0.6B.

## Bài 2 — Hadamard incoherence: LÀM ĐÚNG vs LÀM GIẢ

`exp_c_incoherence.py`. Antigravity đặt tên "Hadamard incoherent" nhưng code chỉ lật dấu
→ với quantizer theo |W| thì lật rồi lật lại = **no-op**. Làm ĐÚNG là xoay trọng số bằng
ma trận Hadamard trực giao (và xoay activation ở runtime) để dập outlier:

| | ternary | int2 |
|---|---:|---:|
| baseline (không xoay) | 43.3% | 34.4% |
| **fake sign-flip (Antigravity)** | **43.3%** (≡ baseline, vô dụng) | 34.4% |
| **Hadamard incoherence THẬT** | **42.4%** (−2.2%) | 33.4% (−2.8%) |

Tác dụng thật nhưng nhỏ; mạnh nhất ở `down_proj`/`o_proj` (−4…5%, nơi nhiều outlier).
**Đủ để đẩy ngưỡng dùng-được từ ~4 bit xuống ~3 bit, KHÔNG tới 1.58.** Đây là kỹ thuật SOTA
thật ([QuIP#](https://arxiv.org/pdf/2402.04396), [QTIP](https://arxiv.org/html/2406.11235v3))
— nhưng SOTA của nó là 2-bit trên model **7B+**, không phải 0.6B.

## Bài 3 — Vì sao "HỌC" cứu được ternary, "LÀM TRÒN" thì không (trái tim BitNet)

`exp_e_ste_qat_toy.py`. Cùng ràng buộc ternary trên `down_proj` thật + activation có outlier:

```
PTQ (làm tròn):  sai số OUTPUT = 52.5%          (tối ưu ||W_t − W||, sai đại lượng)
QAT-STE (học) :  sai số OUTPUT = 21.1%          (tối ưu ||W_t·X − W·X||, đúng đại lượng)
                 → giảm 60% sai số output
```

Nghịch lý mấu chốt: **QAT có sai số TRỌNG SỐ cao hơn (57% vs 53%) nhưng OUTPUT thấp hơn** —
nó được phép DỊCH master weight (41% khỏi gốc) để các trọng số bù trừ nhau trên dữ liệu thật.
PTQ không được phép làm thế, nên mọi trò cascade/Hadamard/codebook đều đụng trần 43%.
**Đây là lý do BitNet phải train from-scratch / distill, không quantize sau được**
([BitNet b1.58 report](https://arxiv.org/pdf/2504.12285)).

## Bài 4 — "Không train TOÀN model" đi tới đâu? Layer-wise reconstruction (exp_f)

`exp_f_layerwise_reconstruction.py` áp reconstruction cho cả **196 ma trận** Qwen3-0.6B: tối ưu
từng ma trận độc lập để khớp output (STE), **không backprop toàn model, không thầy, chạy CPU**.

| | PPL vi | PPL ja |
|---|---:|---:|
| FP32 | 45.6 | 89.2 |
| PTQ ternary làm tròn | 7.708.616 | 7.832.234 |
| **Layer-wise reconstruction** | **5.185** | **12.072** |

Reconstruction kéo PPL xuống **~1500× so PTQ ngây thơ** (bố trí thông minh CÓ tác dụng mạnh),
nhưng **vẫn cách FP ~110×** — chưa dùng được. Sai số/lớp 47% → 23%, nhưng 23% × 28 lớp *tích lũy*
vẫn làm PPL sụp. Và đây là **chặn trên lạc quan** (teacher-forcing: mỗi lớp thấy activation FP,
mù về lỗi 27 lớp kia). Nút "train/không train" là **núm xoay**: một chút tối ưu per-layer đã đi
rất xa từ điểm chết, nhưng error-propagation qua chiều sâu là tường mà chỉ **sequential/block-wise
reconstruction** (GPTQ/BRECQ) hoặc **QAT** vượt được.

> ⚠️ Bẫy: bản LR cao đầu tiên cho reconstruction TỆ HƠN cả PTQ (optimize không hội tụ, e1 > e0).
> Bắt buộc **track-best + LR thấp**. Đây đúng loại lỗi khiến một thí nghiệm hỏng bị đọc nhầm
> thành "phương pháp vô dụng" — luôn kiểm tra tính nhất quán nội tại trước khi kết luận.

## Bài 5 — Kết hợp reconstruction + mixed precision: đường cong PPL-vs-bit (exp_g)

`exp_g_mixed_precision_curve.py` gộp 2 hướng: reconstruction (Bài 4) + "lục lọi layer" (nâng
các lớp nhạy nhất lên int4, còn lại ternary). Quét tỉ lệ int4 → đường cong ngân sách bit:

| % lớp int4 | bit TB | ~MB (linear) | PPL vi | PPL ja |
|---:|---:|---:|---:|---:|
| 0% (toàn ternary) | 2.08 | 109 | 7.121 | 12.547 |
| 15% | 2.35 | 123 | 2.082 | 4.593 |
| 30% | 2.79 | 147 | 1.469 | 3.950 |
| 50% | 3.31 | 174 | 642 | 1.415 |
| **100% (toàn int4)** | **4.50** | **236** | **86** | **174** |
| FP32 | 16 | — | 69 | 125 |

**Kết luận sắc nét — trần của "không train" ≈ 4 bit:**
- int4 + reconstruction đạt **gần FP** (86 vs 69) → **dùng được, không train**.
- Đường cong **dốc đứng** giữa 3.3 và 4.5 bit; xuống vùng ternary/2-bit thì PPL sụp hàng nghìn
  *dù đã kết hợp mọi hướng PTQ* (reconstruction + mixed precision + đây còn là chặn trên lạc quan).
- **Vòng tròn khép:** "không-train tốt nhất" (int4 ~236MB linear + embedding ≈ 370MB tổng) chính
  là **Q4_K_M mà llama.cpp đã cho sẵn ở 92 tok/s**. Mọi nỗ lực không-train hội tụ về Q4 có sẵn.
- ⇒ Muốn **nhẹ hơn Q4** (2-bit, 1.58-bit) thì **bắt buộc train** (QAT/distill). Không có lối tắt.

## Bài 6 — Dải SUB-1.58 / SUB-1-bit trên Qwen3-0.6B (exp_h)

`exp_h_sub1bit_ladder.py` — reconstruction (không train toàn model) ở các format dưới ternary:

| format | bit/w | PPL vi | PPL ja |
|---|---:|---:|---:|
| ternary (mốc) | 2.08 | 11.406 | 14.117 |
| **binary (1 bit)** | 1.50 | **1.861.464** | **6.897.504** |
| binary + 5% salient int8 | 2.14 | 969.227 | 1.519.775 |
| ternary 50% sparse | 2.04 | 10.305 | 19.523 |
| ternary 25% giữ (sub-1) | 1.33 | 33.687 | 47.152 |
| FP32 | 16 | 69 | 125 |

**Phát hiện cốt lõi — giá trị của "1.58 bit" nằm ở MỨC 0, không ở ba mức:**
- Bỏ ternary→binary (mất đúng một mức: **0**) làm PPL nhảy từ ~13k lên **~vài triệu** (×160), dù chỉ
  mất ~0.58 bit. Không tuyến tính. Mức 0 = khả năng "tắt" trọng số = phần lớn sức mạnh của 1.58 bit.
- Xuống sub-1.58 bằng **sparsity (giữ 0)** tốt hơn **binary (bỏ 0)** rất nhiều: ternary-25%-giữ ở
  **1.33 bit → PPL 33k**, trong khi binary 1.5 bit → PPL 1.9 **triệu**. Ít bit hơn mà tốt hơn 50×.
- ⇒ Đây chính là lý do SOTA sub-1-bit (STBLLM) dùng **structured sparsity trên binary**, không phải
  binary thuần: giữ lại khả năng-zero là tối quan trọng.

**Nhưng: mọi format <1.58 bit đều SỤP trên 0.6B** (PPL tốt nhất 10k, cách FP 69 xa vời). So SOTA
trên LLaMA-**7B** (STBLLM 0.55bit PPL 31.7, BTC-LLM 0.7bit PPL 11.0): model 7B dư thừa lớn nên
sub-1-bit sống; **0.6B ít dư thừa → sàn bit cao → là ca KHÓ NHẤT cho extreme quantization.**

## Bài 7 — STBLLM-style: structured binary/ternary + N:M sparsity (exp_i)

`exp_i_structured_binary.py` — kiểm giả thuyết "mức 0 là vua" từ Bài 6 bằng structured N:M sparsity
(mỗi 4 cột giữ N phần tử lớn nhất làm binary/ternary, còn lại = 0). Reconstruction, không train.

| format | bit/w | PPL vi | PPL ja |
|---|---:|---:|---:|
| binary dense (Bài 6) | 1.50 | 1.861.464 | 6.897.504 |
| **binary 2:4** | 1.15 | **33.354** | 70.705 |
| binary 1:4 (sub-1) | 0.75 | 237.377 | 186.729 |
| **ternary 2:4** | **1.44** | **6.833** | 21.470 |
| ternary 1:4 (sub-1) | 0.90 | 30.506 | 184.271 |
| ternary dense (Bài 6) | 2.08 | 11.406 | 14.117 |
| FP32 | 16 | 69 | 125 |

**Hai kết quả mạnh:**
- **Structured sparsity CỨU binary ~56×**: binary-dense 1.5bit PPL 1,9 triệu → binary-2:4 1.15bit PPL
  33k. Ít bit hơn mà tốt hơn 56 lần — chỉ nhờ đưa mức 0 vào có cấu trúc. Xác nhận dứt khoát Bài 6
  và lý do STBLLM dùng structured-sparse-binary chứ không binary thuần.
- **ternary 2:4 @ 1.44 bit là ĐIỂM NGỌT**: PPL 6.833 — vừa *nhẹ hơn* ternary-dense (2.08bit) vừa
  *tốt hơn* nó (11.4k). Structured 2:4 + reconstruction khai thác cấu trúc + mức-0 tốt hơn dense.

**Nhưng vẫn KHÔNG "sống": mọi format vẫn cách FP ~100×** (tốt nhất 6.8k vs 69). Trên 0.6B, ngay cả
đúng vũ khí SOTA (structured binary) cũng không đưa sub-1.58 về vùng dùng được — trong khi STBLLM
đạt PPL 31.7 ở 0.55bit trên LLaMA-7B. **Kết luận cuối: kỹ thuật đúng, model sai cỡ.** Muốn sub-1-bit
"sống" cần model 7B+ (dư thừa chức năng) HOẶC binary QAT (train). 0.6B là giới hạn cứng.

## Bài 8 — Fix-pack + SEQUENTIAL: đòn bù liên lớp (exp_k)

`exp_k_fixpack_sequential.py` — 3 arm cô lập từng biến số (fix-pack = scale Lloyd-survivor +
mask Wanda + 100 step + calib 60 câu; kế toán bit đủ):

| arm | bpw | PPL vi | PPL ja |
|---|---:|---:|---:|
| dense ternary, fix-pack TF | 2.08 | 4.401 | 8.250 |
| ternary 2:4, fix-pack TF | 1.94 | 5.774 | 33.375 |
| **ternary 2:4, fix-pack + SEQUENTIAL block-wise** | 1.94 | **594** | 8.069 |
| FP32 | 16 | 69 | 125 |

**Ba kết luận:**
1. **SEQUENTIAL (BRECQ-lite) ăn ~10×** (5.774 → 594 vi): mỗi block nhận input *đã lượng tử* của
   prefix + khớp quỹ đạo FP → block sau BÙ lỗi block trước. Đây là cơ chế lớn nhất tìm được sau
   fix-pack; PPL vi 594 @1.94bpw chỉ còn 8.6× FP — chạm mép "vùng xám dùng được cho việc dễ".
   Trace cho thấy block 2 và block 27 cực nhạy (mse 32k→81 và 16.5k→723) — sequential cứu đúng chỗ đó.
2. **ĐÍNH CHÍNH sweet-spot Bài 7:** ở NGANG bước tối ưu, dense 2.08bpw (4.401) THẮNG t2:4 1.94bpw
   (5.774) trên TF — chiến thắng của 2:4 ở exp_i là artifact hội tụ (2:4 ít tham số tự do nên hội tụ
   nhanh hơn trong 80 step). Nghi vấn audit #3 xác nhận. (Chưa đo dense+sequential — khoảng trống.)
3. Bất đối xứng vi/ja: sequential kéo vi mạnh hơn ja nhiều (594 vs 8.069) — nghi calib ja chưa đủ
   nặng ký; đáng tăng tỉ trọng ja trong calib ở vòng sau.

## Bài 9 — Thang SUB-1-BIT THẬT: kết quả cuối (exp_l)

`exp_l_true_sub1bit.py` — 6 arm, kế toán đủ payload+mask+scale, cùng fix-pack TF như exp_k
(để so trực tiếp arm1/1b), quét 3 đòn bẩy giá-bit: scale (tensor/row/f8-g64), mask (1:4/1:8/2:8),
payload (ternary/binary):

| arm | bpw thật | PPL vi | PPL ja |
|---|---:|---:|---:|
| A ternary dense, scale/TENSOR | 1.580 | 66.999 | 513.320 |
| B ternary 1:4, f8/g64 | 1.020 | 245.163 | 499.247 |
| F binary 2:8, f8/g64 | 0.976 | 22.311.065 | 11.792.505 |
| C ternary 1:4, f16/ROW | 0.908 | 2.318.366 | 1.660.599 |
| D binary 1:4, f8/g64 | 0.875 | 1.178.193 | 424.452 |
| E ternary 1:8, f8/g64 | 0.698 | 153.717 | 116.642 |
| (mốc exp_k TF: dense g32-f16 2.08 = 4.401 · t2:4 1.94 = 5.774 · t2:4+SEQ = **594**) |

**Kết luận & xếp hạng nguyên nhân chết (đo được, không đoán):**
1. **Mọi cấu hình <1.9 bpw đều chết trên 0.6B** (tốt nhất 67k, cách FP ~1000×). Thang khép lại
   bức tranh: trần không-train của 0.6B ≈ 4bit (dùng được) / ≈1.94bpw+SEQ (vùng xám 594).
2. **Thuế scale là yếu tố thống trị**: per-tensor (A) ×15 so g32-f16; f8/g64 (B) ×8 so f16/g32;
   per-row (C) tệ hơn cả binary-g64 (D). Granularity scale > mức 0 > số mức giá trị.
   PTQ *phải mua* scale mịn bằng bit — trái với QAT (v7a sống khỏe với scale per-tensor).
3. **Binary chết mọi biến thể** (D 1.2M, F 22M); ternary thắng binary ở MỌI cặp so sánh được.
4. Bất ngờ E > B (0.70 bpw 154k < 1.02 bpw 245k): 1:8 ít tham số tự do hơn → hội tụ nhanh hơn
   trong 100 step (nghi cùng loại artifact như 2:4-vs-dense Bài 8) — nhưng cả hai đều chết nên
   không đổi kết luận.
5. PPL-vs-bpw KHÔNG đơn điệu trong vùng chết — đừng nội suy giữa các format khác cơ chế.

## Bài 10 — Thế hệ 2: đường biên cuối + bảng quy công/tội (exp_m/n/o/p/q, đêm 01–02/08)

**Đường biên PPL-vs-bit cuối cùng** (best-of mỗi mức bit, eval dev vi/ja FP=69/125):

| bpw thật | PPL vi | PPL ja | vi/FP | Công thức thắng |
|---:|---:|---:|---:|---|
| 1.94 | 471 | 1.739 | 6.8× | exp_n (SEQ + scale học + norm + 2pass) |
| **1.56** | **400** | **1.708** | **5.8×** | **exp_n + exp_q (gauge + bias + budget) — kỷ lục lab** |
| 1.02 | 767 | 1.919 | 11× | exp_n (gauge không ăn ở mức này) |
| 0.70 | 1.055 | 2.456 | 15× | exp_n + polish (adapter + tail-KL) |
| 0.62 | 1.843 | 6.278 | 27× | exp_n (chưa polish) |

So thế hệ 1 (TF thuần): cùng 0.70 bpw từ 153.717 → 1.055 = **×146**. Điểm 1.56 bpw (dưới ngân sách
BitNet) đạt 5.8× FP — vượt cả kỷ lục cũ ở 1.94 bpw.

**Bảng quy công/tội từng kỹ thuật (tất cả đo bằng ablation, không suy diễn):**

| Kỹ thuật | Phán quyết | Bằng chứng |
|---|---|---|
| Sequential block-wise (BRECQ-lite) | ✅ đòn lớn nhất | ×10 (5774→594); ×219 khi format thô (M3) |
| Scale HỌC ĐƯỢC (thay Lloyd-derive) | ✅ xóa thuế f8/g64 | ×8 thuế → ×1.14 (N3 536 vs exp_l 245k) |
| Norm đồng-tối-ưu + 2 pass | ✅ (trong gói exp_n) | N1 471 vs Lloyd-SEQ 594; ja ×4.6 |
| **Gauge up↔down + v↔o** (0 bit) | ✅ **ở ≥1.5 bpw**: ×1.34; ≈0 ở 1.02 | Q(N3) 400 vs 536; Q(O1) 774 ≈ 767 |
| Bias học được + budget theo block | ✅ (gói với gauge) | trong Q(N3) |
| Adapter + tail-KL-polish | ✅ **chỉ khi bpw < ~1** (bộ hấp thụ lỗi hệ thống) | O2: 1476→1055 (+29%); trung tính ở ≥1 bpw |
| Regression guard step-0 | ✅ giữ (an toàn, miễn phí) | ablation A |
| Mask refresh giữa pass | ❌ LOẠI — thủ phạm ×2.2 | ablation B (1168 vs 521) |
| Relative-MSE | ❌ LOẠI — ja +38% | ablation A |
| Mask 2:4 "sweet spot" | ❌ artifact hội tụ | exp_k ngang bước: dense thắng TF |

**Quy tắc pipeline rút ra (mang sang Laguna):** gauge trước tiên (miễn phí, sanity-check FP bắt buộc)
→ Wanda mask cố định → sequential 2-pass với scale+bias học được + norm co-tune + guard step-0
→ nếu bpw < 1: thêm adapter + tail-KL-polish. KHÔNG refresh mask giữa chừng, KHÔNG relative-MSE.
⚠️ Mọi số trên eval 16 câu/ngôn ngữ — xếp hạng tin được, giá trị tuyệt đối cần kiểm định lại
trên eval lớn (mục "kiểm định" trong danh sách việc kế).

## Bài 11 — QAT-lite trên Modal + hai bảng ĐO THẬT tốc độ/chất lượng packed (02/08)

**Vòng cung QAT-lite @1.56bpw** (validation 100 câu vi; FP=69; script `exp_r_qat_lite.py`, chạy Modal L40S):

| Giai đoạn | Token KD | val-100 vi | ja (16c) | Ghi chú |
|---|---:|---:|---:|---|
| S1 (PTQ, không train) | 0 | 621.5 | 1.605 | |
| + e2e KD v2 (bug loss đã vá) | 0.8M | 448.1 | 1.141 | |
| + fast-loop v3 (batch thật + bf16, ×7 tốc độ train) | 1.3M | 399.5 | 1.085 | overfit sau 1.250 step (data 12k câu) |
| **+ v4: data 5× + cosine + EMA + KD-temp2 + CE0.1** | ~4M | **360.3** | **1.071** | **recipe chuẩn**; en/code TỰ HỒI ×15 dù KD thuần vi/ja |

Bug đã bắt trong nhánh này (đều bằng "kỳ vọng ghi trước + chất vấn số lạ"): (1) KL `batchmean` chia
theo batch=1 → loss to ×seq_len → LR hiệu dụng ×100 phá model; (2) lr 2e-4 + mở scale/norm làm nhảy
ô ternary — công thức đúng là **EfficientQAT-style: lr 2e-5, CHỈ train W+bias, freeze scale+norm**;
(3) checkpoint tải qua mạng phải verify (miniz corrupt). Guard step-0 + best-geo cứu cả 4 run hỏng.

**Tốc độ THẬT (llama-bench, máy B 6 luồng, packed format)** và **chất lượng packed** (llama-perplexity
test vi/ja, imatrix vi/ja):

| Format | bpw | tok/s tg64 | PPL vi | PPL ja | Phán quyết |
|---|---:|---:|---:|---:|---|
| Q8_0 | 8.5 | 57.5 | ~FP | ~FP | mốc |
| Q4_K_M | 4.5 | 84.7 | 50.0 | 75.2 | chuẩn thực dụng |
| Q2_K | 2.6 | 106.6 | 91.8 | 142.3 | ×1.8 — dùng tạm |
| IQ2_XS | 2.31 | 82.9 | 281.3 | 440.0 | ×5.6 — xám tối |
| IQ1_M | 1.75 | 96.2 | 4.387 | 6.413 | ×88 — chết |
| IQ1_S | 1.56 | 104.5 | 20.492 | 18.390 | ×410 — chết |
| **TQ2_0** | **2.06** | **144.5** 🏆 | **21.132.250** ☠️ | 17.068.433 | **nhanh nhất = chết nhất** |

**Ba kết luận vàng:** (1) kernel quyết định tốc độ hơn kích thước (TQ2_0 to hơn IQ1_S mà nhanh hơn 38%
— LUT ternary đạt 79% băng thông lý thuyết); (2) vách chất lượng của tooling đại chúng nằm ngay dưới
Q2_K; (3) cùng lớp 1.56–2 bpw: pipeline của ta 360 vs IQ1_S 20.492 vs TQ2_0 21 triệu → **F0 = retrain
theo ràng buộc TQ2_0-native (dense ternary, scale g256, bỏ bias) rồi requantize lossless → MỘT file
GGUF vừa 144 tok/s vừa PPL ~400** — mảnh khép mục tiêu "nhỏ + cực nhanh trên CPU".

⚠️ Bài học phương pháp lớn nhất nhánh này: **eval chỉ vi/ja = thiên vị chọn lọc cả chuỗi nghiên cứu**.
Calib vi/ja làm en ×808, code ×4.434 ở S1 (FP anchor: en 34.2, code 2.7). Từ nay mọi run có gate
4 miền; calib/KD trộn là bắt buộc cho model tổng quát.

## Bài 12 — Gate hành vi: sinh văn bản 4 miền × 5 mức bit (exp_s, mẫu đầy đủ: exp_s_report.md)

PPL 4 miền (eval cố định; FP anchor vi 69 / ja 125.1 / en 34.2 / code 2.7):

| Config | tầng đã qua | vi | ja | en | code |
|---|---|---:|---:|---:|---:|
| FP32 | — | 69.0 | 125.1 | 34.2 | 2.7 |
| 1.56 QAT-v3 | gauge→S1→S2 | 399.7 | 1.085 | 2.594 | 1.050 |
| 1.02 S1 | gauge→S1 | 676.0 | 2.849 | 21.448 | 16.599 |
| 0.70 S1 | gauge→S1 | 957.9 | 2.850 | 33.708 | 16.325 |
| 0.62 S1 | gauge→S1 | 1.384.5 | 2.890 | 48.339 | 19.201 |

(S1 bản exp_r — gauge+bias+budget — TỐT HƠN exp_n cũ ở bit thấp: 0.70 bpw 1.476→958 không cần train.)

**Phán quyết mẫu sinh (greedy, 60 token):** FP32 mạch lạc kiểu-0.6B; 1.56-QAT ra từ vựng vi thật,
bám đề vài từ rồi **rơi vòng lặp** ("và có thể, và có thể…"); 1.02 nửa chữ nửa spam số; ≤0.70 xà bần.
Bài học: **sinh-mở đòi hỏi cao hơn PPL nhiều** — PPL ×5.8 vẫn lặp khi greedy. Ở 0.6B, artifact
low-bit dùng được thật sự cần model lớn hơn (bậc 30B-A3B/Laguna) hoặc task đóng (dịch có nguồn).
tok/s trong exp_s là mô phỏng fp32 (~13-14) — tốc độ THẬT xem bảng packed Bài 11 (TQ2_0 144.5).

---

## Bài 13 — S1-v2: calib "căn cước model" (mixw) + HOÁN VỊ KÊNH gauge-exact (02/08 chiều)

Câu hỏi: sàn S1 (PTQ thuần) hạ được bao nhiêu nữa, và có CHIA ĐỀU cho các miền không?
Hai dao mổ mới trong `exp_r` (screening 11 ô × `--steps 0`, L40S 21 phút ≈ $0.7):

- **`--calib-mode mixw`** — calib theo căn cước Qwen3 (36T tok, 119 ngôn ngữ, nặng en/zh/code/math):
  vi 20 + ja 20 + en 24 + code 16 + **zh 20 + math 14 + chat 8** (122 câu); probe mở rộng 6 miền
  (thêm `ZH_EVAL`/`MATH_EVAL`; neo FP: vi 69.0 / ja 125.1 / en 34.2 / code 2.7 / zh 50.8 / math 5.5).
- **`--perm-gauge`** — hoán vị kênh CHÍNH XÁC TUYỆT ĐỐI (FP sau perm = 69.0, khớp từng chữ số):
  3 họ gauge (hidden toàn cục → input q/k/v/gate/up; intermediate mỗi block → input down;
  head-dim v↔o mỗi kv-head → input o), chia bài round-robin theo importance để kênh quan trọng
  rải đều nhóm M — mask N:M hết bị ép vứt kênh quan trọng vì trùng nhóm. Khác rotation (bị bác):
  GIỮ NGUYÊN thống kê từng kênh, chỉ đổi cách chia nhóm.

Bảng S1 (PPL; ☠ = miền chết >10⁵; geo6 = trung bình nhân 6 miền):

| bậc | cấu hình | vi | ja | en | code | zh | math | geo6 | đều (max/min ×FP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2:4 1.56 | mix4 (mốc gen3) | **444** | 2207 | 1448 | 142 | ☠* | 109* | ~1227* | ~×413 |
| | mix4+perm | 511 | 1959 | 1302 | 216 | ☠ | 109 | — | — |
| | mixw | 1025 | 3150 | 869 | **133** | 2056 | **37** | **552** | **×7.2** |
| | mixw+perm | 851 | 2381 | 1044 | 196 | **1864** | 42 | 565 | ×7.4 |
| 1:4 1.02 | mix4 | 823 | 4077 | 1607 | 246 | ☠ | 168 | 1767 | ×226 |
| | mixw | 1149 | **3343** | 2078 | 474 | 3234 | 54 | 934 | ×18 |
| | mix4+perm | **696** | 3389 | 1381 | 231 | ☠ | 129 | — | — |
| | mixw+perm | 969 | 5814⚠ | **1103** | **238** | **2281** | **45** | **732** | ×11 |
| 1:8 0.70 | mix4 | **1461** | 4804 | 4622 | 1619 | ☠ | 298 | 3521 | ×114 |
| | mixw+perm | 1666 | **3550** | **3030** | **925** | **3905** | **78** | **1310** | **×24** |
| 1:10 0.62 | mix4 | **1791** | 6321 | 7951 | 2320 | ☠ | 262 | 4779 | ×165 |
| | mixw+perm | 2427 | **5768** | **3786** | **1082** | **9060** | **96** | **1919** | **×23** |

(*zh/math của 2:4-mix4 chưa đo trực tiếp — lấy proxy từ ô mix4+perm; zh chết như nhau.)

**Phán quyết (kỳ vọng ghi trước → số thật):**
1. **Miền vắng calib = chết, lặp lần 3**: zh dưới mix4 = 122k–317k (×2400–6200 FP) — đúng nguyên
   xi bài "en ×808". Probe battery bắt điểm mù như thiết kế → calib PHẢI phủ miền cần sống.
2. **mixw mua ĐỘ ĐỀU bằng đỉnh vi**: max/min ×FP từ ×114–413 → ×7–24; zh sống lại ×24–75, math ×3–4;
   ja thường TỐT lên (chuyển giao chữ Hán zh→ja); giá: vi +14..40% (riêng 2:4: ×2.3).
3. **perm trung tính ở 2:4, THẬT từ 1:4 xuống, mạnh dần theo độ gắt mask** (đúng thuyết va chạm:
   giữ 1/M thì kênh quan trọng trùng nhóm là mất): 1:4 mix4 5/6 miền −6..−23% (geo4 −13%);
   1:8 combo en −34% code −43% math −74%; 1:10 en −52% code −53%. Geo6 full-stack vs mix4:
   **−54% (2:4) · −59% (1:4) · −63% (1:8) · −60% (1:10)** — sàn S1 KHÔNG phải format-bound
   như kết luận vội ở Bài 11; nó còn hạ được bằng calib + gauge, không tốn bit nào.
4. **Mắt xích yếu ja**: riêng 1:4 mixw+perm nổ ja +74% (π tổng gộp lấy chỗ kênh riêng ja chia cho
   zh/math). Vá: tăng share ja calib (20→~32, kiểu 60/40) — cần 1 ô screening trước GEN4.
5. gen3_n3 (mix4+v4+geo4, xong 31/07): **val-100 vi 353.8, vi 323.1 / ja 1077** — KD từ sàn cân
   bằng hơn + chọn best geo là hướng đúng; GEN4 = S1-v2 (mixw+perm) + v4 + KD-mix + geo6.

Bài học vận hành đắt giá: chuỗi driver local phóng `modal run` KHÔNG `--detach` → laptop sleep giết
app ephemeral giữa run, driver tưởng xong phóng bậc kế (mất ~$5, 2 run không xác nhận được).
Mọi run dài từ nay: MỘT hàm screen/chain chạy TRONG container + `--detach`.

## Bài 14 — Đợt A/B: 14 thách thức đấu đường biên S1-v2, 1 thắng (02/08 tối)

Sau Bài 13, screening tiếp 14 ô ($2.6) đấu với mốc S1-v2 (1:4 mixw+perm geo6 **732** @1.023bpw;
1:8 geo6 **1310** @0.699). Kết quả: **đường biên phòng thủ 13/14** — mỗi ô ❌ chốt một định luật.

| Ô | bpw | geo6 | Phán quyết |
|---|---|---|---|
| **A1 mixwj** (ja 20→32 câu calib) | 1.023 | **716** | ✅ **ja 5814→1854 (−68%)**, đều ×11→×13 — VÀO CÔNG THỨC |
| A2 awq-gauge α=.25/.5 | 1.023 | 801/903 | ❌ scale học được đã bao việc của AWQ; α to nổ ja |
| A3 snip mask (\|W·∇W\|) | 1.023 | 838 | ❌ Wanda đủ tín hiệu ở granularity này |
| A4 calib-big ×8 (976 câu) | 1.023 | 2159 | ❌❌ ja NỔ 476k — **chất lượng phân bố ≫ số lượng** |
| A5 calib-seq 192 | 0.699 | 1721 | ❌ ngữ cảnh dài hút stats về vi/en, bỏ đói ja |
| B1a/b absorber SVD(orig−Q) r8 | 0.783/0.700 | ❌×1.5–2 | tháo phần bù sequential (họ mask-refresh) |
| B1a2/b2 absorber SVD(Wfp−Q) r8 | — | ☠ 17–100M | **nhiễu STE ô pruned** (xem dưới) |
| B2 guard6 (6 block 2:4 + 22 block 1:8) | 0.885 | 957 | trung tính — nằm ĐÚNG đường nội suy (938), không xuyên |
| B3a 2:8 | 1.122 | 735 | ❌ = 1:4 nhưng đắt +0.1bpw (entropy mask 0.60 vs 0.40) |
| B3b cascade 1:8 + residual 1:32 | 1.030 | 912 | ❌ thua 1:4 25% cùng giá bit |

**Hai cơ chế giết absorber weight-space (đóng cửa vĩnh viễn, chỉ còn đường activation-fit exp_p):**
1. Đích `orig−Q`: orig−Wfp là phần bù S1 cố ý tạo — mài về orig là THÁO nó (×1.5–2, đo 2 bậc).
2. Đích `Wfp−Q`: gradient STE chảy vào cả ô pruned nhưng ô pruned KHÔNG ảnh hưởng output
   → không có phản hồi sửa sai → Adam đẩy chúng random-walk ~0.1 (gấp 5 trọng số thật 0.02)
   → Wfp−Q = 87.5–90% nhiễu → SVD chọn đúng hướng nhiễu to nhất → 17–100 TRIỆU PPL.
   **VỆ SINH: ô pruned của Wfp sau S1 là NHIỄU — cấm đọc trực tiếp** (bake dùng quant() nên an toàn).

**Pattern xuyên suốt:** mọi format tái phân bổ (guard6/2:8/cascade) đều mua ja bằng cách bán miền
khác — không ai xuyên biên. Đòn ja duy nhất miễn phí là CALIB (mixwj). Frontier uniform-tier +
S1-v2 hiện là điểm tựa vững; muốn xuyên tiếp phải đổi vũ khí (KD-mix ở S2 — GEN4 đang chạy;
activation-fit absorber; hoặc model to hơn).

**GEN4 (đang chạy):** 4 bậc × [mixwj + perm + v4 + **KD-mix** (fineweb-edu/stack-smol/fineweb2-zh/
openwebmath, ja 60/40, vi24/ja36/en15/code10/zh10/math5) + **best-geo6** + mẫu sinh chữ @best]
— lần đầu S1, KD-data và tiêu chí chọn checkpoint CÙNG nhìn đủ 6 miền. steps 5000, ~$5.
Vận hành: Modal PREEMPT giết container giữa run → hàm tự restart TỪ ĐẦU (đốt lại cell đã xong);
results.json theo tag nên số cũ không mất — thiết kế cell idempotent là đúng.

## Bài 15 — GEN4: đường biên 4 bậc × 6 miền HOÀN CHỈNH (03/08 đêm)

Công thức chốt sau 14 ô screening: **S1[mixwj + perm-gauge] + v4-KD[KD-mix 5 miền chủ động:
vi/ja/en/zh/math, ja 60/40] + chọn best geo6 + mẫu sinh chữ @best**. 5.000 bước/bậc, L40S ~3.2h/$6.
(Code-KD vắng mặt: the-stack-smol gated, smollm-corpus trả 0 dòng — code chỉ hồi thụ động.)

| bpw | vi | ja | en | code | zh | math | val-100 vi | geo6 | val-100 vs gen2.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **1.566** | **319** | **926** | **183** | 41 | **338** | **22** | **350.9** | **159** | 360 → 351 |
| **1.023** | 536 | 1129 | 408 | 98 | 584 | 33 | **555.1** | **278** | 571 → 555 |
| **0.699** | 954 | 3310 | 972 | 241 | 1218 | 47 | **1059.7** | **589** | 1299 → 1060 (**−18%**) |
| **0.616** | 1201 | 3000 | 1047 | 472 | 1276 | 55 | **1320.9** | **708** | 1428 → 1321 (−7.5%) |

Đối chiếu FP (69/125/34/2.7/51/5.5):
- **GEN4 thắng gen2.5 trên val-100-vi Ở CẢ 4 BẬC** — trong khi gen2.5 giết en/code/zh (en ×808-class,
  zh 10⁵). Cân bằng 6 miền KHÔNG mất đỉnh vi — trade-off "chuyên hóa vs đều" bị hóa giải bởi
  kiềng 3 chân: S1-calib + KD-data + tiêu chí chọn cùng nhìn 6 miền.
- 1.56bpw: tỉ lệ ×FP = vi 4.6 / ja 7.4 / en 5.4 / code 15 / zh 6.6 / math 3.9 — **đều ×3.9**
  (gen2: ×400). zh hồi ×1000 so gen3 (335.836 → 338) — thuần công KD-mix.
- **Vân tay của mắt xích thiếu**: code suy nhanh nhất theo bậc (×15→×36→×89→×175) — đúng miền
  duy nhất KHÔNG có KD chủ động. Định luật mỏ-neo miền hiện nguyên hình lần 5. Việc kế: nguồn
  code công khai đã kiểm chứng (hoặc HF token vào Modal secret) rồi rerun 2 bậc thấp.
- **Hành vi (mẫu greedy in-run)**: en @1.56 vượt ngưỡng ngữ pháp ("...it is important to learn
  the language. So, the language is the main source..."); code @1.56 có hình dạng Python thật;
  vi vẫn lặp cụm ở PPL 319. Các bậc ≤1.02: mọi miền còn loop. **Ngưỡng hết-loop ≈ ×3–5 FP,
  tùy miền** — mục tiêu hành vi = kéo từng miền qua ngưỡng riêng, không phải hạ geo chung.
- Tái lập xác nhận: S1 của o1 (gen4) = ô A1 screening TỪNG CHỮ SỐ (cùng seed/config).
- Vận hành: session local chết giữa chuỗi — app `--detach` chạy tiếp trọn 3 bậc còn lại, ckpt
  đủ 4 bậc trên volume. Trạng thái ví: ~CẠN sau GEN4 (~$6). Mọi run kế cần nạp credit.

Checkpoint: `/vol/out/qat_gen4_{n4,o1,o2,o3}.pt` (chưa tải về — chế độ 4G, gate server-side).

## Bài 16 — Đêm 03/08: GEN4.1 (code-KD) + exp_v 30B-A3B streaming, 4 định luật mới

**GEN4.1 — vá chân code (nguồn codeparrot-clean, kiểm chứng datasets-server):**
0.70bpw: geo6 589→**531** (code 241→**144**, −41%) · 0.62bpw: 708→**613** (code 472→**204**, −57%).
Giá: vi/val-100 +7–12% (6k câu code chiếm chỗ vi trong KD). Định luật mỏ-neo lần 6 — chiều dương.
Bẫy đã ghi: smollm python-edu chỉ chứa blob_id (0 dòng); the-stack gated.

**exp_v — S1-only STREAMING cho Qwen3-30B-A3B (exp_v_s1_stream.py):** model bf16 61GB ở CPU RAM
144GB, L40S cầm từng block; gauge per-expert + perm toàn cục (gồm cột ROUTER + lm_head untied);
bake-as-you-go 1 pass. Neo FP 30B lần đầu: vi 29.0/ja 54.2/en 13.7/code 2.3/zh 15.9/math 4.6.
Cổng bất biến QUA cả 2 run (29.0→29.1/28.9). 48 block: 27–50 phút, $1.3–2.7/run.

| 30B @2:4 | vi(16c) | ja | en | code | zh | math | geo6 | val-100 vi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 (calib 122, 60 st/blk) | 1123 | 1443 | 5123 | 1052 | 3228 | 131 | 1243 | 1604 |
| v2 (calib **529 thật**, 100 st/blk) | 1274 | 3653⚠ | **519** | 979 | **2394** | **97** | **905** | **472** |

**4 định luật mới (kỳ vọng ghi trước ×2–5 TRƯỢT — và trượt có giáo trình):**
1. **Luật active-capacity**: MoE chịu nén theo cỡ KÍCH HOẠT (~3B), không theo tổng 30B — prior
   "model to chịu nén tốt" là kinh nghiệm dense, ngoại suy theo tổng params là sai.
2. **Đói-expert đo được**: 122 câu ÷ 128 expert ≈ 540 tok/expert → mask nhiễu + phần bù không được
   train + router drift. Calib ×4.3 câu thật → **en −90% (×374→×38), val-100 vi −71%**.
3. **Probe 16 câu ĐÁNH LỪA ở bit thấp**: probe vi nói +13% trong khi val-100 nói −71% — mọi kết
   luận 30B từ nay phải neo bằng thước lớn (bài học small/large gap, tái phạm lần 2).
4. **Code calib cần GIỮ CẤU TRÚC**: prep kd_mix ép mẫu thành 1 dòng (mất newline/indent) → expert
   code nhận phân bố lạ → calib code ×4 mà code chỉ −7%. Fix: escape newline (việc v3).
   ja probe ×2.5 sau calib-to — cần thước lớn ja trước khi kết luận (nghi seesaw share).

So chuẩn: 0.6B full-KD @1.56 val-100 = 351; 30B **S1-thuần** = 472 — tầng KD cho 30B là chỗ ăn
kế tiếp (cần multi-GPU/offload, thiết kế riêng). Ckpt: expv_Qwen3-30B-A3B_2x4.pt (v2, 61GB, volume
tritue12) + qat_gen41_o2/o3.pt. Chi đêm ~$7.5, ví mới còn ~$22.5.

## Bài 17 — Chiến dịch 30B khép vòng: KD-frontier, định luật F0 tái lập, 2 probe kiến-trúc-mới (04/08)

**Đường biên 30B-A3B @1.56bpw hoàn chỉnh (val-100 vi/ja | geo6 probe):**
| Bậc thang | vi | ja | geo6 | Ghi chú |
|---|---:|---:|---:|---|
| S1-g64+bias (bản lab) | 407.2 | 168.1 | 542 | calib 574 câu thật, 3 lần tái lập từng chữ số |
| + LoRA-KD 1.4M token | **383.5** | 243.2 | **233.9** | code −83%, zh −67%, en sinh đúng ngữ pháp+fact |
| Ép hậu kỳ sang TQ2_0 | — | — | **2.3 TRIỆU** | THUẾ RÚT-XƯƠNG-SỐNG: scale-mịn + bias là chỗ model DỰA vào |
| S1-TQ2native (train trong khuôn) | 524.7 | 173.2 | 677 | thuế in-frame chỉ **×1.25** — định luật F0 định lượng ở 30B |
| + LoRA-native | (chết cạn ví trước khi lưu) | | | cần ~$9 để hoàn tất |

**Định luật F0 (lần 2, giờ có số 30B): train-trong-khuôn ×1.25 vs ép-hậu-kỳ ×4.000.**
Bias là phép dịch hằng số — LoRA/norm không biểu diễn nổi → không heal hậu kỳ được.

**Luồng kiến-trúc-mới (H1/H2/H3, $0, 20 phút):** H1 gauge-folding ✗ (real=98.4% null);
H2 response-surface ✗ (intrinsic dim 645/1024 @95%). Hội tụ: thông tin transformer đã train
là ĐẶC — free-lunch chỉ ở tầng format (đã vét bằng gauge/perm/mask). H3 hot-core+cold-overlay
cổng entropy còn sống (không đấu R(D) — tái phân bổ chi phí theo token; dự án tuần).

**Đối đầu IQ1_S (đính chính quan trọng):** unsloth UD-"IQ1_S" thực chất **2.37bpw** (8.42GB) —
không đủ chuẩn ≤1.56; làn true-≤1.6bpw cho model này đang TRỐNG. IQ1_S/Q4_XL cùng thước llama
= ×1.11-1.13. Đòn học được: bảo vệ attention (+0.07bpw) — ứng viên exp_v v4.

**Trạng thái artifact (04/08 05:00):** ckpt S1-native 61GB + tlogits trên volume free30 (an toàn);
4/4 ví Modal cạn (~$120/3 ngày cho toàn chiến dịch). Còn thiếu: pack $1.5 (→ GGUF 8.5GB chạy
llama.cpp máy B) + tùy chọn LoRA-native $9. Suite test local đã soạn sẵn (run_local_suite.ps1).

## Bài 18 — TQ33: format 1.5bpw tự chế, codec lossless + kernel AVX2 chứng minh (04/08, $0)

Câu hỏi user: *"model 2:4-ternary là 1.56bpw, sao đóng gói TQ2_0 lại thành 2.06 —
sao không build riêng?"* → Trả lời bằng format thật (chi tiết: `RESEARCH_TQ33.md`).

**Format**: nhóm-4 ≤2 nonzero ∈ {−1,0,+1} → đúng **33 pattern**; cặp nhóm = 11 bit
(33²=1089≤2¹¹); **block 64 trọng số = 12 byte** (11B codes + 1B scale-idx) = **1.500 bpw
chẵn**, mọi chiều chia hết 64, LUT decode 1089×8 = 8.7KB (L1). Trần Shannon đo trên ckpt
thật = 1.36 bpw (0nz 7.9%/1nz 36.4%/2nz 55.7%) — fixed-rate chỉ trả thêm 0.14 cho O(1) access.

**Bằng chứng ($0, máy B, tensor thật gen4-0.6B):**
| Kiểm | Kết quả |
|---|---|
| Codec Python 6/6 tensor | tách (t,s) exact, 0 vi phạm 2:4, encode→decode **lossless bit-level** → PPL giữ nguyên THEO ĐỊNH NGHĨA |
| Kernel C fp32-exact vs y_ref | rel err 2.9e-07 ✓ (build `python -m ziglang cc`, không cần VS/cmake) |
| Dot ternary AVX2 | `maddubs(\|t\|, sign(xq,t))` — 32 trọng số/lệnh, số 0 tự triệt |
| DRAM-stream 151MB, 6 luồng | kernel v3 **19.3 GB/s** → chiếu 30B-A3B **~31 tok/s** linear-stream, ~21–24 end-to-end |
| Bài học kernel | v2 vpgather CHẬM hơn buffer+reload (2.6 vs 5.6 GB/s/luồng) — gather AVX2 đắt trên Core Ultra; v3 = buffer + FMA-acc/hàng, hsum 1 lần |

**Ý nghĩa**: 30B-A3B S1-native → **~6.2GB** (vs TQ2_0 8.5GB, "IQ1_S" 8.4GB, Q4 16.5GB),
tốc độ dự phóng **+35–50%** so Q4/IQ1 đo thật (16.0/16.5 tok/s). Format ĂN ĐƯỢC chỉ với
gia phả S1 (2:4 + scale f8-grid — đo thật: 43 giá trị duy nhất/tensor). Phase 2 = GGML_TYPE_TQ33
trong llama.cpp fork hoặc runner riêng 0.6B (4–7 ngày). Files: `exp_t24_codec.py`,
`exp_t24_export_bench.py`, `tq33_bench.c`.

**Modal 04/08 sáng**: free30 vẫn spend-limit → pack TQ2_0 vẫn blocked $3; TQ33 không đổi
kết luận này (TQ33 convert từ CÙNG ckpt S1-native, làm local được khi có runner).

## Bài 19 — TQ33 30B-A3B: từ file 1.5bpw đến runner MoE chạy thật, validate 48/48 layer (04/08)

Sau Bài 18 (format TQ33 thiết kế + microbench), hoàn tất toàn bộ chuỗi: tải ckpt 30B →
encode → dựng runner MoE → đo tốc độ thật. Chi tiết đầy đủ: `RESEARCH_TQ33.md` (encode +
CRC saga) và `RESEARCH_TQ33_RUNNER_30B.md` (runner + validate + tốc độ).

**Sự cố CRC-32 khi tải 61GB (~2h, bài học đáng nhớ):** 2 lần tải độc lập lỗi ở 2 chỗ hoàn
toàn khác nhau (transfer-side, không phải hỏng nguồn). Bẫy: CRC-32 chỉ bảo vệ DATA của entry
zip, không bảo vệ local header — lỗi chạm header khiến cả zipfile chuẩn lẫn code tự viết tính
sai vị trí data, báo "hỏng" oan. Fix bằng suy offset từ bất biến alignment-64 thay vì tin
header — tự phục hồi 42/112 tensor. 70 tensor (41/6144 expert) hỏng thật không phục hồi được
sau 3 lần thử → zero-hoá minh bạch, ghi rõ trong `corrupted_zeroed.json`.

**Bug bpw tự bắt được:** bản encode đầu lưu scale float32 thô → 1.875bpw thay vì 1.5 thiết
kế. Sửa bằng codebook 1-byte/tensor (≤256 giá trị, đã verify). **Kết quả cuối: 30B TQ33 =
6.88GB** (5.61GB linear @1.501bpw + 1.27GB embed/lm_head/norm/router giữ bf16).

**Runner MoE (`qwen3moe_runner_tq33.c`)**: mở rộng runner 0.6B (Bài 18), thuật toán routing
xác nhận qua đọc trực tiếp source llama.cpp (`build_moe_ffn`): softmax-128 → top8 → renorm
→ SwiGLU/expert → weighted-sum, không bias/scale/shared-expert. Validate 2 lớp (thuật toán cô
lập rel-err 2.7e-7; TOÀN BỘ 48/48 layer thật vs oracle đọc ckpt gốc — routing khớp hoàn hảo ở
mọi lựa chọn tự tin cao, lệch chỉ ở nhiễu-số-học-ranh-giới). Đa luồng theo EXPERT (không theo
GEMV — bài học 0.6B) đạt 9.3-13.2 GB/s @12-14T so với 0.6B chỉ 4.5-5.7 GB/s @6T.

**Tốc độ thật: 5.1-17.9 tok/s tuỳ luồng/nhiệt, dải bền vững 13.0-14.4 tok/s @8+ luồng** — SO
với Q4_K_XL/IQ1_S đã đo trên cùng máy (~16.0-16.5 tok/s): **KHÔNG vượt trội tốc độ** như ước
tính lạc quan ban đầu (lm_head bf16 622MB/token không nén vẫn chiếm 18-37%, vì TQ33-active
thật ~486MB/token — ước tính "10-15MB" ban đầu sai 32 lần). **Giá trị thật của TQ33 là DUNG
LƯỢNG** (6.88GB vs 8.4-16.5GB các format khác) ở tốc độ tương đương, không phải tốc độ vượt
trội. Chất lượng sinh văn bản (ngoài phạm vi đo ở đây) là ckpt **S1-only geo6 677, chưa qua
LoRA-KD** (233.9) — đây là sàn, không phải trần model có thể đạt.

## Bài 20 — Đổi sân: tốc độ trên GPU YẾU, không còn dựa vào sub-bit (02/09, $0)

Quyết định của user: **dừng ép model xuống 1/1.58 bit để lấy tốc độ** — Bài 19 đã đo, sub-bit
chỉ thắng dung lượng. Câu hỏi mới: *chạy trên GPU yếu mà vẫn thần tốc thì đánh vào đâu?*
Nghiên cứu đầy đủ (roofline, bảng engine theo lớp GPU, lộ trình P0–P4): **`RESEARCH_GPU_WEAK_FAST.md`**.

**Bốn điểm cốt lõi:**

| # | Phát hiện |
|---|---|
| 1 | `tok/s = min(BW/bytes_per_token, 1/overhead_per_token)`. Sub-bit chỉ đánh số hạng 1 và đã bão hoà ở 4-bit; ba trục còn dư địa là **overhead/lượt**, **lượt/token**, **token/request** |
| 2 | **v7a 152M trên GPU 8GB không có bài toán dung lượng**: FP16 304MB → trần 1.470 tok/s. Nhưng chạy bằng HF `generate` thì overhead ~200 kernel × 20–30µs = **150–250 tok/s, CHẬM HƠN bản CPU 350 tok/s**. Model nhỏ trên GPU: CUDA Graph là toàn bộ trò chơi (×3–10) |
| 3 | **TQ33 đổi vai, không chết**: 6.88GB là vé duy nhất để 30B-A3B vào 8GB VRAM, nơi băng thông ×10.7 DRAM → chiếu 13 tok/s (CPU) lên **dải 100–200 tok/s**. Kèm điều kiện: nén lm_head (622MB/token, đã đo chiếm 18–37%) + fused sampling |
| 4 | **Ràng buộc 2:4 của S1 là tính năng phần cứng trên Ampere**, vô nghĩa trên AVX2. sm_86 có sparse tensor core ×2; Sparse-Marlin (INT4+2:4, cc≥8.0) đo ~5.3× vs FP16 dense @batch16. Chỉ cần đổi *vỏ đóng gói* sang layout metadata 2-bit/nonzero — **không train lại gì** |

**Cạm bẫy mới ghi nhận:** PCIe 4.0 x16 (~32 GB/s) **thấp hơn** DRAM máy A (42 GB/s) → stream
expert qua PCIe luôn lỗ so với để CPU tính tại chỗ; `--n-cpu-moe` có đồ thị hình V (chỉ "5×"
khi trước đó overcommit VRAM); iGPU Arc máy B **không tăng tốc decode** vì dùng chung DRAM.

**Làm rõ phạm vi (user chốt ngay sau đó):** mục tiêu KHÔNG phải model tự train mà là **dùng
được các model TO hơn** trên 8GB. Bài toán đó đổi bản chất: nút thắt là **residency**, không
phải bit/weight. Nghiên cứu riêng: **`RESEARCH_BIGMODEL_ON_8GB.md`**.

`t_token = f·S/448 + (1−f)·S/42` với S = active byte/token, f = phần nằm trong VRAM ⇒
`speedup ≈ 1/(1−f)`, và **vách đá 10.7× ở ranh giới ~6GB**. Ba hệ quả:
- Cùng model 14B: 6.3GB (IQ3, vừa VRAM) = ~45 tok/s vs 9.0GB (Q4, tràn) = ~8 tok/s. **Tràn VRAM
  đắt hơn mọi tối ưu khác.** Chọn quant theo ngưỡng vừa VRAM, không theo "bit thấp nhất".
- Hybrid: **80B-A3B nhanh BẰNG 30B-A3B** (~22 tok/s, cùng chặn bởi 42 GB/s DRAM) ⇒ chỉ số vua là
  **tỉ lệ thưa total/active**, đi tìm model TO có active NHỎ. 106B-A12B (8.8:1) tụt còn ~9 tok/s.
- **Đính chính cho Bài 19**: sub-bit không tăng tốc kernel, nhưng nó là **vé vào cửa VRAM** —
  30B-A3B ở Q4 (17GB, f=0.35) ~22 tok/s → TQ33 (6.4GB, f=1.0) **~140–210 tok/s**. Kết luận đúng
  phải phát biểu là *"ép bit để vừa VRAM, rồi VRAM cho tốc độ"*, không phải *"ép bit cho nhanh"*.

**Việc kế tiếp ($0, nửa ngày):** đo llama.cpp trên máy A (sweep `--n-cpu-moe`, bật/tắt
`--model-draft` Qwen3-0.6B) với gpt-oss-20b hoặc 30B-A3B để hiệu chỉnh hằng số và xác nhận định
luật trên chính máy mình — TRƯỚC khi tải model 32GB hay viết kernel CUDA.

## Bài 21 — Đo thật 3 ý tăng tốc "không draft model": 1 thắng, 3 chết (02–03/09, $0, ~33 phút máy)

Ba script mới: `exp_bm_resident_routing.py` (OLMoE), `exp_bn_resident_routing_30b.py`
(**Qwen3-30B-A3B thật**, stream từng layer để vừa 48GB RAM, forward viết tay validate bằng
top-1 acc **0,573**), `exp_bo_entropy_gated.py`. Phân tích đầy đủ: `RESEARCH_NOVEL_NODRAFT_STACK.md`
§10–13.

**THẮNG — xếp expert vào VRAM theo TẦN SUẤT (không nằm trong danh sách ý ban đầu):**

| % expert ở VRAM | 12,5 | 25 | 37,5 | 50 |
|---|---|---|---|---|
| f nếu đặt theo tỉ lệ | 0,125 | 0,250 | 0,375 | 0,500 |
| **f đo được (30B-A3B)** | **0,414** | **0,627** | **0,773** | **0,875** |

⇒ **×1,7–2,0 tốc độ** (Sens 2,71bpw: ~68 → ~138 tok/s), **không đổi một bit trọng số, không rủi
ro chất lượng**. Rào cản: GGUF fuse toàn bộ expert của một layer thành MỘT tensor nên
`--n-cpu-moe`/`-ot` chỉ tách theo *layer* — **`qwen3moe_runner_tq33.c` tách được theo *expert*,
llama.cpp không.** Đây là lý do tồn tại rõ ràng nhất cho runner tự viết.

**CHẾT 1 — draft = tập expert resident**: α đo được tốt (0,558 @25%, 0,686 @50%) nhưng vô dụng,
vì `gain = E(α,K)/(K·d+1)` với `d = t_draft/t_full`: placement càng tốt thì t_full càng nhỏ, d
càng lớn. Ở f=0,85, **kể cả draft hoàn hảo α=1 thì trần chỉ +27%**. Hai ý ăn cùng một miếng,
placement thắng tuyệt đối.

**CHẾT 2 — entropy-gated precision**: cần agreement ≥99,5% ở token dễ; đo được **0,266 (int2)
và 0,020 (ternary) ở decile DỄ NHẤT**, 0/10 decile đạt ngưỡng. Giả thuyết "tổn thất nằm đúng
chỗ model đang tự tin" **sai về cơ chế**: entropy là thuộc tính của model GỐC, còn plane rẻ
không phải "gốc + nhiễu" mà là model khác hẳn — sai số tương quan giữa layer và dồn theo độ sâu.
⇒ **Hệ quả tổng quát: entropy của model mạnh không phải cổng hợp lệ để chuyển sang model yếu.
Cổng đúng phải là độ tự tin của chính plane rẻ, tức phải verify — nên bước verify KHÔNG phải
chi tiết cài đặt bỏ được, nó chính là thứ làm low-bit drafting an toàn.**

**CHẾT 3 — dùng chính bản 1,5bpw làm draft**: bác bỏ bằng tính toán (§0) — `d=0,56` ⇒ chậm hơn
26%. Nest theo **độ sâu** chia sẻ được tính toán (v=1−d), nest theo **độ chính xác** thì không (v=1).

**Phụ phẩm dùng được ngay cho mọi script cũ**: fp16 matmul trên CPU Zen3 chỉ **65 GFLOPS**,
fp32 đạt **250**. Giữ trọng số fp16 + upcast fp32 trong matmul = **189 GFLOPS, nhanh 2,9×,
không tốn thêm RAM** (6 dòng patch `F.linear`). `exp_bc` và mọi exp dùng `DTYPE=float16` trên
CPU đang chạy chậm 3–4× không cần thiết.

**Proxy hợp lệ**: OLMoE (64 expert) cho f = 0,47/0,68/0,80/0,88; 30B-A3B (128 expert) cho
0,41/0,63/0,77/0,88 — trùng gần khít từ 37,5% ⇒ vòng lặp sau dùng OLMoE cho nhanh được.

## Bài 22 — ĐO CHẤT LƯỢNG TRÊN CHÍNH 30B: bảng độ nhạy KHÔNG chuyển giao (03/09, $0)

`exp_br_quality_30b.py` + `exp_bs_sensitivity_30b.py`. Đầy đủ: `RESEARCH_NOVEL_NODRAFT_STACK.md`
§16–18. **Chưa ai từng đo bảng độ nhạy trên 30B** — con số ×1,53 của `exp_be` (OLMoE 1B-7B)
đang bị dùng như thể là của 30B.

**Phát hiện chính: thủ phạm ĐỔI CHỖ.** Ablation trên Qwen3-30B-A3B (miền prose):

| Bước | PPL ×base | agreement | flip@conf>0,8 |
|---|---|---|---|
| **+down=ternary 2:4** | **×1,189** | **0,856** | **3,97%** |
| +gate_up=int3 | ×1,273 | 0,829 | 5,75% |
| +attn=int4 | ×1,265 | 0,810 | 6,58% |

`down=ternary` **một mình gây phần lớn thiệt hại** (14,4% token lật). Trên OLMoE thì NGƯỢC LẠI:
down là bước an toàn nhất (×1,35), gate_up là kẻ sát nhân (ternary → ×1700). Nguyên nhân:
30B-A3B có `moe_intermediate_size=768` với 128 expert — **expert HẸP nên ít dư thừa**, 2:4
ternary (zero-hoá 50%) cắt vào phần thật. `exp_bf` từng thấy OLMoE→Qwen3-0.6B chuyển giao được,
nhưng cả hai đều là model nhỏ expert rộng.
⇒ **Bảng độ nhạy phải dựng lại trên model đích, không port từ proxy.**

**Chất lượng thật của config 2,71bpw trên 30B, và nó PHỤ THUỘC MIỀN**: prose ×1,265 / agreement
0,810 / flip 6,6%; **code ×1,688 / agreement 0,707 / flip 8,7%**. Tức **19–29% token sinh ra
khác đi** (Q4 so fp16 thường giữ 97–99%). PPL che mất điều này — xác nhận "PPL là máy báo cháy,
fidelity mới là thước".

**Hot set không sống qua đổi miền**: profile vi_text → chạy code mất **25–34 điểm** coverage
(0,750 → 0,422 ở N=32), overlap tập hot chỉ 0,32–0,67. ⇒ **đặt TĨNH là sai, phải LRU động.**

**Đối chiếu thực địa — ×2,06 của Bài 21 là TRẦN, không phải kết quả**: llama.cpp PR #27861 đã
hiện thực hoá đúng ý (`--moe-expert-cache N`, LRU động). Số thật: **RX 7600 8GB + Qwen3-30B =
16,5 tok/s @32 slot (+14,6%)**; dual 3090 +31%; dải 13 model +10…+57%. Hit rate họ báo
60,4–74,7% **khớp đúng f-curve đo được** — đo lường đúng, quy đổi tốc độ sai. Ba lỗi: (1) dùng
BW_RAM 42 GB/s trong khi **Bài 19 của chính dự án đã đo 9,3–13,2 GB/s** — tính lại ra 17,3 tok/s,
khớp mốc thật; (2) giả định expert resident được TÍNH trên GPU, còn PR giữ `MUL_MAT_ID` ở CPU;
(3) mượn số chất lượng của 7B cho 30B. Cũng đính chính: `qwen3moe_runner_tq33.c` là **1262 dòng
CPU/AVX2, zero CUDA** — không đặt expert vào VRAM được.

**Cảnh báo**: tác giả PR đo model **512 expert** và thấy *"no exploitable static skew"* ⇒ mẹo
hot-expert gắn với model **128 expert**; lớp Qwen3-Next-80B-A3B (512 expert) routing quá phẳng.

**KẾT LUẬN TRILEMMA (8GB)**: chất-lượng+to → **30B-A3B Q4_K_M + expert-cache ≈ 17–22 tok/s**;
nhanh+chất-lượng → 8B Q4 ≈ 60 tok/s; to+nhanh → phải trả 19–29% token đổi. Không có cấu hình
đạt cả ba.

## Bài 23 — llama.cpp trên máy A: đo thật, 3 bản vá code, và trần của hybrid decode (03/09, $0)

Đầy đủ: `RESEARCH_MEMORY_HIERARCHY_2026.md` §7–14. Tóm tắt các con số **có sai số** (llama-bench,
r≥4, xen kẽ, GGUF trên SSD, `--load-mode mmap`, Qwen3-30B-A3B **Q4_K_M nguyên bản**):

| Cấu hình | tok/s |
|---|---|
| `-ngl 99 -ncmoe 48` (mọi expert ở CPU) | 20,9 |
| `+ --moe-expert-cache 32` (PR #27861) | 26,3 |
| **`-ngl 99 -ncmoe 33 -t 4 -fa on -ctk q4_0 -ctv q4_0`** | **28–31** ← đáy hình V, cấu hình khuyên dùng |
| `-ncmoe 30` (tràn VRAM) / `cache 64` | 26 / 12 |
| CUDA graphs OFF | −12…15% ⇒ để ON |
| `--poll 0/100`, `--prio 2`, KV f16 vs q4_0 | ±0,3 (nhiễu) |
| pin core `-C 0x55` | **−10…13%** |

**Ba bản vá code đã làm & đo — đều không cho tốc độ**: (1) chia `mul_mat_id` theo expert
(`GGML_MOE_EXPERT_PARALLEL`): 0–5%, không bền; (2) `cudaDeviceScheduleSpin` cho cc8.6 (upstream chỉ
áp cc12.1): 0%; (3) shim MSVC `flockfile` (chỉ để build). Ý L3-prefetch expert layer kế bị
**microbench giết trước khi code** (`l3_prefetch_bench.c`): GEMV Q4×int8 đã đạt **43,7 GB/s = trần
DDR4**, L3-ấm chỉ ×1,43, pipeline prefetch **chậm 2×**. Tiền đề routing-dự-đoán-được thì đúng
(`exp_bv`: top-12 phủ 0,971) — để dành cho tầng NVMe sau này.

**Ngân sách token (~35ms)**: ~16ms GEMV expert CPU (ở trần DRAM) · ~3ms GPU · **~16ms cho 33 lần
chuyển giao CPU↔GPU** (0,3–0,4ms/split: hoàn thành GPU + D2H + launch WDDM). Không knob nào ở
tầng ứng dụng chạm tới phần thứ ba; chỉ **khấu hao** được bằng speculation (`llama-server
--spec-type ngram-*`, 0 tải — đang đo bench13) hoặc **phần cứng**.

**Speculation không draft model (bench13, `llama-server --spec-type ...`)** — 0 tải, lossless:
**`ngram-mod` (n-max 3): prose +10% / code +11% / list +29%, acceptance 100%, không thua ở đâu**
← khuyên dùng. `ngram-simple` n-max 2: +2/0/+23%; n-max 3: +9/−3/+20%; n-max 6 lỗ. `ngram-map-k`
thua (−8% prose). Đây là đòn duy nhất trong ngày cho thêm tốc độ thật trên cấu hình đã tối ưu.

**Draft model thật (bench14, Qwen3-0.6B Q4 `-md -ngld 99 --spec-type draft-simple`)**: **LỖ 30–38%**
ở ncmoe 33 (16–17 tok/s vs 24–28), còn −10…−25% ở ncmoe 36; acceptance chỉ 42–46% và **giống nhau
ở mọi loại prompt**. Cơ chế: lượt verify K token trên MoE hybrid phải đọc **hợp expert của K token**
(tới 24/layer) từ RAM ⇒ phí tăng nhanh hơn lợi; draft cũng chiếm ~0,5GB VRAM ở ngân sách kịch.
⇒ Trên máy này speculation phải **chọn lọc cao**: `ngram-mod` (100% accept) — không dùng draft
chạy liên tục. Số "+2–3×" của spec-decode datacenter **không chuyển** sang hybrid CPU/GPU MoE.

**Model kế tiếp vừa 32GB RAM (arch đã hỗ trợ trong build)**: **Qwen3.6-35B-A3B Q4_K_M 22,1GB**
(256 expert, DeltaNet hybrid — ứng viên #1), ERNIE-4.5-21B-A3B ~13GB (nhanh hơn), gpt-oss-20b
~12GB, Nemotron-3-Nano-30B-A3B ~18GB. Không vừa: Qwen3-Next-80B, GLM-4.5-Air, gpt-oss-120b.

**TÓM TẮT NGÀY 03/09 (đọc cái này trước, chi tiết theo mốc giờ ở dưới và §18 doc)**
1. **Model & cấu hình**: Qwen3.6-35B-A3B UD-Q4_K_M, `-ngl 99 -ncmoe 30 -t 4 -fa on -ctk q4_0 -ctv q4_0 --load-mode mmap+mlock -c 8192
   -b 2048 -ub 2048 --reasoning off` ⇒ **42 tok/s decode** (khi GPU ở P-state cao), **pp2048 ≈ 700 tok/s**. Nhanh hơn 30B-A3B ~45% vì
   expert FF 512 × 40 layer (0,42 GB/token trên CPU vs 0,70). Chọn model MoE theo `expert_ff × layer × top_k`.
2. **Nguyên nhân "lúc 42 lúc 30"** (mất cả buổi chiều mới tìm ra): **driver NVIDIA hạ GPU về P3 (SM 780/2100 MHz, mem 5001/7001)**
   khi thấy decode hybrid chỉ bận ~50% — không phải mmap, không phải RAM, không phải đổi model. Chữa: NVIDIA Control Panel → Power
   management mode = *Prefer maximum performance* (hoặc `nvidia-smi -lgc 1800,2100`). Luôn log `nvidia-smi --query-gpu=pstate,clocks.sm`
   khi đo. (15z kiểm nhân quả bằng spinner giữ util.)
3. **Sửa code thật (một ký tự)**: `ggml/src/ggml-cuda/mmvq.cu` `should_use_small_k`: `<` → `<=` ⇒ MUL_MAT_ID q4_K batch 1: 66 → 45 µs
   (×1,47), q4_0 23,5 → 21,7. Trường hợp biên k=2048 (expert MoE FF nhỏ) bị heuristic upstream bỏ qua. Ứng viên PR. Lãi hệ thống ~+2–3%
   (chỉ đo được khi khóa xung GPU).
4. **Đòn bẩy config rút từ đọc source**: prefill của layer expert-CPU thực chất là chép weight lên GPU mỗi ubatch (offload batch ≥ 32,
   PCIe-bound) ⇒ `-ub 2048` tăng pp 2,2–2,3×.
5. **Đã đóng**: speculation n-gram (0 draft trong dùng thật, cả 3 cơ chế), draft model thật (lỗ), residency-aware quantization (bản sạch
   chỉ +1–3%, chất lượng bám gốc — ưu tiên thấp), chừa VRAM (−1 tok/s/layer), KV type (không khác), thread 5/6 (+2%/0), sub-4-bit trên CPU
   (iq2_xs chậm 2,4× q4_K).
6. **Ngân sách token 35B (24 ms ở P-state cao)**: DRAM expert CPU ~9,8 ms + GPU byte ~2,5 ms + kernel GPU nhỏ ~3–6 ms + sync ~1,6 ms
   (11–34 µs/lần, KHÔNG phải 0,3 ms như từng nghĩ) + CPU op/scheduler ~2–3 ms.

**[15:53] 15z — spinner giữ xung: giữ được P2 nhưng vô dụng**: kernel spin chiếm timeslice GPU ⇒ llama sụp 12,5 ± 16,8 (Z1 không spinner:
39,9 ± 0,7 ở P2). Z3 ngay sau khi tắt spinner: 28,2 ± 2,5 (P3 trở lại). Kết luận: P-state chỉ chữa được bằng cài đặt driver (NVCP Prefer maximum
performance / `nvidia-smi -lgc`) — việc của bạn. Chuỗi 15a→15z kết thúc 15:53.

**[15:45] 15x — tương quan P-state ↔ tốc độ đủ hai phía**: cửa sổ GPU giữ P2 suốt bench (SM 1756, mem 6268) ⇒ **39,3 ± 0,5**; các cửa sổ
P3 780 MHz ⇒ 27–31; cửa sổ chuyển P2→P3 ⇒ σ 6. A/B patch mmvq ở mức hệ thống (X1 33,3 / X2 29,7 / X3 39,3) bị P-state nuốt mất — phải
khóa xung GPU rồi đo lại theo cặp. Kết quả op-level (×1,47) vẫn đứng.

**[15:33] 15v — residency-aware quant bản SẠCH chỉ +1–1,5%** (V3 31,0 vs gốc 30,7, trạng thái chậm P3): lãi +4–6% của v2 là do
recipe UD bị ghi đè. Chất lượng V3 bám sát gốc (list 99% từ trùng, prose 89%, prefix 827 ký tự) — an toàn nhưng lãi nhỏ. Kết luận: đúng nguyên lý, ưu tiên thấp;
patch mmvq (miễn phí) và giữ GPU ở P-state cao quan trọng hơn.

**[15:31] 15u — chất lượng RESQ v2**: server GỐC 41,9 vs RESQ v2 43,7 tok/s (+4,3%, trạng thái nhanh) nhưng đầu ra lệch mạnh (prose/list
chỉ 56–57% từ trùng, đổi hướng từ ký tự đầu) ⇒ ghi đè recipe UD làm đổi model. Không dùng v2; chờ 15v (bản sạch, 21,69 GB).

**[15:30] BẮT QUẢ TANG "TRẠNG THÁI CHẬM" = GPU KẸT P-STATE THẤP**: trong lúc bench 35B đang chạy, nvidia-smi báo **P3, SM 780 MHz
(max 2100), mem 5001 MHz (max 7001)**, 56 W, 52 °C — driver hạ xung vì GPU chỉ bận ~50% (decode hybrid). Giải thích trọn vẹn mọi lần
chậm 28–30% độc lập mmap/mlock/dio/model/RAM và tự lật. **Cách chữa (người dùng làm)**: NVIDIA Control Panel → Power management mode =
*Prefer maximum performance* (hoặc `nvidia-smi -lgc 1800,2100`). Đang log tiếp để đối chiếu trạng thái nhanh.

**[15:14] 15y — giả thuyết phân mảnh LUNG LAY**: xả standby thất bại (thiếu quyền), Y1 vẫn 30,5, nhưng **Y2 nạp lại ngay sau = 42,3**
— trạng thái tự lật về nhanh, không can thiệp. Nghi mới: **GPU kẹt P-state thấp** (nvidia-smi lúc 14:00 báo mem 5001 MHz thay vì 7001).
Đang log nvidia-smi 2s/lần song song các vòng còn lại để đối chiếu tg ↔ xung. Nếu đúng: NVIDIA Control Panel → Power management =
Prefer maximum performance.

**[03/09 19:57] Antigravity đợt 2**: đã xuất GGUF cắt expert thật (sửa `qwen3moe.cpp` cho số expert khác nhau theo tầng). Tốc độ thật khi
vừa VRAM: **107–124 tok/s** (32–42 expert/layer, 5–6,3 GB) — đúng roofline. Nhưng **mọi bản đều sinh rác** (giai thừa → giỏ hàng; FPS42 →
ký tự vô nghĩa), kể cả bản 68 expert 9,6 GB; dự đoán dung lượng thấp hơn thật 1,1–1,7×; so sánh "data-free" dùng input ngẫu nhiên nên
vô nghĩa. Chi tiết §18.31.

**[03/09 tối] Đánh giá kế hoạch "0.0x bit theo độ nóng expert" (Antigravity)**: số dung lượng thiếu ~2× (config đã đo ở Q4_K_M
thật là 9,8 / 14,2 GB, không phải 5,72 / 7,20); chất lượng tự đo được (đúng bộ thước của ta) chỉ **41–54% trùng top-1, PPL ×1,35–1,89**
trong điều kiện rò rỉ dữ liệu profile; hybrid 3 tầng chưa đo end-to-end, đo 1 layer với input ngẫu nhiên còn kém ternary đều.
Không đạt được. Chi tiết §18.30.

**[03/09/2026 17:35] TẠM DỪNG theo yêu cầu.** Máy sạch (0 tiến trình, 0 task). DLL = patched (17:29). Khi tiếp tục: khóa xung nhớ → chạy
`run_bench16b.cmd` (A/B patch theo cặp, touch đã sửa) → `win16b.py`. Chi tiết §18.29.

**[17:30] GỐC "trạng thái chậm" = xung GPU, đã bóc tách xong**: driver hạ về **P3 khi decode hybrid** (lý do `gpu_idle`, không phải
công suất 240 W/nhiệt). Trước NVCP: P3 = SM 780 / mem 5001 ⇒ ~30 tok/s. Sau khi bật **Prefer maximum performance**: SM giữ 1800 nhưng
mem vẫn rơi 6801 → 5001 ⇒ **36,7 ± 0,9**. Đủ xung (P2 1800/6801) = 42. Còn thiếu: khóa xung nhớ (`nvidia-smi -lmc 6801,6801`,
`-lgc 1800,2100`; gỡ `-rmc -rgc`) — cài đặt admin. Sửa thêm: A/B patch ở 15x vô hiệu (copy .orig không đổi mtime ⇒ ninja không rebuild);
16b đã sửa (touch), chờ khóa xung rồi chạy.

**[15:08] 15t + bộ đếm RAM — "trạng thái chậm" là PHÂN MẢNH KHUNG TRANG, không phải đổi model**: nạp lại cùng 35B sau 90s/90s
vẫn 29,7/30,0/30,3 (không hồi); Free page list chỉ 454 MB, standby cache 27 GB (trang của các file model cũ). Khi mlock 22 GB phải
thu hồi khung từ standby ⇒ khung rời rạc ⇒ TLB/prefetch kém ⇒ −28%, không tự hồi. Giai đoạn nhanh (13:09–13:57) là khi trang của
CHÍNH file đó còn trong standby (khung liền từ lần đọc tuần tự). 15y kiểm bằng cách **xả standby list rồi nạp lại** (không cần reboot).
Nếu đúng: mẹo vận hành = xả standby trước khi nạp model (RAMMap/NtSetSystemInformation), và hướng code = large page (cần quyền
"Lock pages in memory").

**[15:00] 15s — PATCH ĂN: `mmvq.cu` `should_use_small_k` `<` → `<=` ⇒ MUL_MAT_ID q4_K batch 1: 66,3 → 45,1 µs (×1,47), q4_0 23,5 → 21,7.**
Một ký tự, không đổi kết quả số học, đúng chỗ đã chẩn đoán từ đo lường (trường hợp biên k=2048). Ước +2–3% tg hệ thống cho 35B,
~+3% cho 30B (15 layer expert GPU); 15x đo A/B xen kẽ patched/gốc/patched. Đây là ứng viên PR upstream (Ampere/Ada, MoE expert FF nhỏ).

**[15:05] 15r + đọc source — tìm ra gốc của "q4_K chậm 2,8× trên GPU"**: MUL_MAT dense q4_K chỉ chậm 1,33× q4_0 (300 vs 399 GB/s)
⇒ không phải K-quant "hỏng" mà là **hình dạng expert k=2048**: `should_use_small_k` (mmvq.cu:934) dùng `blocks_per_row_x < nwarps*bpi`;
với q4_K k=2048 là `8 < 8` = false ⇒ rơi đúng trường hợp biên: mỗi thread một lượt unpack nặng, không pipelining. **Patch một ký tự
`<` → `<=`** (15s: rebuild ggml-cuda, đo lại MUL_MAT_ID và llama-bench 35B). Nếu ăn, đây là fix đáng gửi upstream cho mọi MoE có
expert FF nhỏ (Qwen3.6, GPT-OSS…) trên Ampere/Ada.

**[14:55] 15q — residency-aware quant lần 2: +5–6% (32,1 vs 30,5; ncmoe 29 vừa VRAM: 32,9) nhưng nhiễu**: ftype Q4_K_M ghi đè
recipe UD (ssm_alpha/beta F32→Q4_K, attn Q8_0→Q4_K…) ⇒ CPU bớt 3,8% byte ⇒ lãi kernel q4_0 thật chỉ ~+2–3%, chất lượng nghi ngờ.
15v làm lại sạch bằng `--tensor-type-file` ghim đúng kiểu gốc cho 703 tensor, chỉ 30 tensor expert blk.30–39 → q4_0.

**[14:30] 15p — ĐÒN BẨY XÁC NHẬN: `-ub 2048` tăng prefill 2,24×** (35B ncmoe 32, pp2048: ub 512 → 318,6; 1024 → 491,3; **2048 → 714,9
tok/s**). Đúng mô hình đọc từ source: prefill expert-CPU = chép weight lên GPU mỗi ubatch (PCIe-bound) ⇒ ubatch to hơn = chép ít
lần hơn. Đã kiểm: ncmoe 30 vẫn vừa VRAM (308 → 699), 30B 409 → 936 (×2,29); decode không bị chậm đi. **Khuyến nghị chốt: thêm `-b 2048 -ub 2048`.**

**[14:25] 15o — "trạng thái chậm" là thật và KHÔNG phải mmap**: 35B mlock nạp ngay sau 30B → **30,9 ± 0,5** (thay vì 42), ổn định
suốt 8 lượt. Nghi bố trí vật lý trang RAM sau xáo trộn (TLB/prefetch) hoặc Windows quản lý standby nền; **dio cũng chậm y hệt (31,1)** ⇒ không phải page cache/mmap;
lần nào nạp model KHÁC ngay trước đó cũng chậm, nạp lại CÙNG model thì nhanh ⇒ nghi phân mảnh trang vật lý (TLB). 15t phân xử.
Mẹo chắc chắn: giữ một server thường trực, đừng nạp qua lại nhiều model. Bài học tạm: **sau khi đổi model, tốc độ có thể thấp 25–30% một lúc**.

**[14:12] PHÁT HIỆN 15n (CUDA)**: trên GPU, `mul_mat_id` batch 1: **q4_K 66 µs vs q4_0 23,5 µs (2,8×)**, q8_0 40, f16 70 — kernel K-quant
theo `ids` chỉ đạt ~107 GB/s (q4_0: ~300). Trên CPU thì ngược lại (q4_K 89 nhanh nhất, q4_0 114). ⇒ **Lượng tử hóa theo nơi cư trú**:
layer expert trên GPU (blk.30–39 ở ncmoe 30) → q4_0, layer trên CPU giữ q4_K ⇒ ước −1,3 ms/token (+5–6%) cho 35B, VRAM không đổi.
Đây là đòn bẩy code/format-level chưa thấy ai làm; 15q sẽ requantize bằng `llama-quantize --tensor-type` và đo.

**[14:12] Phát hiện từ 15n — vì sao prefill nhanh dù expert ở CPU, và một đòn bẩy cụ thể chưa thử**: op `mul_mat_id` trên
CPU ở batch lớn là compute-bound ~340 GFLOPS ⇒ nếu prefill 30B chạy expert trên CPU thì chỉ ~140 tok/s, nhưng đo được 525.
Đọc source: ggml-backend **offload op có batch ≥ 32 lên GPU kể cả khi weight nằm ở host** (chép weight qua PCIe mỗi ubatch;
`GGML_OP_OFFLOAD_MIN_BATCH`, mặc định 32). 30B: ~11 GB weight expert/ubatch ÷ 12 GB/s ≈ 0,9 s/512 token ≈ 550 tok/s — khớp.
Vì `-ub` mặc định 512, pp2048 ≈ pp512 (chép lại mỗi ubatch). ⇒ **Tăng `-ub 1024/2048` có thể tăng pp 2–4× cho prompt dài**
(đổi VRAM compute buffer) — 15p đang đo. Decode không đổi (K < 32 không được offload ⇒ speculation vẫn chạy CPU, ~K×).

**[14:05] 15n — đo thẳng op expert bằng `test-backend-ops`**: `MUL_MAT_ID` q4_K (128 expert, top-8, 768×2048 = shape 30B, batch 1)
= **89 µs/op** (~79 GB/s khi L3 ấm) ⇒ đường ggml-cpu không nặng overhead; phần ~9 ms/token "chưa gọi tên" của 35B nhiều khả năng là
**~800 kernel GPU nhỏ ở batch 1** (launch/xoay kernel 3–6 ms) — hướng sửa code còn giá trị là fusion kernel GPU (việc upstream),
không phải CPU. Bonus: cùng shape, **iq2_xs (2-bit) 217 µs = chậm 2,4× q4_K** — bằng chứng số cho quyết định bỏ sub-4-bit.

**[13:58] 15l — knob còn lại của 35B**: thread t=4/5/6 → 42,2/43,1/42,6 (giữ t=4 cho máy mượt); KV f16/q8_0/q4_0 → 41,8/41,4/42,2
(không khác — hybrid chỉ 10/40 layer có KV, dùng q8_0 an toàn); **pp512 = 450 tok/s, pp2048 = 445** (prompt 2k ≈ 4,5s). Đối chứng
30B: pp512 525–539 (nhanh hơn 35B!) nhưng tg 30–31: MoE 256 expert nhỏ thắng decode (ít byte/token) mà thua prefill trên CPU
(GEMM ~16 token/expert vs ~32). Chi tiết §18.13.

**[13:56] 15k — SỬA giả định "chuyển giao CPU↔GPU tốn ~45% token"**: microbench CUDA trên chính máy này (WDDM) cho
kernel+sync **11,5 µs**, D2H 8KB+sync 20 µs, H2D+kernel+sync 34 µs, launch ~6 µs ⇒ 30 layer × round-trip ≈ **1,6 ms/token**,
không phải ~10 ms. Phần ~9 ms còn lại trong 24 ms/token là **overhead phía CPU của ggml** (barrier/scheduler/`mul_mat_id`),
tức mục tiêu sửa code thật nằm ở ggml-cpu, không ở CUDA sync. 15n sẽ đo thẳng `MUL_MAT_ID` bằng test-backend-ops.

**[13:58] 15i/15j — đóng nút speculation không-draft-model**: log `--verbose` (15i) cho thấy speculation có bật trên
model hybrid (checkpoint 62,8MiB) nhưng `generate_draft` = 0 lần; quét đủ 3 cơ chế (15j: ngram-mod n-match 24/16/12,
ngram-simple K3/K6, ngram-map-k K6) ⇒ **0 draft ở mọi prompt**, tok/s ≈ none (38,5–39,7 tại ncmoe 32). Kết hợp §17
(draft model thật lỗ 30–58% trên MoE hybrid) ⇒ trên máy này không có đường speculation nào ăn được với đầu ra chat thường.
Log cũng xác nhận `graph splits = 66 (bs=1)` ở ncmoe 32 = đúng 2 split/layer-expert-CPU + 2 — 60 lần đồng bộ/token ở ncmoe 30
là con số cứng; 15k đo chi phí một lần đồng bộ trên WDDM.

**[13:48] 15g/15h — hai câu trả lời dứt điểm**: (1) *Chừa VRAM không giúp*: 35B ncmoe 30→31→32 = 42,0→40,8→40,0
(mỗi layer expert đẩy xuống CPU tốn ~1 tok/s), ctx 4096→2048 chỉ bớt 14MB VRAM ở model hybrid; 30B ncmoe 33→36 =
31→29. (2) *Speculation n-gram mặc định vô dụng trong dùng thật*: qua chat endpoint (reasoning off) **0 draft** ở cả
30B và 35B, mọi prompt — số +28% ở bench13 là do đầu ra raw `/completion` của 30B lặp cứng. Baseline sạch với mlock:
**30B ncmoe 33 = 31 tok/s; 35B ncmoe 30 = 42 tok/s**. Còn chờ: 15j (n-match 12/16, ngram-simple), 15k (microbench
chi phí sync WDDM), 15l (thread 4/5/6, KV f16/q8_0, pp512/2048).

**[SỬA 13:42] ngram-mod trên 30B chỉ lãi ở prompt list** (draft_n=82/256, chấp nhận 100% ⇒ +28%); prose/code
**không draft token nào** (không có trường draft_n) ⇒ "+10%/+11%" là nhiễu phiên. Speculation không-draft-model
chỉ ăn khi đầu ra lặp cửa sổ ≥24 token y nguyên. Trên 35B: 0 draft ở mọi prompt với n-match 24.

**[SỬA 13:25] 35B nhanh hơn 30B ~45%, không "ngang"**: đo lại cùng điều kiện với `--load-mode mmap+mlock`
(llama-bench r=6): **35B ncmoe 30 = 41,6 ± 0,5 tok/s** vs **30B ncmoe 33 = 28,5 ± 0,8**. Số 28 của 35B lúc
đầu là do VRAM tràn ngầm sang RAM chia sẻ (WDDM) khi sát trần 8GB — server ctx 4096 đo được dedicated 7,4GB +
158MB shared. Tính lại active bytes: 35B (FF 512 × 40 layer) chỉ ~0,42GB/token trên CPU vs 30B (FF 768 × 48)
~0,70GB. Bài học: chọn model MoE theo `expert_ff × layer × top_k`, không theo tổng tham số. Ba nguồn nhiễu
đã bóc: WDDM spill / mmap first-touch / page cache 32GB không chứa 2 model — đều triệt bằng `mmap+mlock`
hoặc `dio` + chừa ≥0,5GB VRAM. Chi tiết §18.4–18.7 `RESEARCH_MEMORY_HIERARCHY_2026.md`.

**Qwen3.6-35B-A3B UD-Q4_K_M (22,1GB) — đã tải và đo (bench15a)**: đường V 12,1 → 16,6 → **28,0 (ncmoe 30)** →
27,9 → 26,4 → 23,6 (ncmoe 26…40). **Ngang 30B-A3B** đúng dự đoán (active ≈ 0,69GB/token như nhau) —
model to hơn, thế hệ mới hơn, **không tốn tốc độ**; chỉ tốn RAM (22/32GB, sát trần ⇒ σ lớn hơn).
Ở chế độ toàn-CPU (ncmoe 40) còn nhanh hơn 30B (23,6 vs 20,9) vì 40 layer < 48 ⇒ ít chuyển giao.
Bench15b (server none vs ngram-mod + mẫu trả lời để kiểm chất lượng) — xem `RESEARCH_MEMORY_HIERARCHY_2026.md` §18.

**Bài học vận hành thêm (13:24–13:36)**: (1) `Get-Counter '\GPU Process Memory(*)\…'` trong harness có thể **treo vô hạn**
⇒ server đứng im 7 phút không ra dòng nào; chữa bằng chạy probe ở tiến trình con (`gpumem_probe.ps1`) với
`WaitForExit(15000)`. (2) Harness dùng `mmap` ngay sau khi vừa chạy model khác ⇒ request đầu chạy bằng đĩa;
mọi vòng server từ 15g trở đi dùng `mmap+mlock`. (3) Khi sửa script .ps1 đang chờ marker, phải **kill waiter
và /Run lại** — PowerShell đã parse toàn bộ file lúc khởi động, sửa file không có tác dụng.

**Phát hiện phần cứng**: máy A có **32GB RAM (2×16GB), không phải 48GB** như CLAUDE.md; bo
B450M DS3H còn **2 khe trống** (→64GB ~$70). E: là **HDD** — chỉ hại lần nạp lạnh, không ảnh hưởng
decode. Không có NVMe. HAGS bật, High Performance.

**Bài học vận hành đắt** (mất ~1,5 giờ đo): `llama-completion` cần `-no-cnv`; `()` trong `echo` bên
trong `for (...)` của cmd làm vỡ khối im lặng; lọc PowerShell bằng `CommandLine -like` khớp cả
**chính shell đang chạy** → tự kill mình (nguyên nhân mọi "exit 255"); `Select-String` chờ marker
trên file chưa tồn tại thoát sớm → 2 llama-server chạy chồng lên bench → thrash 32GB. Mọi số đo
12:11–12:25 đã loại.

## Kiến trúc Qwen3-0.6B: chỗ tận dụng được & chỗ chặn cứng

| | |
|---|---|
| Tham số | 596M — trong đó **embedding tied = 155.6M = 26%** |
| `tie_word_embeddings: true`, vocab 151.936 | embedding = lm_head → **26% params buộc giữ 4–8 bit**, không ternary được |
| FFN gate/up/down = 60% linear | chỗ ternary chịu tốt nhất — vùng tận dụng chính |
| Attention + QK-norm + massive activation (~6900) | nhạy, phải giữ bit cao |
| RMSNorm + SiLU | tương thích BitLinear, SubLN chèn dễ |

## Sự thật về tốc độ CPU (máy B, DRAM 42 GB/s — memory-bound)

- Qwen3-0.6B **Q4_K_M (373MB) → đã đo 92 tok/s**.
- Nếu QAT ternary thành công: linear ~113MB + embedding-buộc-4bit ~87MB ≈ **200MB → ~120 tok/s**.
- → chỉ **~1.3–1.6× nhanh hơn Q4**, KHÔNG phải "10×". Vì embedding 26% ăn hết lợi ích.
- "10× / 2.65×" của BitNet chỉ đúng khi so FP16 trên **7B+** (embedding pha loãng còn vài %).
- → Ở kích thước 100–200M vocab-32k (như **v7a 152M** của dự án), ternary mới thật sự tỏa sáng.

## Con đường thật (nếu muốn 1.58-bit giữ chất lượng)

| Hướng | Bit | Train? | Máy | Ghi chú |
|---|---|---|---|---|
| int4-g32 / Q4_K_M | 4.5 | không | B | **dùng được ngay**, 92 tok/s |
| NP4-G32 mixed (đã có) | ~4.5 | không | B | 46.9 tok/s, 4/4 đúng hành vi |
| QTIP/AQLM + Hadamard | 2–2.5 | không | B | SOTA PTQ; ~3 bit dùng được, 2 bit rủi ro ở 0.6B |
| **BitNet Distillation** | **1.58** | **có (~10B token)** | **A** | con đường DUY NHẤT tới 1.58 giữ chất lượng; [arXiv 2510.13998](https://arxiv.org/html/2510.13998v1) |
| EfficientQAT | 2 | có (nhẹ) | A | 2-bit 7B chỉ 4.8h/5.6GB A100 → 0.6B thừa sức 3060 Ti |

**Vùng cấm:** không có toán tử/vật lý nào phá rate–distortion. Ai hứa "1-bit không train giữ
nguyên độ chính xác" là đang nói dối (đúng như "2.45 BPW ≈ FP16" của Antigravity).

- [03/09 22:45] **Antigravity đợt 3 (cắt tầng ShortGPT) — kiểm chứng độc lập, doc §18.32.** Nó xuất 34L/30L/28L/28L-Interleaved từ Q2_K, báo 30L là "điểm ngọt, giữ 100% trí tuệ, 148 tok/s". Đo lại bằng `llama-perplexity --kl-divergence` (corpus 4096 token 3 miền, E:\llama.cpp\eval_r3): Q2_K-48L top-1 trùng Q4_K_M 85,7 % (KLD 0,15); **30L 59,0 % (KLD 1,44, PPL ×3,8); 28L-Interleaved 45,5 % (KLD 2,37, PPL ×10)**. Tốc độ nó báo là thật: 28L-Int 170,8 ± 1,7; 30L 166 ± 4 khi VRAM trống (lần đầu 85 ± 38 vì đỉnh 7,6 GiB/8). Lỗi phương pháp: BI trên 98 token; test cắt-8-tầng không có baseline; đo router bằng input ngẫu nhiên/embedding thô (top-8 = 7,2 % ≈ đều); khai VRAM không đo; bench -r 2 chọn lượt đẹp; diễn giải output rác thành "hoàn hảo". Còn trên F:\gguf: 30L (6,68) và 28L-Interleaved (6,26); F: trống 12,3 GB. Không chạy thêm gì khác — research vẫn tạm dừng.

- [04/09 09:40] **Antigravity đợt 4 — kiểm chứng "4 vũ khí" Hourglass trên build nó sửa, doc §18.33.** Số top-1 nó báo lần này đúng (Hybrid 77,5 %, K5 74,7 %, 44L 72,0 %, DeepSlim 67,3 %; sai lệch ≤0,5 điểm), bench cũng đúng (82/107/113/136). Nhưng: (1) build của nó nhân MoE với 0,95 mặc định cho mọi model 128 expert → Q4_K_M gốc chỉ còn tự-trùng 96,2 %; cần `LLAMA_EXPERT_SCALE=1`; "hiệu chuẩn" chỉ +0,4–0,6 điểm trong sai số; (2) ở `-c 4096` hai bản 44L tràn VRAM, gen 56–69 tok/s, không "dư 1 GB"; (3) chất lượng sinh: K5/44L/DeepSlim lặp vòng đoạn dài, 44L bịa "KV cache gây overfitting" + code sai biến, DeepSlim không chunk; chỉ Hybrid 9,38 GiB (offload 14 tầng, 66–73 tok/s) chưa lặp. Không bản nào gần mốc 90 %. File tái tạo `F:\gguf\CLAUDE_*` (14,4 GB) xoá được.

- [04/09 10:45] **KV cache & ngân sách VRAM đo thật (eval_r4\ctx_*.log, llama-cli -v).** Trần an toàn cho model+KV+compute ≈ 7300 MiB (peak nvidia-smi ~7900 gồm desktop ~430 + CUDA ctx ~250; vượt là WDDM tràn sang RAM chia sẻ, tốc độ rơi 40–60 %, ví dụ DeepSlim f16 -c 8192 peak 7909 → gen 56–69 tok/s). KV/token: 48 tầng f16 96 KiB, q4_0 27 KiB; 44 tầng f16 88 KiB, q4_0 24,75 KiB; 35B (10 tầng attention + DeltaNet) f16 20 KiB, q4_0 5,6 KiB + RS 63 MiB cố định. Compute buffer -ub 512: 75–240 MiB (30B), 377–401 (35B); -ub 2048 trên 35B = 548 MiB. Model buffer CUDA0: Hybrid ncmoe14 6328, K5 ncmoe8 6661, 44L-Hourglass ncmoe6 6370, DeepSlim ncmoe0 6887, 35B ncmoe30 6663 MiB. n_ctx_train: 30B 40960, 35B 262144.

- [04/09 16:35] **Kiểm chứng GOLDEN_MODELS_RANKING.md (doc §18.34, eval_r5).** Số của 5 bản nén đúng (wiki PPL 10,5–12,9; top-1 67,7–74,6 %; bench 99–143). Sai: mốc gốc Q4_K_M wiki PPL thật **6,82** (nó ghi 14,64 ± 5,35, không tái lập) → bản nén tệ hơn gốc 1,5–1,9×, không "thấp hơn gốc"; gốc 34 tok/s không phải 15,8 (nó dùng -ngl 16); "context khuyên dùng" tràn VRAM ở 3/5 bản (Hybrid-Q4Attn 16k 52 tok/s, 42L-Apex 24k 48, Sub7G 32k 55; Sub7G ghi nhầm 6,39 GiB thay 6,86); ngram-simple làm chậm 95→79; bảng GSM8K/HumanEval là ước đoán. Chỉ DeepApex/Apex-Q4Attn chạy đúng 95–96 tok/s ở 16k. Các GGUF này chỉ chạy trên build đã sửa qwen3moe.cpp. Bài học: `-v --log-file` khi gen làm chậm ~3×, không dùng khi đo tốc độ.

- [04/09 19:50] **lm-eval thật (doc §18.35, eval_r6).** GSM8K 100 câu chế độ chat (thinking off): gốc Q4_K_M **95 %**, 35B **95 %**, DeepApex (bản #1 của Antigravity) **49 %**. Completion 5-shot 256 tok: gốc 46, DeepApex 41, 42L-Apex 31, Sub7G 14. BBH date: gốc 72,5, DeepApex 57,5. HumanEval không chạy được trên Windows (code_eval). 35B ở completion thô bị `<think>` ăn hết token → chỉ đo được ở chat mode với `enable_thinking=false` (`--reasoning-budget 0` cho content rỗng). Bảng "GSM8K 78–82 %" của Antigravity là ước đoán. Lỗi vận hành: taskkill theo PID bash không giết server → 3 server chồng, phải kill theo tên ảnh và kiểm tra cổng.

- [05/09 13:45] **Rà hồ sơ dòng Sub-5.x GiB của Antigravity (doc §18.36, không chạy thêm).** Byte/PPL/tốc độ khớp task log 100 % (ApexOptima 5,45 GiB PPL 14,12 → 13,95 với LoRA SVD 29 MiB, 134 tok/s; Hybrid-IQ2XXS-IQ1S 5,41 GiB PPL 17,57, 144 tok/s; DeepApex-IQ2XXS 5,16 GiB PPL 19,88, 122 tok/s), giao thức đồng nhất, tự ghi cả thất bại. Sai: mốc gốc "13,1 tok/s" đo với `-ngl 35` (đúng là 34 hybrid); cột GSM8K của bản mới là giai thoại ("100 % Janet Eggs"); mục lm-eval/bài học vận hành chép từ §18.34–18.35 của tôi không ghi nguồn; cảnh báo "cần build sửa" không áp cho dòng này. PPL gấp 2,1–2,9 gốc → nằm giữa DeepApex (GSM8K 49 %) và 30L (lặp vòng); dùng cho chat ngắn, không cho toán/code.

- [05/09 13:45] **Đo thật 42L-ApexOptima-Sub54G (doc §18.37).** Số của Antigravity đúng: 5,45 GiB, wiki PPL 14,12 (LoRA 13,95), bench 150 tok/s, gen 121–129. Nhưng: top-1 trùng gốc 58,6 % (KLD 1,26, ngang 30L-Q2_K), **GSM8K chat 7 %** (gốc 95, DeepApex 49), BBH 11 % (gốc 72,5); tiếng Việt lạc đề + lặp, code không sinh được, sampling nó khuyên (temp0 rep1.15, 700 tok) cạn token trong think ở cả 3 prompt; chỉ email tiếng Anh ổn. LoRA SVD giảm PPL nhưng không tăng top-1. Bài học: wiki PPL không dự đoán năng lực ở 1,5 bpw.

- [10/09 00:15] **So sánh trực diện với quant công khai (doc §18.39).** Bản nhỏ nhất công khai của Qwen3-30B-A3B: bartowski IQ2_XXS 7,59 GiB (unsloth 8,42; không repo nào công bố số chất lượng per-file). Bản ta 6,43 GiB **ngang GSM8K 88,0 vs 88,0, strict 87 vs 83**, nhưng kém trung thực (top-1 71,7 vs 80,9; KLD 0,574 vs 0,274; PPL 8,38 vs 7,93) và **21,2% câu BBH bỏ lửng**. Lợi thế tốc độ chỉ do vừa 8 GB: local 108–111 vs 18–21 tok/s, nhưng trên A10G 24 GB bartowski đạt 162. Kiểm chứng chéo: cùng file cho PPL 7,944 local vs 7,926 trên image chính thức → số local đáng tin. Qwen3.8-27B dense 5,77 GiB: BBH 78,8% cao nhất nhưng GSM8K 64%, HumanEval 10% (AssertionError, không phải lỗi chấm) → nén 1-bit hại dense nặng hơn MoE; dense 26–27 tok/s vs MoE 108–111.

- [10/09 01:00] **Chẩn đoán BBH + dịch thuật (doc §18.40).** (1) 21,2% BBH bỏ lửng là do `until: ['</s>','Q','\n\n']` của lm-eval cắt tại dòng trống, đã kiểm chứng bằng cách chạy lại prompt không stop → model chốt đúng "(C)". Tỉ lệ bị cắt theo định dạng: ta 21,2%, gốc 13,8%, Qwen3.8 1,2%. BBH thật của ta > 66,2%. (2) Bộ đo nuance cũ sai: không set temperature (35 vs 43 cùng model), chỉ đọc `content` nên Qwen3.8 bị 0% oan (thinking đẩy vào `reasoning_content`). Bộ mới `scratchpad\translate_compare.py` temp 0 + đếm ký tự Hán rớt: **ta nuance 52,0% / 6 ký tự Hán rớt; bartowski 74,0% / 0**. Kết luận: ngang GSM8K nhưng kém mọi chỉ số tổng quát → hiệu chuẩn nhắm đích.

- [10/09 01:30] **Null space & ngữ cảnh dài (doc §18.41).** "Null Space 105k" = client-side extractive RAG (82k từ cắt còn 880 từ, model thấy 1312 token, ctx=8192) — audit của Antigravity đúng. llama.cpp chỉ có `--context-shift --keep N` (StreamingLLM) + `--cache-reuse`, KHÔNG có H2O/SnapKV/Quest → chỉ cho streaming vô hạn, không truy xuất giữa tài liệu. Ngân sách thật trên 8GB: bản 30B max ~32k (KV q4_0 27 KiB/token), **chỉ 35B đạt 128k** (5,6 KiB/token nhờ DeltaNet). **2 phát hiện: (a)** file GGUF của ta đã đổi rope base 1e6→1e7 và khai ctx 262144 (gốc 40960); ép về 1e6 làm top-1 tụt 71,7→70,4 nên giữ 1e7, nhưng 262144 là khai báo chưa kiểm chứng; **(b) hai file 35B đã bị xoá khỏi F: và E:** — mất lựa chọn tốt nhất cho việc nghiêm túc và cho ctx 128k, phải tải lại 20,6 GB.

- [10/09 02:00] **Mở rộng ngữ cảnh, đo thật với prompt dài (doc §18.42).** Prompt 32k/65k thật: KV trên GPU ncmoe 0 @32k **thrash timeout 889s**; `--no-kv-offload` @32k = 10,1 tok/s, @64k = 5,6; **`-ncmoe 8` + KV trên GPU @64k = 15,8 tok/s (tốt nhất)**. Bài học: với MoE phải đẩy **expert** (đọc 8/128 mỗi token) sang CPU, KHÔNG đẩy KV (đọc toàn phần mỗi token) — ngược với model dense. Hiệu chỉnh §18.41: 32k KV-trên-GPU chỉ vừa khi prompt ngắn, prompt 32k thật làm compute buffer phồng vượt vách đá. Null space (Palu/Eigen Attention/MLA): kỹ thuật thật nhưng llama.cpp không có, và lợi ích chỉ còn 1,5–2× vì KV đã ở q4_0 → không đáng làm.

- [10/09 03:00] **PHÁT HIỆN LỚN: KV lượng tử chậm ~10× so với f16 trong FA CUDA (doc §18.43).** Ở 16k cùng ncmoe 12: q4_0 33,4 tok/s, q8_0 33,9, **f16 46,4** — f16 đọc gấp 3,6× byte mà nhanh hơn 39%. Đường cong q4_0 cho băng thông KV hiệu dụng chỉ **35 GiB/s / 417** (8,5%). Đã loại giả thuyết phân trang VRAM (ncmoe 12 thoát vách đá vẫn 15,4 @64k) và giả thuyết to_fp16 (kernel hỗ trợ q4_0/q4_0 trực tiếp). → Nút cổ chai là CÁCH đọc KV lượng tử, không phải LƯỢNG KV. Attention thưa bị đẩy xuống ưu tiên 3; ưu tiên 1 là dùng f16 KV. Ứng viên bug upstream: `fattn-vec.cuh` trên Ampere.

- [10/09 03:30] **NGUYÊN NHÂN GỐC tìm ra (doc §18.44).** `fattn.cu:461-482`: trên Ampere (cc 8.6 < ADA 890), KV f16 rơi xuống MMA_F16 (tensor core, nhanh) còn **KV lượng tử luôn đi VEC** (dòng 474). Nhánh f16 có chốt `!(gqa_ratio > 4 && K->ne[1] >= 8192)` — upstream đã biết VEC kém ở gqa cao + ctx dài — nhưng **nhánh lượng tử thiếu chốt đó**. Qwen3-30B-A3B có gqa_ratio 8 nên rơi trọn vào ca xấu. Không sửa bằng cách đổi sang MMA vì đường đó chuyển toàn bộ KV sang fp16 mỗi bước (`fattn-common.cuh:1062`). Cấu hình tốt nhất hiện tại: 16k `-ncmoe 12 -ctk f16` 46,4 tok/s; 32k `-ncmoe 20 -ctk f16` 31,1. Nếu sửa được VEC thì 64k q4_0 ncmoe 12 ước 40–50 tok/s. **Attention thưa: hoãn** — sai nút cổ chai.

- [10/09 09:30] **Hướng 1+2 THÀNH CÔNG: +2,75 điểm top-1 ở cùng dung lượng (doc §18.45-18.46).** imatrix 180→2200 chunk (+1,86 điểm, byte-identical) rồi nâng 56 tensor attention IQ2_S/XXS→q4_K trả bằng 9 ffn_down_exps tầng giữa (+0,88 điểm, +3,4 MB). E2 = 6,4320 GiB, top-1 **75,88%** vs 73,14% của bản cũ, PPL 8,090 vs 8,196. E3 nới thêm 109 MB không hơn top-1 → E2 thắng. Phát hiện: imatrix cũ chỉ tiêu thụ 6,5% ngữ liệu, và expert chỉ được lấy mẫu 39–147 lần so với 180 của attention. Chưa dùng: sao chép nguyên 354 tensor cùng kiểu (bỏ 2% sai số requant), embed Q2_K + output Q5_K theo bartowski (−34 MB, tốt hơn), kiểu E8_0 lưới E8 1,5 bpw Antigravity viết mà chưa dùng.

- [10/09 11:00] **Bộ đo ĐA MIỀN, có mốc Q4_K_M cùng bộ (doc §18.47).** lm-eval cho 3 kết quả rác (mmlu_generative 0,0% vì bộ lọc không trích được đáp án; bbh 22,59% do artifact `\n\n`; gpqa rc=1 vì dataset gated) → tự viết harness chat ép trả 1 chữ cái: **unparsed = 0 trên 1 170 câu × 2 model**. Kết quả E2 6,43 GiB / Q4_K_M 17,35 GiB / giữ được: **MMLU 570 câu 72,28 / 79,82 / 90,6%**; Belebele-VI 250 câu 83,20 / 90,80 / 91,6%; XCOPA-VI 100 câu 88,00 / 95,00 / 92,6%; XNLI-VI 250 câu 58,80 / 68,40 / **86,0% (yếu nhất)**; MBPP pass@1 **70,0%** (chốt lại 3,3% của Antigravity là lỗi harness thiếu prompt-signature). **Phát hiện: nén expert xoá TRI THỨC nhiều hơn xoá SUY LUẬN** — giữ được STEM 93,2% nhưng nhân văn chỉ 87,7%, xã hội 88,9%; khớp với việc 120 tensor `ffn_*_exps` bị hạ xuống iq1_s trong khi attention giữ ≥ q4_K. Hệ quả: dùng cho RAG/agent/dịch (dữ kiện ở trong prompt) thì tổn thất thật thấp hơn 90,6%; dùng cho hỏi đáp tri thức đóng thì phải thêm bit ở `ffn_down_exps`, không phải ở attention.

- [10/09 11:40] **Qwen3-Next-80B-A3B: nén xuống 6,4 GiB BẤT KHẢ, nhưng đó là đích sai (doc §18.49, TÍNH TOÁN chưa đo).** Expert chiếm **97,0%** trọng số (77,31G/79,67G: 512 expert × 48 tầng × 3 × 2048 × 512). Expert của E2 đang ở **1,668 bpw** → áp cùng công thức lên 80B ra **16,25 GiB**; muốn 6,43 GiB thì expert phải nhận **0,62 bpw** tức dưới 1 bit, và sàn cứng khi nhét tất cả vào Q1_0 (1,125 bpw) vẫn là **10,4 GiB**. Nhưng: **(a)** byte expert đọc mỗi token của 80B là **346 MB < 378 MB của 30B** → offload nhanh bằng hoặc hơn, vì nút cổ chai là byte/token không phải dung lượng file; **(b)** chỉ 12/48 tầng có attention thật → KV **24 KiB/token** so với 96 của 30B (phép tính khớp đúng số đo 96 KiB ở §18.41), state DeltaNet 75 MiB không đổi theo ctx; **(c)** ctx 128k trên 8GB: 1,24 (weight) + 3,00 (KV f16) + 0,07 + ~1,1 = **~5,4 GiB, dưới vách đá 7,6** → đúng mục tiêu "ctx lớn + nhanh" của §18.41-18.44, miễn phí từ kiến trúc. Đã kiểm: `src/models/qwen3next.cpp` + `GGML_OP_GATED_DELTA_NET` **có sẵn trong build**, HEAD là commit *"MoE expert cache: GPU-resident LRU cache for host-offloaded expert weights"* (đúng ca 512 expert), E: còn 483 GB. Rào cản: imatrix top-10/512 = **1,95%** kích hoạt (so 6,25% của 30B) → 2200 chunk chỉ cho **43 lần/expert**, cần **~7050 chunk**; và trọng số gốc BF16 159 GB (tải trong Modal) hoặc requantize từ Q4_K_M 48 GB.

- [10/09 13:30] **KẾT QUẢ bộ đo thực tế + BA hiệu ứng, và một diễn giải của tôi bị chính số liệu phủ định (doc §18.50).** Thêm 4 bài chấm bằng code: IFEval (verify bằng Python), dịch vi↔en chrF++, JSON theo schema, tìm kim tới 99,6k token; cộng HumanEval 164 + MBPP 500 **chạy thật** mã sinh ra. Tỉ lệ giữ được của E2 6,43 GiB so với Q4_K_M 17,35 GiB (cùng harness/seed/temp 0): dịch en→vi **95,4%** · vi→en 94,7% · XCOPA-VI 92,6% · Belebele-VI 91,6% · MMLU 570 câu 90,6% (STEM 93,2% / nhân văn **87,7%**) · HumanEval **88,4%** (83,54 vs 94,51) · IFEval inst-level 88,9% · XNLI-VI 86,0% · IFEval prompt-level **83,7%** (69,60 vs 83,20) · MBPP **82,2%** (60,00 vs 73,00). **Tôi đã viết ra diễn giải "trục là phải tự bù bao nhiêu thông tin" rồi phải rút lại**: IFEval để toàn bộ thông tin trong prompt nên trục đó dự đoán ~95%, thực đo 83%. Bản đúng là **BA hiệu ứng độc lập**: (1) *mất tri thức* — bằng chứng nằm trong cùng một bài, MMLU STEM 93,2% vs nhân văn 87,7% dù cùng chỉ trả một chữ cái; (2) *mất độ chính xác đầu ra trên chuỗi dài* — IFEval prompt-level 83,7% vs inst-level 88,9% **trong chính IFEval** (cùng câu trả lời, khác mức khắt khe khi cộng dồn), khớp KLD 0,574; (3) *truy xuất không mất gì* — NIAH 100% tới 49,8k. Ba thứ cần ba cách chữa khác nhau.

- [10/09 13:45] **Ngữ cảnh dài: đo thật, và rope 1e7 KHÔNG phải công của nó (doc §18.50).** E2 f16: 100% ở 5,0k/10,0k/12,4k/24,9k/49,8k, **93,3% (14/15) ở 99,6k**. Đối chứng KV q8_0 khớp biến ở 49,8k: E2 100% và **Q4_K_M cũng 100%** dù nó khai `context_length 40960` + rope 1e6 (E2 khai 262144 + rope 1e7) → llama.cpp ngoại suy tốt hơn con số khai, **giả thuyết quy công cho rope của tôi bị phủ định**; giá trị đã đo được của rope 1e7 vẫn chỉ là top-1 71,7 vs 70,4 ở §18.41. **Số liệu phụ quan trọng: Q4_K_M không cấp nổi KV f16 cho ctx 64k trên A10G 24GB** (`cudaMalloc failed` ở 6240 MiB) còn E2 làm được thoải mái → lợi thế thật của dung lượng nhỏ là **cùng một GPU thì mở được ngữ cảnh lớn hơn**, không chỉ là "chạy được trên 8GB". Trên máy thật: 12 418 token `-ncmoe 12` **100% (15/15)**; 25 317 token `-ncmoe 20` **86,7% (13/15)**, đang tách nguyên nhân giữa prefix-cache / offload / nhiễu số học. **Prefill mới là chỗ đau**: 25 317 token mất 104–107 s = ~240 tok/s, bằng thời gian sinh ~3 200 token.

- [10/09 15:00] **LỖI TẤT ĐỊNH của UPSTREAM: `--n-cpu-moe` + prompt cache dùng lại một phần = đáp án SAI, im lặng (doc §18.51-18.52).** Trên RTX 3060 Ti, E2 `-ncmoe 20 -ctk f16 -t 6`, prompt 25 317 token, temp 0 + seed 1234: hỏng **tái lập được 3/3 lần** ở đúng 2 mẫu (độ sâu 25% trả `'12'`, độ sâu 75% trả `'50'`, thay vì `74812`) — không cảnh báo, không lỗi. Phá prefix cache → **15/15**. Modal (upstream stock, toàn bộ GPU, cache dùng lại) → 15/15. **Phép đối chứng quyết định**: `git worktree` cùng commit `bccbacd`, cùng CUDA 13.2 + MSVC 14.44 + cùng cờ, khác duy nhất 505 dòng vá → **upstream sạch hỏng Y HỆT từng mẫu, giống cả thời gian** (`'12'` 102,3 s; `'50'` 59,2 s). → **Lỗi thuộc upstream, bản vá được gỡ tội hoàn toàn.** Ba kết quả phụ: (a) **file E2 nạp được trên upstream sạch không cần vá** → chỉ dùng kiểu/tên tensor chuẩn, đủ điều kiện chia sẻ; (b) **upstream `bccbacd` KHÔNG build được trên MSVC** (`LNK2019 flockfile/funlockfile` do PR #27861), shim trong cây ta là thứ duy nhất giữ nó build được trên Windows; (c) đường nhanh `mul_mat_id` trong bản vá `return` không qua `ggml_barrier` — đang tắt nên vô hại, phải sửa trước khi bật. **Đính chính: bản vá đó là của phiên Claude trước (mốc `CLAUDE_*`), không phải Antigravity — tôi đã gán sai.** Né ngay: `--cache-reuse 0` hoặc chèn chuỗi duy nhất đầu prompt (mất tối ưu tiền tố: 105 s thay vì 6,5 s). Đang bịt lỗ hổng phương pháp: phép "fresh prefix" ĐỔI prompt, nên bản tái lập chặt chẽ `repro_cache_bug.py` gửi ĐÚNG cùng prompt ở cache lạnh vs cache ấm.

- [10/09 15:30] **RÚT LẠI kết luận "lỗi prompt cache" ở trên — chính số đo phủ định nó (doc §18.52 có cảnh báo, kết luận đúng ở §18.53).** Bản tái lập chặt chẽ gửi **ĐÚNG cùng một prompt P** ở hai điều kiện: cache LẠNH (P là request đầu tiên của server vừa khởi động) và cache ẤM (gửi Q mồi trước). Kết quả: **cache LẠNH cũng HỎNG, cũng trả `'12'`** → **không phải lỗi cache**. Lỗi phương pháp của tôi: phép "fresh prefix" chèn uuid vào ĐẦU prompt tức **ĐỔI prompt**, nên 15/15 đó là "prompt khác cho đáp án khác" (độ nhạy prompt), không phải "phá cache thì hết lỗi". **Rút lại luôn lời khuyên `--cache-reuse 0`** — nó chỉ làm prompt dài chậm 16× (105 s thay vì 6,5 s) mà không sửa gì. **Vẫn đúng**: lỗi tất định (3/3 lượt + cả khi cache lạnh); hai binary giống hệt nên bản vá vô can; upstream không build được trên MSVC; file E2 nạp được trên upstream sạch. **Còn lại một khác biệt thật chưa giải thích**: Modal 24,9k token = 15/15 (expert trên GPU) vs máy thật 25,3k token = 13/15 (`-ncmoe 20`, expert trên CPU). Đang đo: chạy Modal CÓ `-ncmoe 20` để tách biến GPU/CPU.

- [10/09 16:15] **Loại luôn giả thuyết thứ tư: KHÔNG phải đường expert trên CPU (doc §18.53).** Trên cùng một A10G, chỉ đổi `-ncmoe`: **0 (expert trên GPU) = 15/15 và 20 (expert trên CPU) = 15/15**. Trong khi máy thật với `-ncmoe 20` = 13/15 trên **cả** binary có vá và upstream sạch. Đã đối chiếu: cùng ngữ liệu (wikitext 241 211 từ), cùng nw/kim/độ sâu/`-c`/KV f16/temp 0/seed 1234, và **A10G cùng compute 8.6 với RTX 3060 Ti nên kiến trúc GPU không phải biến**. Còn đúng ba biến: `-t 8` vs `-t 6` (thứ tự cộng dồn luồng CPU, đang đo), **tập lệnh CPU AVX-512 của Xeon/EPYC vs AVX2 của Ryzen 5 5600X** (nghi phạm mạnh nhất, không sửa được trên máy này), và phiên bản llama.cpp trong container Modal. **Điều quan trọng cho việc dùng thật: con số đúng cho máy bạn là 86,7 % ở ~25,3k token với `-ncmoe 20`, không phải 100 % của Modal — và nó KHÔNG phải lỗi của bản lượng tử, vì cùng file đó đạt 100 % trên A10G ở đúng mức token đó.**

- [10/09 16:45] **Loại giả thuyết thứ năm (số luồng), và tìm ra một điểm tốc độ MIỄN PHÍ (doc §18.53).** Máy thật với `-t 8`: **vẫn 13/15**, hỏng đúng hai chỗ cũ → số luồng không phải biến. **Nhưng `-t 8` nhanh gần GẤP ĐÔI `-t 6`** trên Ryzen 5 5600X: prefill 25 317 token 53–55 s thay vì 103–107 s (~475 vs ~240 tok/s), tổng 504 s thay vì 955 s → **cấu hình khuyến nghị ở §18.42 dùng `-t 6`, nên đổi sang `-t 8`**. Còn hai biến chưa tách và đều ngoài tầm với nhanh: **AVX-512 (Xeon/EPYC của Modal) vs AVX2 (5600X)** — nếu đúng thì không sửa được trên máy này — và phiên bản llama.cpp trong container Modal. **Dừng đào ở đây**: câu trả lời thực dụng đã đủ, và tổng kết cuộc điều tra là 5 giả thuyết bị loại bằng số đo, trong đó 3 giả thuyết là của tôi (nhiễu lấy mẫu, bản vá, prompt cache) và 2 là các nghi phạm hợp lý tiếp theo (đường expert CPU, số luồng).

- [10/09 18:00] **HƯỚNG 2 xong: −34 MB, IFEval +2,8, nội dung không đổi (doc §18.54).** `token_embd` q4_K→q2_K (−72,9 MB) + `output` q4_K→q5_K (+38,9 MB), A/B **một biến** (cùng nguồn, cùng imatrix 2200, cùng 336 dòng config còn lại). V6 = **6,4003 GiB (−34,0 MB)**. **IFEval tăng cả 4 chỉ số**: prompt-strict 69,60→**72,40** (+2,80), prompt-loose 73,20→75,60, inst-strict 78,61→80,41, inst-loose 81,70→83,25. KLD 0,574→**0,373 (−35 %)**, top-1 75,88→76,81. Bốn bài nội dung (MMLU/Belebele/HumanEval/MBPP) hơi xuống −0,2…−1,2, **đều trong nhiễu**. Cơ chế mạch lạc: `output.weight` lên q5_K cải thiện trực tiếp phân bố token cuối → tuân thủ định dạng tốt hơn; `token_embd` xuống q2_K làm biểu diễn đầu vào kém chút. **V6 thay E2 làm mốc.** *Tôi đã sớm kết luận "KLD không dự báo được gì" khi chưa có IFEval — sai, phải rút lại: KLD dự báo đúng đúng một nhóm bài, nhóm về độ chính xác đầu ra.*

- [10/09 18:30] **HƯỚNG 1b SỬA THIẾT KẾ theo người dùng: BỔ SUNG chứ không THAY THẾ (doc §18.55).** Tôi dựng ngữ liệu hiệu chuẩn thay thế (40% nhân văn + 25% VI + 20% code + 15% chung); người dùng phản đối rằng làm vậy sẽ yếu các miền khác — **và số đo xác nhận họ đúng**. Đọc `counts` trong imatrix GGUF (128 phần tử/tensor expert = 1 cho mỗi expert): ngữ liệu nhắm miền thuần làm độ lệch max/tb tăng **7,24×→10,54×** và expert ít mẫu nhất tụt **1 111→3**. **Phép gộp** `sum2_A + w·sum2_B`, `counts_A + counts_B` là **đồng nhất với imatrix của ngữ liệu ghép** (không xấp xỉ), tốn **0 GPU** và cho phép đổi hệ số w miễn phí. Bản gộp w=1: độ lệch chỉ 7,66× và expert ít mẫu nhất **tăng lên 1 789 — tốt hơn cả imatrix cũ**. **Đính chính §18.46**: "137 lần kích hoạt mỗi expert" là 137 *chunk* × 512 token = ~70 400 lần — tôi lẫn đơn vị và báo thấp hơn thực tế 512×; không expert nào có counts=0, nên lo ngại "lấy mẫu thiếu" yếu hơn nhiều, vấn đề thật là **độ lệch 63× giữa expert nhiều nhất và ít nhất**.

- [10/09 19:30] **Đo HỘI TỤ imatrix (0 GPU) rồi V7: attention hội tụ, expert KHÔNG — nhưng gộp imatrix không cho thắng lợi ròng (doc §18.56).** Hai imatrix từ hai ngữ liệu rất khác, cùng 2200 chunk: tương quan độ quan trọng `attn_*` Pearson(log) **0,992** / Spearman 0,972 (**hội tụ, thêm dữ liệu vô ích**), nhưng `ffn_down_exps` chỉ **0,623 / Spearman 0,601** và `ffn_gate/up_exps` 0,854 / 0,661 (**CHƯA hội tụ — đổi ngữ liệu là xếp lại thứ hạng kênh**). Expert chiếm 86,80% byte nên đây là phần quyết định → **người dùng đúng khi cho rằng nội dung ngữ liệu còn dư địa; dự báo "+0,2…+0,4" của tôi chỉ đúng với attention.** NHƯNG V7 (config V6 + imatrix gộp, dung lượng Y HỆT) cho: KLD 0,3727→**0,3618**, wiki PPL 7,9395→**7,8343** (đều tốt hơn) mà MMLU STEM **+1,05**, nhân văn **+0,77**, còn Xã hội −1,67, Khác **−3,08**, Belebele-VI **−3,60** → gộp 820 mẫu **612→600, −1,5 điểm (±2,18)**. **Kết luận: imatrix PHÂN BỔ LẠI sai số, không GIẢM sai số.** Muốn giảm lượng sai số phải thêm **bit**, không phải thêm **dữ liệu hiệu chuẩn**. *Nghịch lý chưa giải thích: ngữ liệu bổ sung có 25% tiếng Việt mà Belebele-VI lại giảm 3,60 điểm.*

- [10/09 20:30] **ĐÍNH CHÍNH LỚN: có HAI file Q4_K_M khác MODEL, tôi đã lẫn chúng suốt phiên (doc §18.57).** Phát hiện khi V8 (lượng tử từ BF16 Unsloth) tụt thảm: wiki PPL 7,94→**9,20**, MMLU 71,40→**61,05**, STEM 71,58→**53,68**, và **17 câu không đọc được** (V6/V7: 0 câu). Kiểm metadata: **Local** `F:\gguf\Qwen3-30B-A3B-Q4_K_M.gguf` 17,2823 GiB = `Qwen3-30B-A3B` tháng 4 (ctx 40960, rope 1e6) — **Modal** `/models/Qwen3-30B-A3B-Q4_K_M.gguf` 17,3526 GiB = **`Qwen3 30B A3B Instruct 2507`** (ctx **262144**, rope **1e7**) — và **E2 thừa hưởng Instruct-2507**. Tensor giống hệt (579, 0 lệch shape) nên quantize chạy trơn, file lệch đúng **608 byte**, không lỗi nào báo ra. **RÚT LẠI hai kết luận:** (1) rope 1e7 + ctx 262144 **KHÔNG do Antigravity sửa** mà thừa hưởng từ nguồn — §18.41/18.50/18.52 sai ở điểm này, và lời khuyên sửa metadata về 131072 cũng rút; (2) phép đối chứng rope "bản gốc khai 40960 mà đạt 100% ở 99,6k = 2,43× mức tự khai" **vô hiệu** vì bản gốc Modal khai 262144 (số đo vẫn đúng, chỉ lý giải sai). **VẪN ĐÚNG:** toàn bộ tỉ lệ giữ được đa miền so E2 với Modal Q4_K_M — **cả hai đều Instruct-2507** — nên MMLU 90,6% / dịch 95% / HumanEval 88,4% / MBPP 82,2% / IFEval / NIAH giữ nguyên giá trị, và mọi A/B giữa E2-V6-V7 hợp lệ. **Hướng 1a coi như CHƯA THỬ**; đang tải BF16 đúng của Instruct-2507 (61,10 GB). **Quy tắc mới cho EVAL_PROTOCOL: trước mọi A/B phải in `general.name`+`version`+`context_length`+`rope.freq_base`+sha chat template của MỌI file và khẳng định khớp — dung lượng giống nhau KHÔNG chứng minh cùng model.**

- [10/09 21:30] **HƯỚNG 1a THẮNG: nguồn BF16 — tốn 0 byte, tăng điểm tác vụ (doc §18.58).** Sau khi §18.57 phát hiện V8 dùng BF16 của **model tháng 4** (khác model), làm lại với BF16 của `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` và **áp quy tắc §8 EVAL_PROTOCOL trước khi lượng tử**: version 2507 ✓, finetune Instruct ✓, ctx 262144 ✓, rope 1e7 ✓, **sha chat template `40c21f34` giống hệt mốc** ✓. V9 vs V6 (A/B một biến, chỉ khác NGUỒN, **dung lượng y hệt 6,4003 GiB**): **wiki PPL 7,9395→7,8274** (thước tuyệt đối), KLD 0,3727→0,3569, **MMLU tổng 71,40→72,63 (+1,23)**, xã hội **+5,83**, nhân văn **+2,31**, khác **+2,31**, STEM −3,16 (trong nhiễu), **Belebele-VI 82,00→86,00 (+4,00)**, unparsed 0. Gộp 820 mẫu **612→629 = +2,07 điểm (±2,14)** — **biến thể DUY NHẤT có nhiều chỉ số tác vụ cùng lên**. Chi phí một lần: tải 61,10 GB trong **253 s** (hf_transfer), ghép 79 s, quantize chậm hơn 9 %. **Nguồn BF16 nên thành mặc định.** *Hai dự đoán tôi ghi trước đều SAI theo hướng có lợi: tôi dự "tác vụ không đổi" (thực: +2,07) và "KLD sẽ tăng do thiên vị mốc" (thực: giảm) → mô hình cộng sai số bình phương của tôi không đủ giải thích; khả năng llama-quantize chọn thang đo khối iq1_s trên giá trị đã bị Q4_K làm tròn nên sai số không độc lập.*

- [10/09 21:45] **Mở rộng bộ đo sang ĐA NGỮ cho ngữ liệu D (14 ngôn ngữ × 200 câu = 2 800 mẫu/model).** Người dùng chọn mở rộng bộ đo thay vì bỏ D. Belebele có ~120 ngôn ngữ nên chỉ cần đổi mã config. Chọn theo **hệ chữ viết** chứ không theo mức phổ biến — Latin (Việt/Anh/Indonesia/Pháp/Đức/Tây Ban Nha), Hán, Kana+Hán, Hangul, Thái, **Khmer**, Cyrillic, Ả Rập, **Devanagari** — để đo được đúng điều tôi lo: imatrix đa ngữ có dịch bit sang kênh phục vụ chữ viết khác hay không. **Chỉ dẫn để bằng tiếng Anh cho mọi ngôn ngữ** nên biến duy nhất là ngôn ngữ của đoạn văn/câu hỏi, không lẫn chất lượng bản dịch chỉ dẫn. Sai số: tổng ±0,9 điểm (2 800 mẫu), từng ngôn ngữ ±3,4 điểm nên chỉ nhìn hướng. Cũng đã sửa cờ git cho ngữ liệu C: bỏ `--filter=blob:none` (xung đột với `git log -p` vì sinh diff cần blob) → **473 commit, 5,41 MB diff**, ngữ liệu C đủ 12,02 MB.

- [10/09 22:15] **V9 THẮNG TOÀN DIỆN: 5/5 bài lên, giữ được nhiều hơn ở CẢ 6 chỉ số, và nhỏ hơn 34 MB (doc §18.58).** V9 = config V6 (`token_embd` q2_K + `output` q5_K) + **nguồn BF16 Instruct-2507** + imatrix cũ 2200. So V6: MMLU **+1,23**, Belebele-VI **+4,00**, HumanEval **+3,05** (136→141), MBPP **+2,00** (299→309), IFEval prompt-strict **+4,00**, inst-strict **+2,84** — **5/5 bài cùng chiều lên (p=3,1%)**, gộp 1 484 mẫu 1 047→1 079 = **+2,16 điểm (±1,66)**. **Tỉ lệ giữ được so Q4_K_M 17,35 GiB, E2 → V9:** MMLU 90,6→**91,0** · Belebele-VI 91,6→**94,7** · HumanEval 88,4→**91,0** · MBPP 82,2→**84,7** · IFEval inst-strict 88,9→**94,2** · IFEval prompt-strict 83,7→**91,8 (+8,1)**. Hai thay đổi tạo ra kết quả, **cả hai tốn 0 byte**: (1) embed/output đổi phân bổ (thực ra −34 MB), (2) nguồn BF16 bỏ một tầng làm tròn (tốn 61 GB tải một lần). **Không đóng góp gì**: imatrix bổ sung miền. **Cơ chế chưa giải thích được**: mô hình cộng sai số bình phương của tôi dự báo expert ở 1,56 bpw hầu như không hưởng lợi từ nguồn sạch (√(18²+1,5²)=18,06 vs 18) — thực tế trái ngược. Giả thuyết cần kiểm: `llama-quantize` chọn thang đo/mã lưới cho khối iq1_s **dựa trên giá trị đã bị Q4_K làm tròn**, nên sai số không độc lập và nguồn sạch giúp chọn đúng hơn ở chính phần chiếm 86,8% byte.

- [10/09 22:20] **Đường liều-đáp ứng phủ định hướng 1b CHẮC CHẮN hơn (doc §18.56).** Người dùng đẩy tôi thử tiếp thay vì bỏ — đúng, vì nó biến một kết quả âm đơn lẻ thành đường liều-đáp ứng. V6 (w=0) / V10 (w=0,3) / V7 (w=1,0): MMLU **71,40 / 70,70 / 70,88**, Belebele-VI **82,00 / 79,60 / 78,40** (**giảm đơn điệu theo liều**), gộp 820 mẫu **612 / 602 / 600**. Trong khi wiki PPL **cải thiện đơn điệu** 7,9395 / 7,8767 / 7,8343 — **lần thứ tư trong phiên chỉ số trung thực đi ngược tác vụ**. Nhưng giới hạn của kết luận: nó phủ định **ngữ liệu B** (nhân văn/VI/code — ba miền model ĐÃ giữ 88–92%), không phủ định hướng 1b nói chung.

- [10/09 22:30] **PHÁT HIỆN từ bộ đo đa ngữ mới: tổn thất do nén CỰC KỲ không đều giữa các ngôn ngữ (14 ngôn ngữ × 200 câu, 0 unparsed).** Tỉ lệ giữ được của V6 so Q4_K_M: Anh **100,5%** · Trung 95,7 · Pháp 94,2 · Ả Rập 94,0 · Hàn/Tây Ban Nha 92,9 · Nhật/Thái 92,7 · Nga 92,4 · Indonesia 92,3 · Việt 91,2 · Đức 90,8 · **Hindi 85,7** · **Khmer 58,3** (78,00 → **45,50**, mất 32,5 điểm = **6,5σ**). Tổng 90,00 → 81,79 = 90,9%. **12 ngôn ngữ nằm gọn 90–100%, hai ngôn ngữ chữ viết riêng biệt ít tài nguyên thì sụp.** → **Lật lại lời tôi nói rằng ngữ liệu D là lỗ**: model gốc CÓ SẴN năng lực Khmer 78%, nén đã phá nó không cân xứng, và imatrix hiện tại gần như không có tín hiệu Khmer nào nên các kênh đó bị lượng tử hoá mù. Đây là **thiếu hụt đủ lớn đầu tiên trong dự án để hiệu chuẩn có thể lấy lại** (32,5 điểm, so với 4–9 điểm ở 12 ngôn ngữ kia). Đang tính imatrix D; tiêu chí thành công đo được cả hai phía: Khmer/Hindi lên bao nhiêu, và 12 ngôn ngữ kia trả giá bao nhiêu.

- [10/09 23:15] **TRẦN THẬT đo được: Q4_K_M gần như KHÔNG mất gì so với BF16 — 99,4 % (doc §18.59).** Đo BF16 56,90 GiB trên **A100-80GB** (A10G 24 GB không đủ; tạo bản sao `*_big.py` thay vì sửa file gốc). Q4_K_M 17,35 GiB giữ được so BF16: MMLU **99,3 %** · Belebele-VI **100,0** · XCOPA-VI **100,0** · XNLI-VI 98,3 · HumanEval 98,7 · MBPP **100,3** (365 vs 364 câu, tức bằng nhau) → trung bình **99,4 %**, bốn chỉ số trùng khớp tuyệt đối, `unparsed = 0` cả hai. **→ Điều tôi cảnh báo trước khi đo KHÔNG xảy ra: mọi tỉ lệ giữ được đã báo suốt phiên lệch dưới 1,2 điểm so với trần thật, KHÔNG cần sửa.** **PHÁT HIỆN CHÍNH: toàn bộ cái giá nằm ở nửa sau** — 56,90→17,35 GiB là **3,28× mất 0,6 %**, còn 17,35→6,4003 GiB là **2,71× mất 5–15 %**; tổng V9 nén **8,89×**. Định lượng được một đánh đổi trước đây chỉ cảm tính: **chấp nhận file 8–10 GiB thì lấy lại được phần lớn khoảng cách**, vì ở 4,8 bpw gần như không mất gì. Trần code: **HumanEval BF16 95,73 % (157/164), MBPP 72,80 % (364/500)** → V9 giữ 89,8 % và 84,9 %; E2 giữ 87,3 % và 82,4 % → **V9 hơn E2 +2,5 điểm ở cả hai bài code**. Kỹ thuật: `hf_transfer` tải 61 GB trong 253 s, `llama-gguf-split --merge` ghép 79 s, BF16 chạy trọn VRAM 80 GB không cần offload.

- [10/09 23:50] **V11 (imatrix A+0,3·C toán/diff git, nguồn BF16): âm tính thứ BA cho hướng 1b.** So V9 (cùng nguồn BF16, chỉ khác imatrix, **dung lượng y hệt 6,4003 GiB**): MMLU 72,63→72,28 (−0,35), Belebele-VI 86,00→85,20 (−0,80), wiki PPL 7,8274→7,9037 (xấu hơn), KLD 0,3569→0,3535 (tốt hơn). Gộp 820 mẫu **629→625**. **Và đúng mẫu phân bổ lại như hai lần trước**: nhân văn **+2,30**, STEM **+1,58**, nhưng xã hội **−5,83**. Ba ngữ liệu/liều đã thử đều âm: B@w=1, B@w=0,3, C@w=0,3. Phân bổ expert của A+0,3·C là **tốt nhất từ đầu** (min 1 680, lệch max/tb **5,96×** so với 7,24× của A) mà điểm tác vụ vẫn không lên → **phân bổ imatrix đều hơn KHÔNG kéo theo chất lượng cao hơn**, một kết quả âm tính nữa đáng ghi. Còn chờ code battery của V11 vì MBPP là chỉ số C được thiết kế riêng để nhắm.

- [10/09 23:55] **Ba imatrix bổ sung đã dựng xong, có thống kê phân bổ đầy đủ.** A gốc: min 1 111, lệch max/tb 7,24×, tb/min 63,4×. A+0,3·B (nhân văn/VI/code): 1 416 / 6,49 / 64,6. A+0,3·C (toán/suy luận/**diff git 473 commit, 5,41 MB**): 1 680 / **5,96** / 54,5. A+0,3·D (**40 ngôn ngữ**): **1 818** / 7,55 / **50,3**. Riêng từng ngữ liệu thì cực lệch — B min 3 (lệch 10,54×), C min **1** (6,75×), D min 3 (**14,70×**, lệch nhất vì 40 tiếng nghĩa là định tuyến rất tập trung) — **nhưng gộp vào đều làm phân bổ tốt hơn bản gốc**, xác nhận lại nguyên tắc "bổ sung chứ không thay thế" của người dùng.

- [10/10 00:05] **So DeepSeek-V2-Lite-Chat với Qwen3-30B-A3B: model NHỎ chính xác CAO hay LỚN chính xác THẤP?** DeepSeek-V2-Lite = 15,7B tổng / 2,4B kích hoạt, 27 tầng, 64 expert + 2 shared, **MLA**, ra giữa 2024. Qwen3-30B-A3B-Instruct-2507 = 30,5B / 3,3B, 48 tầng, 128 expert top-8, GQA, tháng 7/2025. Hai điểm so chọn để DeepSeek **được ưu thế nhẹ**: `i1-IQ3_XXS` **6,96 GB** (V9 là 6,872 GB → DeepSeek lớn hơn 1,3 %) và `i1-Q4_K_M` **10,36 GB** (+51 %). Dùng bản `i1-*` (có imatrix) cho công bằng. Bài đo: đúng bộ của Qwen — MMLU 570, Belebele/XNLI/XCOPA-VI, **dịch vi↔en chrF++**, HumanEval 164 + MBPP 500 chạy thật. Lưu ý kỹ thuật: V2-Lite **không có chế độ thinking** nên harness tự bỏ `enable_thinking` nếu server từ chối. **Ngân sách KHÔNG thật thấp hơn** — phép so này trả lời "cùng dung lượng model nào tốt hơn", chưa phải "đổi sang đây tiết kiệm bao nhiêu"; nếu DeepSeek thắng thì mới bàn xuống `i1-IQ2_M` 6,33 GB (−7,8 %) hay `i1-IQ1_S` 4,99 GB (−27 %).

- [10/10 00:40] **HƯỚNG 1b ĐÓNG LẠI 4/4 ÂM TÍNH, và có kết quả MẠNH NHẤT của phiên (doc §18.61).** Belebele **14 ngôn ngữ × 200 câu = 2 800 mẫu/model**, `unparsed = 0` cả bốn model. **V12 (imatrix đa ngữ, có 300 KB tiếng Khmer, phân bổ expert tốt nhất trong 4 bản) thất bại đúng ở mục tiêu: Khmer 55,00 → 54,00 = −1,00**; tổng 83,21 → 83,57 = +0,36, sai số ±0,9 nên trong nhiễu. **Trong khi V6→V9 (chỉ đổi nguồn Q4_K_M→BF16, không nhắm ngôn ngữ nào): Khmer 45,50 → 55,00 = +9,50, tổng 81,79 → 83,21 = +1,42.** → **Minh chứng sạch nhất cho kết luận trung tâm: BỎ BỚT sai số thắng PHÂN BỔ LẠI sai số, và nó xảy ra đúng trên chỉ số mà phía "phân bổ lại" lẽ ra có lợi thế lớn nhất.** Bốn phép thử 1b: B@w=1 (Belebele −3,60), B@w=0,3 (−2,40), C@w=0,3 (MBPP +0,80 nhưng HumanEval −1,83, ròng +1 câu/664), D@w=0,3 (Khmer −1,00). **Quy tắc: imatrix chọn NƠI đặt sai số trên ngân sách bit cố định, không giảm LƯỢNG sai số, nên không lấy lại được năng lực đã bị nén phá — kể cả khi có tín hiệu đúng miền và thiếu hụt tới 32,5 điểm.**

- [10/10 00:45] **Tổn thất do nén CỰC KỲ không đều giữa các ngôn ngữ (doc §18.61).** Tỉ lệ giữ được của V6 so Q4_K_M theo ngôn ngữ: Anh **100,5 %** · Trung 95,7 · Pháp 94,2 · Ả Rập 94,0 · Hàn 92,9 · Tây Ban Nha 92,9 · Nhật 92,7 · Thái 92,7 · Nga 92,4 · Indonesia 92,3 · Việt 91,2 · Đức 90,8 · **Hindi 85,7** · **Khmer 58,3** (78,00→45,50, mất 32,5 điểm = **6,5σ**). Mười hai ngôn ngữ nằm gọn 90–100 %, hai ngôn ngữ chữ viết riêng biệt ít tài nguyên thì sụp. V9 kéo Khmer lên 70,5 % và tổng lên 92,5 %.

- [10/10 00:50] **KHÔNG nên đổi sang DeepSeek-V2-Lite (doc §18.62).** Ở cùng ngân sách (`i1-IQ3_XXS` 6,96 GB, thực ra **lớn hơn V9 1,3 %**): MMLU tổng **47,19 vs 72,63 = −25,44**, STEM 40,00 vs 68,42 = −28,42, **Belebele-VI 43,20 vs 86,00 = −42,80** (43,2 % gần mức đoán bừa 25 % → gần như không đọc hiểu được tiếng Việt). Dự đoán của tôi đúng hướng nhưng **thấp hơn thực tế rất nhiều**. Câu "đổi sang để lấy dung lượng thấp hơn" mất cơ sở vì nó đã thua 25–43 điểm ở mức dung lượng *cao hơn*. Vẫn chờ bản `i1-Q4_K_M` 10,36 GB để biết 47,19 % là giới hạn model hay là lỗi nén IQ3_XXS — nếu là lỗi nén thì có kết luận riêng đáng giá: **model 15,7B không chịu được nén sâu như model 30,5B**.

- [10/10 01:20] **KIỂM TOÁN XẾP HẠNG: KHÔNG tồn tại top-10 hợp lệ trong 79 bản của database.** Chia ba nhóm theo chất lượng số đo: **nhóm A (10 bản)** có bộ bài tác vụ thật; **nhóm B (73 bản)** chỉ có top-1/KLD/PPL — chính năm thước §18.60 đã chứng minh **không dự báo được chất lượng dưới 2 bpw**; **nhóm C (4 bản)** không có gì. Xếp hạng nhóm B là xếp hạng bằng thước đã biết là sai. Ví dụ rõ: `48L-NeuroHierarchical-E8-LoRA` 6,483 GiB có top-1 **75,39** (cao hơn E2 73,14) mà **chưa đo một bài tác vụ nào** — có thể tốt, có thể tệ, không biết được. Các bản nhóm B đáng đo lại nếu quay lại: `Qwen3-30B-A3B-Q2_K_48L_original` (10,49 GiB, top-1 85,70, KLD 0,154 — nằm đúng vùng 8–10 GiB mà §18.59 chỉ ra là còn dư địa), `48L-NeuroHierarchical-E8-LoRA` và `48L-CortexLadder-E8-LoRA` (6,36–6,48 GiB, top-1 73,9–75,4). Đã ghi bảng nhóm A vào `scratchpad\rank_groupA.json`.

- [10/10 01:25] **ĐÍNH CHÍNH: V9 và V12 KHÔNG phân biệt được — tôi đã gọi V9 là "tốt nhất" khi chưa đủ cơ sở.** Trên 4 chỉ số cả hai đều có: V12 hơn ở MMLU (**72,81 vs 72,63**) và tổng đa ngữ 14 tiếng (**83,57 vs 83,21**); V9 hơn ở Belebele-VI (86,00 vs 84,80) và Khmer (55,00 vs 54,00) — **tỉ số 2–2**, gộp 3 450 mẫu V12 hơn 8 câu tức trong nhiễu. V9 chỉ đang dẫn vì **được đo 7 bài so với 4 bài của V12**, đó không phải cơ sở để tuyên bố thắng. Đang chạy code battery + IFEval cho V12 để phân định.

- [10/10 02:10] **TẢI V9 và V12 về máy, và XÁC NHẬN `-t 8` bằng llama-bench.** Volume Modal vẫn đọc được dù workspace bị vô hiệu hoá, nên đã tải về `E:\gguf_golden\`: **`Qwen3-30B-A3B-V9-bf16src-embed.gguf`** và **`Qwen3-30B-A3B-V12-bf16src-imxD.gguf`**, cả hai 6 872 246 656 B đúng byte. Trước đó máy chỉ có E2 (bản cũ, kém hơn). V9 chạy tốt trên 3060 Ti: NIAH 16k **15/15 = 100 %** với `-ngl 99 --n-cpu-moe 12 -t 8 -ctk f16`. **`llama-bench` trên E2 xác nhận lợi ích `-t 8`: tg128 = 62,54 tok/s so với 46,4 tok/s của `-t 6` ở §18.42 → nhanh hơn 35 %**; pp512 = 1 438,86 tok/s. *Lưu ý không lẫn hai phép đo prefill:* 1 439 tok/s là bench 512 token với KV rỗng, còn **240 tok/s** là prompt 25 317 token thật với `-ncmoe 20` — prefill giảm dần theo độ dài (đã đo 305→246 tok/s từ 2k lên 18k), nên lập kế hoạch phải dùng số theo độ dài prompt thật. **Cấu hình khuyến nghị cập nhật: `-ngl 99 --n-cpu-moe 12 -fa on -t 8 -ctk f16 -ctv f16` (§18.42 dùng `-t 6` là chưa tối ưu).**

- [10/10 02:15] **Database lên v1.4.0: 88 bản.** Thêm V6/V7/V9/V10/V11/V12 (mỗi bản ghi rõ `cau_tao` = config + imatrix + nguồn lượng tử để tái lập), **trần BF16** (`BF16_Instruct2507_CEILING`), và hai bản DeepSeek-V2-Lite. Hai entry DeepSeek phải đọc từ **các dòng in giữa** vì workspace Modal bị cắt trước khi ghi `[RESULT]` — đã ghi `ghi_chu` rõ rằng **ô thiếu là CHƯA ĐO, không phải bằng 0**.

- [10/10 02:20] **⚠ WORKSPACE MODAL BỊ VÔ HIỆU HOÁ** (`workspace ac-M1DJCVB9SvzfqNqMKQa5By is disabled`). Lệnh đọc (`app list`, `volume ls/get`) vẫn chạy, mọi job tính toán bị cắt. Bị cắt giữa dở: MBPP+IFEval của V12 (hai số duy nhất còn thiếu để phân định V9 vs V12), MBPP của DeepSeek IQ3_XXS, en2vi+code của DeepSeek Q4_K_M, và **imatrix tính từ BF16** (hướng "bỏ bớt sai số" tiếp theo). Nguyên nhân gần chắc là đụng hạn mức chi tiêu — hôm nay đã dùng nhiều giờ A10G cộng A100-40GB và A100-80GB. Việc thuộc tài khoản người dùng, không tự xử lý.

- [10/10 02:45] **KIỂM CHỨNG CHÉO trên máy thật: mọi chênh lệch V9−E2 TÁI LẬP đúng chiều và đúng độ lớn.** Chạy lại E2 và V9 trên RTX 3060 Ti (build đã vá, `-ncmoe 12 -t 8`) so với Modal (A10G, upstream stock, trọn VRAM). **Ba trong bốn nhóm MMLU cho CÙNG con số chênh tới hai chữ số thập phân**: nhân văn **+1,54 / +1,54**, xã hội **+4,17 / +4,17**, khác **+1,54 / +1,54** (local/Modal); STEM −3,16 / −3,69; MMLU tổng +0,52 / +0,35; Belebele-VI +1,60 / +2,80; NIAH 16k 100 % cả hai. Số tuyệt đối của E2: local MMLU 71,23 vs Modal 72,28 (−1,05) và Belebele 83,60 vs 83,20 (+0,40) — **hai chênh lệch đi NGƯỢC chiều nhau nên là nhiễu ngẫu nhiên, không phải thiên lệch hệ thống**. → Sau sự cố §18.57 (hai file cùng tên khác model), đây là bằng chứng chuỗi đo không còn lỗi ẩn tương tự.

- [10/10 02:50] **ĐÍNH CHÍNH 1: "STEM −3,16 trong nhiễu" là SAI.** Khi báo V9 lần đầu tôi viết STEM −3,16 là "6 câu trên 190, trong nhiễu". Nó đã **tái lập ở môi trường thứ hai với cùng độ lớn** (−3,16 local, −3,69 Modal) → **là đánh đổi THẬT của V9**: thắng rõ ở xã hội (+4,17), nhân văn và khác (+1,54), nhưng **kém E2 khoảng 3,2–3,7 điểm ở STEM**. Ai dùng model chủ yếu cho toán/khoa học cần biết điều này.

- [10/10 02:55] **ĐÍNH CHÍNH 2: hai thay đổi của V9 KHÔNG "miễn phí" — chúng tốn 2,9 % tốc độ sinh.** `llama-bench` cùng cấu hình: E2 tg128 **62,54** tok/s, V9 **60,71** tok/s (−2,9 %); pp512 1 438,86 vs 1 419,31 (−1,4 %). Cơ chế: `output` lên q5_K (5,5 bpw thay vì 4,5) được đọc **mỗi token**, còn `token_embd` xuống q2_K chỉ là tra cứu nên không bù lại. Đổi 2,9 % tốc độ lấy +2,4…+6,8 điểm ở ba bài và −34 MB — vẫn rất lợi, nhưng phải nói đủ.

- [10/10 03:00] **ĐÍNH CHÍNH 3: con số 62,54 tok/s là tốc độ ĐƯỜNG OFFLOAD, không phải tốc độ của model.** Người dùng chỉ ra rằng nếu nạp trọn VRAM thì phải đạt ~11x tok/s — đúng, và lỗi ở cấu hình bench của tôi: tôi dùng `--n-cpu-moe 12` (12/48 tầng expert đọc qua DDR4 ~25 GB/s thay vì VRAM 448 GB/s, chậm hơn 18× ở phần đó) vì đó là cờ cần cho NIAH ctx 16k, rồi dùng luôn cho bench mà không nghĩ lại. Nhưng `llama-bench -p 512 -n 128` chỉ cần KV ~**60 MiB** nên 6,4 GiB **vừa 8 GB ở ncmoe 0**. Ngân sách: bench ncmoe 0 = 6,40 + 0,06 + ~0,5 = **~7,0 GiB vừa**; ctx 16k ncmoe 0 = 6,40 + **1,50** + 0,8 = 8,70 GiB **không vừa**. Các bản 5,2–5,5 GiB đạt 111–137 tok/s vì **nhỏ hơn nên vừa VRAM ở ctx 16k**, không vì nhanh hơn bản chất. Đang quét `ncmoe = 0/4/8/12/20` trên cả E2 và V9. **Cấu hình khuyến nghị phải chia theo mục đích: ngữ cảnh ngắn → ncmoe 0 cho tốc độ tối đa; ngữ cảnh dài → mới offload và trả giá tốc độ.**

- [10/10 03:20] **BẢNG LOCAL HOÀN CHỈNH E2/V9/V12 trên RTX 3060 Ti** (`-ngl 99 --n-cpu-moe 12 -fa on -t 8 -ctk f16`, ctx 16k). Gộp 820 mẫu (MMLU 570 + Belebele-VI 250): **E2 615 · V9 622 · V12 633**. Chi tiết: tg128 62,54/60,71/58,91 · pp512 1438,86/1419,31/1436,99 · NIAH 16k **100 % cả ba** · MMLU **71,23/71,75/72,63** · Belebele-VI **83,60/85,20/87,60**. **Gộp CẢ HAI môi trường (1 640 mẫu): E2 1 235 · V9 1 251 · V12 1 260** → V12 hơn V9 **9 câu = +0,55 điểm (±1,6)**, **vẫn không phân định được** dù đã gấp đôi dữ liệu; V12 hơn E2 25 câu = +1,52 điểm (~0,95σ). **Kết luận đúng mức: V9 và V12 đều hơn E2, và hai bản đó không phân biệt được với nhau.** Hai số còn thiếu để phân định vẫn là MBPP + IFEval của V12 (cần Modal).

- [10/10 03:25] **Thứ tự MMLU `E2 < V9 < V12` TÁI LẬP ở cả hai môi trường**, và V12 lấy lại phần STEM mà V9 mất: V12−V9 theo nhóm = STEM **+2,63/+0,53**, Khác +2,31/+0,77, Nhân văn 0,00/+0,77, Xã hội −2,50/−1,67 (local/Modal) — ba trong bốn nhóm cùng chiều. → **Nói lại cho đúng mức về "hướng 1b âm tính 4/4"**: kết luận đó vẫn đúng cho các *chỉ số mục tiêu* (Khmer −1,00, MBPP ròng +1 câu/664), nhưng imatrix bổ sung miền **KHÔNG phải "không làm gì"** — nó **dịch chuyển hồ sơ năng lực**, và hướng dịch chuyển **không khớp miền được nhắm**: ngữ liệu D là 40 ngôn ngữ mà lợi ích lại đến ở MMLU/STEM còn tiếng Việt/Khmer thì mất.

- [10/10 03:30] **RÚT LẠI: "V9 hơn V12 ở tiếng Việt" KHÔNG được xác nhận.** Belebele-VI là chỗ duy nhất hai môi trường **ngược chiều nhau**: V12−V9 = **+2,40 trên máy thật** nhưng **−1,20 trên Modal**, lệch 3,6 điểm. Với n=250 và sai số ±2,3 thì mỗi chênh chỉ ~1σ → **không phân định được ở cỡ mẫu này**. Khác hẳn MMLU theo nhóm, chỗ đó trùng khớp tới hai chữ số thập phân. Bài học: **một chênh lệch chỉ đáng tin khi nó tái lập; n=250 không đủ cho chênh lệch 1–3 điểm.**

- [10/10 03:35] **RÚT LẠI đính chính "V9 chậm hơn E2 2,9 %".** V12 làm đối chứng nhiễu: nó **cùng kiểu tensor và cùng dung lượng tới từng byte** với V9 (6 872 246 656 B) — giá trị trọng số không thể ảnh hưởng tốc độ giải lượng tử — mà tg128 chênh **3,0 %** (58,91 vs 60,71). Vậy 3,0 % đó chắc chắn là **nhiễu đo với `-r 2`** trên GPU tiêu dùng có xung nhịp dao động, và chênh 2,9 % giữa E2 và V9 nằm trong **cùng biên độ nhiễu** → không khẳng định được. Đang chạy lại với **`-r 6`** và V12 làm đối chứng để đo trực tiếp biên độ nhiễu.

- [10/10 03:50] **QUÉT TỐC ĐỘ `ncmoe 0…20`: nạp trọn VRAM đạt 137 tok/s, gấp 2,26× con số tôi báo trước.** Người dùng chỉ ra 62 tok/s là quá thấp và **họ đúng**. Bảng tg128 (`-r 6`, E2/V9/V12): **ncmoe 0 = 137,02±0,08 / 135,43±0,56 / 132,70±1,82** · ncmoe 4 = 96,72/97,31/95,11 · ncmoe 8 = 75,49/74,74 · ncmoe 12 = 60,64/60,29 · ncmoe 20 = 38,36/37,30. pp512 ở ncmoe 0 = **2 240/2 252** (không phải 1 439). **Quy về thời gian mỗi token thì đường cong rất gọn:** 7,30 → 10,34 → 13,25 → 16,50 → 26,04 ms, tức **mỗi tầng expert đẩy sang CPU tốn ~0,78 ms/token** (0→12 tầng), lên 1,19 ms/tầng (12→20). **CẤU HÌNH KHUYẾN NGHỊ CHIA THEO MỤC ĐÍCH** — ngân sách 8 GB, trọng số 6,4 GiB, KV 96 KiB/token: chat/dịch/câu ngắn → `-ncmoe 0`, **137 tok/s**, ctx ~4k (KV 0,375 GiB, tổng 7,6 GiB sát vách đá); cân bằng → `-ncmoe 4`, 97 tok/s, ~8k; đọc tài liệu → `-ncmoe 12`, 61 tok/s, 16k; ngữ cảnh dài → `-ncmoe 20`, 38 tok/s, 32k.

- [10/10 03:55] **XÁC NHẬN việc rút lại "V9 chậm hơn E2 2,9 %".** Với `-r 6`: ncmoe 12 cho E2 **60,64** vs V9 **60,29** = chênh **0,6 %**, và ở ncmoe 4 thì V9 còn **NHANH HƠN** (97,31 vs 96,72). Lượt `-r 2` trước đó cho 62,54 vs 60,71 tức lệch 3 % — **`-r 2` không dùng được trên GPU tiêu dùng có xung nhịp dao động**. → **V9 không chậm hơn E2 một cách đo được.** Quy tắc: llama-bench phải `-r 6` trở lên, và nếu có hai model cùng kiểu tensor + cùng dung lượng thì dùng chúng làm **đối chứng nhiễu** (ở đây V9 vs V12 cho biết biên độ nhiễu là ~3 % với -r 2 và <1 % với -r 6).

- [10/10 04:00] **⚠ Modal: `Workspace has exceeded its spend limit`** — xác định bằng phép thử 1 CPU/không GPU/2 giây, nên **không phải vấn đề context hay tài nguyên job**. MBPP + IFEval của V12 chỉ dùng ctx 8192, nhẹ hơn nhiều thứ đã chạy hôm nay. Cần nâng hạn mức hoặc nạp thêm trong tài khoản Modal. Người dùng đã quyết **bỏ hướng biến thể 8–10 GiB**.

- [10/10 05:10] *(⚠ hai chỉ số ĐẾM trong mục này — CJK rớt và số câu sập — ĐÃ BỊ RÚT LẠI ở mục 06:30; các số chrF++/BLEU vẫn đúng)* **BỘ DỊCH THUẬT KHÓ 7 chiều × 3 bản — km→vi tách được ba bản, sáu chiều còn lại không.** Trước hôm nay **chưa hề đo dịch cho V9/V12**, và bài dịch duy nhất từng chạy là en↔vi đoạn ngắn tức cặp **dễ nhất**. Thêm 5 chiều khó (bộ 60 đoạn dài nhất; zh/ja/km làm nguồn; vi→zh sinh chữ Hán). Kết quả chrF++ (E2 / V9 / V12): en→vi chuẩn 57,06/56,92/56,35 · vi→en 58,25/58,99/58,48 · en→vi khó 56,51/56,68/56,51 · zh→vi 51,48/50,98/50,84 · ja→vi 47,83/47,66/47,42 · vi→zh 21,11/21,63/21,65 · **km→vi 45,65 / 37,64 / 45,47**. **Biên độ nhiễu đo từ chính cặp đối chứng V9/V12** (cùng kiểu tensor, cùng dung lượng từng byte) là **0,02–0,57** trên sáu chiều, nên **−7,83 của Khmer là thật**. CJK rớt tổng 7 chiều: **E2 18 → V9 8 → V12 5**, đúng hướng ở 4/5 chiều có rớt → **đây mới là phát biểu đúng cho lợi thế nguồn BF16: giảm rác ký tự**, không phải "lấy lại năng lực ngôn ngữ ít tài nguyên".

- [10/10 05:15] **MỔ CHIỀU KHMER — thâm hụt ở CHẤT LƯỢNG NỀN, không chặn được bằng sampler.** Chạy lại km→vi bằng harness v2 (đã lọc trùng, có lưu văn bản) rồi chấm trên **tập giao sạch** = 51/60 đoạn không bản nào sập: E2 **45,21**, V12 **44,68**, V9 **41,15** (BLEU 18,85 / 18,68 / 15,33). Hiệu ứng **lặp lại** (−5,11 trên bộ đã lọc trùng) và **không mất khi bỏ câu sập** (−4,06). Số câu sập: E2 5, V12 2, V9 6. → repetition penalty **không** sửa được. **Quy trách nhiệm:** V12 dùng *cùng* nguồn BF16 + *cùng* config V6, khác V9 đúng một thứ là **imatrix**, mà không tụt → nguồn BF16 **không** phải thủ phạm đơn độc, thâm hụt đi theo tổ hợp **(BF16 + imatrix A)**. Chưa tách được khỏi "V9 là một lần lượng tử gặp may xấu" vì mỗi ô 1 mẫu; phép thử phân định là **V11** (BF16 + A+0,3×C, C không có ngôn ngữ nào) — chỉ có trên Modal, đang bị chặn hạn mức.

- [10/10 05:20] **⚠ ĐẢO KẾT LUẬN "bổ sung miền cho imatrix: 4/4 âm tính".** Kết luận đó dựa trên MMLU/Belebele/HumanEval/MBPP/IFEval — **không bài nào bắt model SINH văn bản từ ngôn ngữ ít tài nguyên**, tức không có khả năng thấy kết quả dương. Ngữ liệu D (40 ngôn ngữ) **giữ km→vi ở mức E2** trong khi V9 không có D thì tụt 4 điểm. Người dùng đã yêu cầu đúng hướng này ("giữ imatrix cũ, chỉ bổ sung miền yếu") và **họ đúng**; bộ đo của tôi mù với chỗ nó phát huy tác dụng. **Quy tắc mới: trước khi tuyên bố một hướng âm tính, phải kiểm bộ đo có bài nào đo được đúng thứ hướng đó nhắm vào.** Không bác §18.61 — D không kéo Khmer **lên trên** E2 (44,68 vs 45,21), nó chỉ **chặn một hồi quy**. Cũng phải rút lại suy diễn "đọc hiểu Khmer +9,50 ⇒ BF16 lấy lại năng lực Khmer": V9 **nhất ở đọc hiểu Khmer nhưng bét ở dịch Khmer** — hai năng lực khác nhau, không suy từ trắc nghiệm sang sinh văn bản.

- [10/10 05:25] **ĐÁNH GIÁ ĐỊNH TÍNH: đọc 12 bản dịch thật, xếp hạng gần NGƯỢC với chrF++.** vi→zh bị chrF++ xếp **bét (21,11)** nhưng là bản **tốt nhất** cả bộ (Trung văn trôi chảy, chỉ hỏng tên riêng: `格陵岛` thay `格陵兰` sai nhất quán **6 lần/đoạn**). km→vi được 45,65 nhưng là **bịa đặt trôi chảy** — "Nhờ có cáp quang biển"→"**Xin cảm ơn bạn đã kết nối cáp điện thoại di động**"; đoạn muỗi không truyền bệnh→"**đồ dùng kiểu Nordic… họ vẫn khiến người khác cảm thấy lạnh**". Ba lớp lỗi theo mức khó phát hiện: (1) **từ hỏng** `quá khứ`→`quá khích`, `thay đổi`→`hằn đổi`/`hằn hối` — *cùng chỗ hỏng ở hai chiều khác nhau nên là tật cố định*; (2) **xoá từ giảm nhẹ** "will **likely**"→"**chắc chắn**", mất `có lẽ`, `ほとんど`→"không có" — biến phỏng đoán thành khẳng định và câu đọc *hay hơn* nên không ai nghi; (3) **bịa đặt trôi chảy**. Ba mức dùng: dùng ngay = vi→zh, en↔vi (chỉ để nắm ý) · kèm đối chiếu nguồn = zh→vi, ja→vi · **không dùng** = km→vi.

- [10/10 05:30] **4 LỖI HARNESS của chính dự án, đã đo và vá — xem `EVAL_PROTOCOL.md §10`.** (1) **chrF++ có sàn riêng theo hệ chữ**, đo bằng đoạn tham chiếu *không liên quan* cùng ngôn ngữ: →vi 19,12 · →en 23,45 · →zh **3,16** · →ja 6,77 · →km 18,50 → `vi→zh 21,11` là **6,7× sàn** và **không so được** với `ja→vi 47,83`; so cùng đích thì hợp lệ (xếp hạng độ khó en>zh>ja **vẫn đúng**). (2) **Belebele 900 dòng chỉ có 488 đoạn khác nhau** — lọc theo khoá `(link, question_number)` không lọc được trùng nội dung: bộ "120 cặp" Modal → **69 đoạn thật**, bộ "60 đoạn" local → **32–33**; sai lệch điểm **<0,35** không nhất quán dấu nên **số cũ giữ nguyên**, chỉ khoảng tin cậy rộng hơn **1,32×**. (3) **Bộ đếm lặp mù với chữ Hán/Nhật** (tách theo khoảng trắng): chỉ **14/200** đoạn Trung và **0/200** đoạn Nhật đủ dài để nó chạy → `lặp 0` ở vi→zh **vô nghĩa**. (4) **chrF++ che tổn thất theo độ dài**: đoạn dài 1,58× thì chrF++ xuống 0,55 mà BLEU xuống **2,09**. Lỗi nặng nhất về phương pháp là **v1 chỉ lưu điểm, không lưu bản dịch** → không đánh giá định tính được, không chấm lại được. `local_translate_hard2.py` lưu nguồn + tham chiếu + bản dịch + cờ sập từng đoạn.

- [10/10 05:45] **CHẤM MÙ BẰNG MẮT NGƯỜI — thang điểm nén khoảng cách xuống 5 lần.** Xáo nhãn ba bản độc lập từng đoạn, **ẩn bản tham chiếu tiếng Việt** (dùng đoạn tiếng Anh song song làm chuẩn nội dung, tránh đúng cái bẫy của chrF++), chọn đoạn theo bước đều, ghi phán quyết TRƯỚC khi mở khoá. 8 đoạn × 3 bản. Kết quả: **V12 20,5 · E2 19,0 · V9 8,5** — **V9 trắng 0/8 đoạn**, và **toàn bộ** thảm hoạ toàn đoạn đều là V9 (bỏ phiếu Pháp → **chuyện rút thăm xổ số**; đặt phòng → **bịa "người bảo vệ"**; muỗi Bắc Âu → **"đồ ăn truyền thống Nordic"**; **Iraq → Iran**; Tuyên ngôn Độc lập → **"cuốn sách"**). **Phát hiện chính: chrF++ cho V9 đạt 91 % điểm của E2, chấm mù cho 45 %** — thang đo nén khoảng cách ~5 lần, vì **bịa đặt trôi chảy vẫn ăn điểm n-gram** (đoạn tiếng Việt mượt nói về xổ số vẫn trùng `người`/`trong`/`đã được`/`theo quy trình` với tham chiếu). **Chỗ không đồng ý:** thứ tự E2 vs V12 bị đảo (chrF++ 45,21 vs 44,68; chấm mù 19,0 vs 20,5) — cả hai khoảng cách nhỏ → **E2 và V12 không phân biệt được**, chỉ V9 kém rõ. **QUY TẮC: với bài SINH văn bản, thang n-gram chỉ dùng để XẾP HẠNG, không dùng để ước lượng ĐỘ LỚN chênh lệch** — 4 điểm chrF++ có thể là "kém chút" hoặc "thua trắng mọi đoạn". Xem RESEARCH §18.63.

- [10/10 06:30] **⚠ RÚT LẠI "nguồn BF16 giảm rác ký tự CJK 18 → 8 → 5" (ghi ở mục 05:10).** Chạy lại đủ 7 chiều × 3 bản bằng harness v2 (**bộ đã lọc trùng, 60 đoạn thật**): CJK rớt tổng là **E2 10 · V9 8 · V12 7** — sai số Poisson 10±3,2 so 7±2,6 nên **KHÔNG phải chênh lệch thật**. Con số 18 của v1 phồng lên vì bộ đó đếm trùng và bộ đoạn khó của nó dài hơn (827 vs 754 ký tự → nhiều rớt hơn). **Cũng rút lại việc dùng tổng số câu sập để xếp hạng:** v1 cho E2 16 / V9 18 / V12 14, v2 cho E2 15 / **V9 8** / V12 11 — V9 từ tệ nhất thành tốt nhất chỉ vì đổi bộ đoạn. Đếm nhỏ, phụ thuộc mạnh vào bộ đoạn → **không phải chỉ số xếp hạng**. Bài học: đừng công bố chỉ số đếm-sự-kiện trước khi chạy trên bộ đã lọc trùng và ít nhất hai bộ đoạn khác nhau.

- [10/10 06:35] **THÂM HỤT KHMER CỦA V9 — bằng chứng dứt điểm, lặp lại lần thứ ba.** v2 đủ 7 chiều: km→vi **E2 44,70/BLEU 18,00 · V9 39,23/13,17 · V12 44,39/18,44**. Quyết định nhất: ở chiều này **V9 SẬP ÍT HƠN E2 (4 câu vs 7)** mà vẫn **thua 5,47 chrF++ và 4,83 BLEU** → sập ít hơn nhưng điểm thấp hơn nhiều thì **không còn giải thích được bằng vòng lặp**, xác nhận §18.62.3 (thâm hụt ở chất lượng nền). Sáu chiều còn lại ba bản chênh ≤0,74 chrF++ và ≤1,35 BLEU. **E2 và V12 không phân biệt được ở cả 7 chiều** (BLEU trung bình 28,60 vs 28,51); V9 kém đúng một chiều (27,76 trung bình, toàn bộ sai biệt đến từ Khmer).

- [10/10 06:40] **vi→zh: BLEU đúng tokenizer XÁC NHẬN phán quyết đọc bằng mắt, chrF++ mặc định sai hẳn.** Cùng một tập bản dịch: chrF++ **21,89** (xếp bét 7 chiều) nhưng **BLEU[`tokenize=zh`] 33,52** (xếp thứ ba, ngang en→vi 34,45), chrF thuần ký tự 28,95. Lệch **hơn 11 điểm** giữa hai thang, toàn bộ do thành phần n-gram TỪ của chrF++ ≈ 0 khi đích không có khoảng trắng. Xếp hạng theo BLEU đúng tokenizer (en→vi 34,45 ≈ **vi→zh 33,52** > vi→en 30,62 > zh→vi 26,00 > ja→vi 23,17 > **km→vi 18,00**) **trùng với chấm mù bằng mắt** ở §18.63 — hai phương pháp độc lập, cùng ngược với chrF++ mặc định. Bản vá bộ đếm lặp cũng hiệu lực: v1 báo `lặp 0` cho vi→zh (mù với chữ Hán), **v2 bắt được 1–2**. Ba bản đều sinh chữ Hán **ngắn hơn tham chiếu** (dài/ref 0,92 / 0,85 / 0,85) — hụt nội dung nhẹ, chrF++ không thấy.

- [11/09 01:00] **ĐỔI BASE sang `Qwen3.6-35B-A3B` — có thinking bật/tắt, mới hơn V12 chín tháng.** Base cũ `Instruct-2507` là biến thể KHÔNG suy nghĩ (template không có `enable_thinking`, đã kiểm trong GGUF). Tra HF: `Qwen3.8-Flash-Next` 180B → 37,7 GiB ở 1,8 bpw nên LOẠI; `Qwen3.8-27B` dense nên LOẠI; **`Qwen3.6-35B-A3B`** 34,66B/~3B kích hoạt, **256 expert top-8**, 40 tầng, **attention lai `LLLF` × 10** (full ở tầng 3,7,…,39), ngữ cảnh 262K. Nguồn `ggml-org` (org của llama.cpp): BF16 một file 69,38 GB, **Q4_K_M chuẩn** 20,42 GB — không dùng `unsloth` vì repo đó chỉ có `UD-Q4_K_M` (đã tinh chỉnh per-tensor = đối thủ kỹ thuật, không phải mốc trung lập). Cả `conversion/` lẫn runtime của `E:\llama.cpp` đều đã hỗ trợ `Qwen3_5MoeForConditionalGeneration` / `LLM_ARCH_QWEN35MOE`. **⚠ Số tham số: HF ghi 35,95B nhưng gồm tháp thị giác ~1,3B; GGUF văn bản là 34,66B — dùng 35,95B làm bpw thấp hơn thực tế 3,7 %.**

- [11/09 01:30] **THANG ĐIỂM Q4_K_M mới, và thinking đáng +5,27 điểm MMLU.** 19,017 GiB: MMLU **83,33** (base cũ 79,82 → **+3,51**, cùng harness) · MMLU **thinking BẬT 88,60** · Belebele-VI 89,60 (n=250) / 86,67 (n=150) · MBPP 73,20 · HumanEval 92,07 (n=164) / 96,34 (n=82) · en→vi 61,06/38,66 · vi→en 62,44/34,21 · ja→vi 53,02/27,99. **Cỡ mẫu ảnh hưởng thật:** Belebele lệch 2,93 điểm giữa n=250 và n=150; HumanEval lệch 4,27 giữa n=164 và n=82 → mọi so sánh phải cùng n. **HumanEval 48,78 ban đầu là BUG harness của tôi** (ghép `prompt + "\n" + code` phá thụt lề khi model trả về chỉ phần thân); vá đúng = chỉ bù thụt cho DÒNG ĐẦU, các dòng sau đã mang thụt lề tuyệt đối.

- [11/09 02:00] **TỈA EXPERT bị số liệu phủ định.** Đọc `.counts` trong imatrix (256 phần tử/tensor, 1200 chunk): top8 = 4,83 % · **bottom128 = 41,81 %** · `max/min = 3,0` · **0 expert dưới 1 % trung bình**. Phân bố cực phẳng — Qwen huấn luyện 256 expert này dùng ĐỀU NHAU, không có expert thừa. Giả thuyết của tôi ("256 expert nên mỗi cái ít quan trọng, tỉa sẽ rẻ") **bị bác bỏ ngược**. Ngưỡng đặt TRƯỚC khi đo là 35 %; kết quả vượt cả ngưỡng xấu nhất.

- [11/09 03:00] **★ BẢNG ĐỘ NHẠY `qwen35moe`: NÉN ATTENTION ĐẮT GẤP 28–127 LẦN NÉN EXPERT.** Đổi đúng một nhóm sang `iq1_s`, chấm bằng bài tác vụ (không dùng PPL vì §9 đã chứng minh nó không dự báo): `ffn_down_exps` tiết kiệm 4,50 GiB → MMLU −2,80 (**0,62/GiB**), Belebele **+2,00 tức KHÔNG thiệt hại** (0,0/GiB) · `ffn_gate/up_exps` tiết kiệm 8,17 GiB → −5,96 (0,73/GiB), Belebele −2,67 (0,33/GiB) · **attention + SSM tiết kiệm chỉ 1,27 GiB → MMLU −26,14 (20,6/GiB), Belebele về 33,33 tức sát sàn ngẫu nhiên 25 % (42,0/GiB)**. **Bảng này KHÁC bảng Qwen3-30B** — xác nhận lần thứ ba rằng độ nhạy không chuyển giao: OLMoE (interm. 2048) → `gate_up` là thủ phạm; Qwen3-30B (768) → `down`; **Qwen3.6-35B (512) → `attention`**. Tôi đã sinh config cho `down` NHIỀU bit nhất theo tiền lệ 30B — giả định sai, `down` là nhóm chịu nén TỐT nhất.

- [11/09 03:30] **★ PRESET llama.cpp PHÂN BỔ SAI: +28,24 điểm MMLU chỉ từ cách chia bit.** L1 preset `IQ1_S` 7,144 GiB → MMLU **50,18**; **C1 config** 7,524 GiB → **78,42**. Cùng lớp bpw, chênh dung lượng 5,3 %. Nguyên nhân đã định lượng: preset ép toàn bộ attention+SSM xuống 1,56 bpw để lấy 1,27 GiB trên 19 — giá là 26 điểm MMLU. **C1 còn hơn preset `IQ2_S` ở 10,047 GiB tới 2,63 điểm dù nhỏ hơn 25 %** (75,79 vs 78,42). Thang preset đầy đủ: L1 7,14→50,18 · L2 9,02→71,58 · L3 10,05→75,79. **C1 đủ 7 bài, giữ 94,7 % Q4 ở 39,6 % dung lượng**: vi→en ~100 % · en→vi ~100 % · ja→vi 97,9 % · Belebele 96,2 % · MMLU 94,1 % · HumanEval 88,6 % · MBPP 85,9 %. **Hơn V12 5,61 điểm MMLU ở cùng bpw, cộng thinking.**

- [11/09 04:00] **`Q1_0` (1,125 bpw) PHÁ MODEL — bản 6,43 GiB thất bại, và mục tiêu 6,4 GiB KHÔNG đạt được.** Lấy đúng config đã cho 78,42, chỉ đổi `ffn_gate/up_exps` từ `iq1_s` sang `Q1_0`: MMLU 45,96 · Belebele **18,67 với 96/150 câu không đọc được** · en→vi chrF++ **12,00 tức DƯỚI SÀN 19,14** · ja→vi BLEU 0,38 · **HumanEval 0,00** · MBPP 17,50. Từ 1,5625 xuống 1,125 bpw là **VÁCH ĐỨNG**, tôi đã giả định là dốc và sai. **Cơ chế:** MMLU không đọc được 0/570 nhưng Belebele 96/150 — cả hai chỉ đòi một chữ cái, khác biệt duy nhất là Belebele có ĐOẠN VĂN DÀI → **prompt dài vỡ trước prompt ngắn**, nên bộ đo chỉ có prompt ngắn sẽ BỎ SÓT lỗi này. Chưa tách được (a) 1,125 bpw quá thấp cho `gate/up` với (b) kiểu `Q1_0` hỏng cho arch này. **RÚT LẠI** phát biểu "mục tiêu 6,40 GiB đã khả thi".

- [11/09 04:30] **Tốc độ thật trên 3060 Ti, bản 6,43 GiB, `-r 6`:** `ncmoe 0` (trọn VRAM) **pp512 1966 ± 51 · tg128 115,63 ± 2,58 tok/s** · ncmoe 4 → 1632/101,37 · ncmoe 8 → 1443/90,93. **Nạp trọn VRAM được, không cần offload.** So V12 ở `ncmoe 0` (137,02) → **84 %**; giả thuyết chưa kiểm là 30/40 tầng dùng SSM mà llama.cpp tối ưu chưa bằng attention thường.
