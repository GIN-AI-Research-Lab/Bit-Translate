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
