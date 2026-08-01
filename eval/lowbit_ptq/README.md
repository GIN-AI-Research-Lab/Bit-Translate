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

---

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
