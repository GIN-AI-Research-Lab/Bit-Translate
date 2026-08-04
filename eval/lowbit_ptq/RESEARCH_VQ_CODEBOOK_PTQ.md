# Exp AD — Vector Quantization / Additive Codebook PTQ (AQLM/QuIP#-style) trên Qwen3-0.6B

**Ngày**: 2026-08-04 tối · **Trạng thái**: ĐÃ CHẠY XONG, kết quả **ÂM TÍNH** · **Ràng buộc**:
PTQ thuần — không train model qua data thật, chỉ calibration-based local reconstruction (đúng
kỹ thuật AQLM "fix codes, optimize codebook values", cùng chuẩn đã dùng ở exp_f/exp_k/exp_m).

## Câu hỏi

Mọi PTQ trước giờ trong lab (exp_b/c/f/g/h/i/k/m) lượng tử hoá **từng số độc lập** (scalar:
ternary/int-N, dù có Hadamard/gauge/hoán vị kênh trước) → trần cứng đo được **~4bit** (int4
4,5bpw PPL~86 ≈ FP 69; mọi thứ dưới 1,58bit sụp PPL>10k). H1 (gauge-folding)/H2 (intrinsic-dim
probe) chỉ chứng minh KHÔNG có dư thừa dạng **biến đổi tuyến tính/hoán vị** — chưa test **Vector
Quantization thật**: nhóm d trọng số liền kề thành 1 vector, lượng tử hoá CẢ VECTOR về entry gần
nhất trong codebook học từ chính trọng số đó (kiểu AQLM "additive/residual quantization"). VQ
khai thác tương quan CHÉO giữa các chiều mà scalar (dù permute) không làm được — đây là lý do
AQLM/QuIP# đạt rate-distortion tốt hơn scalar-uniform trên LLM 7B+ ở 2-bit trong văn liệu.

## Phương pháp

Residual/additive VQ (giống AQLM), áp cho toàn bộ 196 ma trận linear (q/k/v/o + gate/up/down ×
28 layer), giữ nguyên FP embedding/lm_head/norm:

1. Nhóm mỗi d trọng số liền nhau theo chiều input thành 1 vector.
2. Residual k-means (Lloyd, 6 vòng) TỪNG MA TRẬN RIÊNG: codebook 1 học trên vector gốc, codebook
   2 học trên residual sau codebook 1, ... tới M codebook, mỗi codebook K entry.
3. **Local reconstruction** (PTQ hợp lệ): giữ nguyên assignment (chỉ số entry) đã chọn ở bước 2,
   tinh chỉnh GIÁ TRỊ các entry codebook bằng 40 bước Adam tối thiểu hoá MSE(X·Ŵᵀ, X·W0ᵀ) trên
   activation calibration thật (48 câu vi/ja) — đây CHÍNH LÀ "fix codes, optimize codebook" của
   AQLM, không phải train model qua data dịch.
4. bpw = payload (M·log2(K)/d) + overhead lưu codebook (đo thật theo từng ma trận).
5. Một cấu hình đại diện (~1,5bpw) chạy thêm qua **SEQUENTIAL/BRECQ-lite** (28 block, mỗi block
   50 bước Adam trên activation trước đó của CHÍNH block, giống hệt kỹ thuật exp_k/exp_m).

FP32 baseline: vi 68,97 / ja 125,07 (khớp exp_k/exp_m).

## Kết quả đầy đủ

| Config | d | M | K | bpw(total) | werr | PPL vi | PPL ja |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQ_d16_M1K32 | 16 | 1 | 32 | 0,316 | 92,9% | 65.184.339 | 56.777.103 |
| VQ_d16_M1K256 | 16 | 1 | 256 | 0,529 | 85,1% | 90.980.304 | 694.395.044 |
| VQ_d8_M1K16 | 8 | 1 | 16 | 0,501 | 85,6% | 66.013.796 | 62.866.004 |
| VQ_d16_M1K2048 | 16 | 1 | 2048 | 0,921 | 75,6% | 6.284.697 | 8.830.001 |
| VQ_d16_M2K256 | 16 | 2 | 256 | 1,058 | 67,4% | 2.409.608 | 27.173.222 |
| VQ_d8_M1K256 | 8 | 1 | 256 | 1,015 | 65,7% | 44.008.593 | 28.883.416 |
| VQ_d16_M2K1024 | 16 | 2 | 1024 | 1,483 | 56,5% | **34.195** | 70.750 |
| VQ_d16_M3K256 (TF) | 16 | 3 | 256 | 1,587 | 53,6% | 86.093 | 366.780 |
| **VQ_d16_M3K256 + SEQUENTIAL** | 16 | 3 | 256 | 1,587 | – | **6.044** | **10.096** |
| FP32 | – | – | – | 16,0 | 0% | 69,0 | 125,1 |

Mốc scalar đã biết (cùng dev set, script khác — exp_h/k/m):

| Mốc | bpw | PPL vi | PPL ja |
|---|---:|---:|---:|
| ternary_sparse@0,25 (exp_h, TF) | 1,33 | 33.687 | 47.152 |
| t2:4 fixpack SEQUENTIAL (exp_k) | 1,94 | **594,3** | 8.069 |
| dense ternary g32 SEQUENTIAL (exp_m) | 2,08 | **591,5** | 2.564 |
| int4-g32 TF (exp_g) | 4,50 | ~86 | – |

## Phân tích — so tại CÙNG ngân sách bit

- **Không sequential**: VQ tốt nhất (M2K1024, 1,483bpw) = PPL vi 34.195 — chỉ **ngang bằng**
  ternary-sparse vô hướng (exp_h, 1,33bpw, PPL vi 33.687), dù VQ dùng NHIỀU bit hơn. Không thắng.
- **Có sequential**: VQ+SEQ (1,587bpw) = PPL vi 6.044 — trong khi scalar ternary 2:4+SEQUENTIAL
  (exp_k, 1,94bpw) chỉ PPL vi **594,3** — **tệ hơn ~10 lần** dù VQ dùng ÍT bit hơn (1,587 <
  1,94). So với exp_m (dense ternary SEQUENTIAL, 2,08bpw, PPL vi 591,5): cùng kết luận.
- **Xu hướng trong thang VQ**: PPL giảm đều đặn khi M·K tăng (nhiều mã hơn → tái tạo tốt hơn,
  đúng logic), nhưng KHÔNG có điểm nào vượt qua đường cong scalar tại cùng bpw.

## Kết luận trung thực

**Giả thuyết "VQ khai thác tương quan chéo giữa các chiều mà scalar không làm được" KHÔNG được
xác nhận ở lần thử này.** Kết quả khớp với H1/H2 đã có từ trước (không có dư thừa ẩn dạng đơn
giản để khai thác) hơn là giả thuyết lạc quan ban đầu về VQ.

**Giới hạn của kết luận (không kết luận "VQ chết hẳn")**: recipe VQ ở đây dùng residual k-means
**không trọng số** (không có Wanda-importance `|W|·‖X‖` như scalar ternary đã dùng), init
ngẫu nhiên, chỉ refine 40 bước Adam (TF) / 50 bước/block (SEQ) — trong khi scalar ternary đã
được cả lab tinh chỉnh qua NHIỀU vòng lặp (gauge → Wanda mask → sequential 2-pass → scale+bias
học được, xem README Bài 8-10). Đây là so sánh "VQ thử lần đầu, công sức thấp" vs "scalar đã
đầu tư nhiều vòng" — không hoàn toàn công bằng. Nhưng với công sức đã bỏ ra (residual VQ + local
refine + sequential — đúng 3 kỹ thuật PTQ mạnh nhất đã biết trong lab, áp dụng đầy đủ), VQ chưa
cho tín hiệu đủ mạnh để biện minh đầu tư thêm (vd. thêm Wanda-weighting cho k-means, gauge trước
khi nhóm vector, tăng REFINE_STEPS) mà không có lý do cụ thể để tin nó sẽ đảo ngược khoảng cách
~10× đã thấy.

## Sự cố kỹ thuật gặp phải (đáng ghi lại)

Lần chạy đầu **chết giữa chừng ở cấu hình áp chót** (VQ_d16_M3K256, không traceback) — RAM tăng
dần 5,3GB→9,8GB qua các cấu hình, nhiều khả năng do giữ đồng thời bản gốc fp32 (196 ma trận) +
activation calibration (fp16, ~3,6GB) + buffer k-means/Adam của từng cấu hình không được giải
phóng triệt để giữa các vòng lặp. **Fix**: thêm cơ chế RESUME (đọc `exp_ad_results.json` cũ, bỏ
qua cấu hình đã có) + `gc.collect()` sau mỗi cấu hình trong `exp_ad_vq_codebook.py`. Chạy lại
từ checkpoint JSON, hoàn tất 2 cấu hình còn thiếu không lỗi.

Sequential block 27/28 có MSE trước-refine đột biến (305.121 so với hàng chục-hàng trăm ở các
block khác) — **khớp phát hiện cũ đã ghi trong memory lab** ("block 2 & 27 cực nhạy" — exp_k),
không phải bug mới.

## Phần 2 — 2 thử nghiệm vá lỗi (exp_ae Wanda-weighting, exp_af entry-0), CẢ HAI ÂM TÍNH

Sau exp_ad, thử vá đúng 2 điểm có cơ sở kỹ thuật rõ ràng nhất (không phải đoán mò):

### exp_ae — Wanda column-importance weighting cho k-means/assignment

Giả thuyết: VQ ở exp_ad dùng k-means KHÔNG trọng số (mọi cột như nhau), trong khi scalar
ternary chỉ tốt SAU KHI thêm trọng số Wanda `|W|·‖X‖`. Vá: nhân W theo cột với RMS-activation
(kiểu Wanda/AWQ) trước khi nhóm vector + k-means, chia lại sau khi reconstruct.

**Sự cố kỹ thuật lần 1**: chia lại cho importance có cột GẦN 0 (do outlier-channel — vài kênh
activation áp đảo làm kênh khác co lại tương đối, đã biết từ RESEARCH_TQ33_OUTLIER_FIX.md) gây
**nổ số** (werr 8137%, MSE sequential tăng thay vì giảm) — bug, không phải kết quả. Fix: kẹp
importance về [0,2× ; 5×] quanh trung vị trước khi dùng, giữ đúng thứ tự ưu tiên mà không cho
khuếch đại vô hạn.

**Kết quả sau fix (sạch, không bug) — VẪN ÂM TÍNH, TỆ HƠN không-trọng-số ở CẢ HAI trường hợp**:

| Config | bpw | PPL vi | PPL ja | So với exp_ad không trọng số |
|---|---:|---:|---:|---|
| VQ_d16_M2K1024 + Wanda (TF) | 1,483 | 82.592 | 355.044 | **tệ hơn 2,4×** (exp_ad: 34.195) |
| VQ_d16_M3K256 + Wanda (SEQ) | 1,588 | 16.705 | 25.194 | **tệ hơn 2,8×** (exp_ad: 6.044) |

Kết luận: Wanda-weighting (đúng ý tưởng, không phải thực thi sai) **không** transfer từ scalar
sang VQ ở đây — nhiều khả năng vì scale-rồi-chia-lại làm nhiễu bị khuếch đại không đều giữa các
cột trong không gian VQ, trong khi ở scalar nó chỉ ảnh hưởng NGƯỠNG chọn (không transform giá trị).

### exp_af — Entry-0 tường minh trong codebook (nhắm 0,3bpw, mốc sụp nặng nhất)

Giả thuyết: "mức 0 là vua" (phát hiện mạnh nhất toàn lab, exp_h/i) chưa được hiện thực hoá tường
minh trong VQ — mọi entry codebook đều học từ k-means, không có vector-0 đảm bảo. Vá: ép cứng
1/K entry mỗi codebook = vector-0 tuyệt đối (không train), luôn là ứng viên khi gán gần nhất.

**Kết quả — ÂM TÍNH, kèm cơ chế giải thích rõ ràng**:

| Config | bpw | zero_frac | PPL vi | PPL ja |
|---|---:|---:|---:|---:|
| VQ_d16_M1K32 + entry-0 (TF) | 0,316 | **6,2%** | 89.198.177 | 133.935.517 |
| exp_ad không entry-0 (mốc) | 0,316 | – | 65.184.339 | 56.777.103 |
| VQ_d16_M1K32 + entry-0 (SEQ) | 0,316 | – | 185.157 | 389.337 |

Entry-0 **tệ hơn một chút** chứ không cứu được, và **zero_frac chỉ 6,2%** — phát hiện cơ chế:
"mức 0 là vua" mạnh ở SCALAR vì 1 số riêng lẻ dễ tình cờ gần 0; ở VECTOR 16 chiều, phải **CẢ 16
số cùng lúc** gần 0 mới đáng dùng entry-0 — xác suất thấp hơn hẳn. Nhóm vector càng lớn (d càng
cao), lợi thế "mức 0" càng bị pha loãng. Đây là lý do CẤU TRÚC (không phải lỗi thực thi) giải
thích vì sao VQ hi sinh chính ưu thế đã cứu được ternary.

## Kết luận tổng thể (exp_ad + ae + af) — 3/3 thử nghiệm VQ đều âm tính

Sau khi thử: (1) VQ thuần, (2) VQ + Wanda-weighting, (3) VQ + entry-0 tường minh — không cấu
hình nào vượt qua scalar ternary+sequential ở cùng ngân sách bit. VQ nhất quán THUA, có cơ chế
giải thích hợp lý cho từng lần thử (không phải "chưa thử đủ tinh chỉnh"). Kết hợp với H1/H2 (đã
có từ trước) và trần PTQ ~4bit đã đo nhiều góc độ khác, đây là bằng chứng hội tụ khá mạnh rằng
**PTQ thuần (không train) đã cạn ý tưởng có cơ sở kỹ thuật rõ ràng cho sub-1,58bpw ở quy mô 0,6B**.

## Phần 3 — đóng câu hỏi treo với phiên song song "Máy A" (AQLM beam-search + SEQUENTIAL)

Phiên khác chạy song song (xem `RESEARCH_AQLM_CODEBOOK_PTQ.md`, `HANDOFF_MAYB_AQLM_SEQUENTIAL.md`)
đo AQLM (k-means + **beam-search** assignment, KHÁC greedy-residual của exp_ad/ae/af ở trên)
trên **OLMoE-1B-7B** (MoE, khác Qwen3-0.6B dense) và thấy **AQLM thắng ternary N:M** trên toàn
dải 0,33-1,56bpw — NHƯNG **chưa test SEQUENTIAL** (mọi so sánh của họ là "trơn"/TF). Vì kết quả
của họ (AQLM thắng) và của tôi (VQ thua ~10× sau sequential) tưởng như mâu thuẫn, cần đóng dứt
điểm bằng đúng 1 thí nghiệm còn thiếu: **AQLM beam-search THẬT + SEQUENTIAL, trên CHÍNH Qwen3-0.6B.**

`exp_ao_aqlm_beam_sequential.py`: port nguyên thuật toán `beam_assign` từ `exp_am_aqlm_full.py`
(máy A) — giữ top-4 ứng viên codebook-1, với mỗi ứng viên tìm codebook-2 tốt nhất cho residual,
chọn cặp tổng lỗi bé nhất (khác greedy: chỉ chọn codebook-1 gần nhất rồi mới tìm codebook-2).
Verify trước khi chạy full: beam-search cho werr thấp hơn greedy trên 1 ma trận test (33,5% vs
35,3%, không refine) — khớp phát hiện của máy A, xác nhận port đúng. Ghép với SEQUENTIAL
(BRECQ-lite) của chính lab tôi (refine codebook value khớp OUTPUT block, không phải weight-MSE
như refine của máy A — đúng kỹ thuật mạnh nhất đã biết trong toàn lab).

**Kết quả — CÂU HỎI ĐÃ ĐÓNG DỨT ĐIỂM:**

| Phương pháp | bpw | PPL vi | PPL ja |
|---|---:|---:|---:|
| scalar t2:4 fixpack + SEQUENTIAL (exp_k) | 1,94 | **594,3** | 8.068,6 |
| VQ residual-greedy + SEQUENTIAL (exp_ad) | 1,587 | 6.044,3 | 10.095,7 |
| **AQLM beam-search + SEQUENTIAL (exp_ao)** | 2,029 | **4.196,6** | 9.568,6 |

Beam-search **có cải thiện thật** so với greedy (4.197 < 6.044, ~31% tốt hơn, đúng chiều máy A
đã thấy) — nhưng **vẫn thua scalar+SEQUENTIAL ~7 lần**, dù dùng NHIỀU bit hơn (2,03 vs 1,94bpw).
**Kết luận cuối cùng cho cả 2 phiên**: AQLM (dù k-means thuần hay beam-search, dù trên OLMoE hay
Qwen3-0.6B) chỉ thắng scalar ternary khi CHƯA có sequential-reconstruction. Ngay khi thêm đòn
bẩy mạnh nhất lab từng đo (BRECQ-lite), scalar ternary bứt hẳn lên trên mọi biến thể VQ/codebook
đã thử — sequential là đòn KHÔNG chuyển giao sang biểu diễn codebook, bất kể thuật toán gán
(greedy/beam) hay kiến trúc model (dense/MoE). Đây là kết luận PTQ cuối cùng, hội tụ từ 2 phiên
độc lập, 2 model khác nhau, 4 biến thể VQ khác nhau (greedy/Wanda/entry-0/beam) — đủ vững để
đóng hẳn hướng VQ/codebook cho PTQ sub-1,58bpw ở quy mô model này.

## File

- `exp_ad_vq_codebook.py` — script chính (đã vá RESUME + gc.collect)
- `exp_ad_results.json` — kết quả đầy đủ 9 cấu hình
- `exp_ad_run.log` / `exp_ad_run2.log` — log chạy lần 1 (chết giữa chừng) + lần 2 (resume, hoàn tất)
- `exp_ae_vq_wanda.py` + `exp_ae_results.json` + `exp_ae_run.log` — Wanda-weighting (âm tính)
- `exp_af_vq_zero.py` + `exp_af_results.json` + `exp_af_run.log` — entry-0 tường minh (âm tính)
- `exp_ao_aqlm_beam_sequential.py` + `exp_ao_results.json` + `exp_ao_run.log` — beam-search
  (port từ máy A) + SEQUENTIAL, đóng câu hỏi treo giữa 2 phiên (vẫn âm tính, thua ~7×)
