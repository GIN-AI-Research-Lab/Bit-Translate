# HƯỚNG 2 (PTQ-ONLY): codebook đa-bộ kiểu AQLM/QTIP — hướng PTQ duy nhất chưa thử-đủ

**Ngày**: 2026-08-04 · **Trạng thái**: thiết kế, CHƯA code, CHƯA toy-verify · **Ràng buộc**:
PTQ-only — xem §0 "ranh giới PTQ vs train" để biết chính xác cái này có vi phạm ràng buộc
không (câu trả lời: không, nhưng cần đọc kỹ vì có sắc thái).
**Tiền đề**: README.md Bài 0–10 (đo cạn hướng scalar-quantize+mask), bảng "Con đường thật"
cuối README (dòng *"QTIP/AQLM + Hadamard | 2–2.5 bit | không (train) | máy B | SOTA PTQ;
~3 bit dùng được, 2 bit rủi ro ở 0.6B"* — lab TỰ liệt kê hướng này, CHƯA từng code/chạy).
Tra cứu hôm nay (web search, không suy đoán): AQLM ([arXiv:2401.06118](https://arxiv.org/abs/2401.06118),
giải thích kỹ ở [Towards Data Science](https://towardsdatascience.com/the-aqlm-quantization-algorithm-explained-8cf33e4a783e/)),
QTIP ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235), [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip)).

---

## 0. Ranh giới PTQ vs train — đọc TRƯỚC khi bác bỏ hướng này vì "có gradient"

AQLM dùng **gradient-based codebook refinement** (residual k-means init → beam search gán
mã → tinh chỉnh codebook bằng gradient). Điều này CÓ backprop, nhưng khác bản chất với
QAT/KD mà user vừa loại:

| | S1 sequential (lab đang coi là PTQ, dùng xuyên suốt Bài 8–19) | AQLM codebook refine | QAT/KD (exp_ab, vừa dừng) |
|---|---|---|---|
| Tối ưu cái gì | Wfp+scale+bias MỖI TENSOR, khớp activation FP cùng layer | Codebook (M×2^B vector nhỏ, VÀI NGÀN số) + gán mã, khớp W gốc | TOÀN BỘ Wfp của TOÀN model, khớp logit teacher |
| Data cần | Vài chục câu calib (đã có) | Vài trăm-vài nghìn ĐOẠN calib (không cần nhãn/dev-loss) | Corpus train thật (train_slice.vi/ja) + dev set + nhiều nghìn bước |
| Mục tiêu hàm loss | Tái tạo W hoặc activation cục bộ | Tái tạo W (giống S1, KHÔNG nhìn activation downstream) | Hành vi sinh văn bản (KD loss trên logits) |
| Lab gọi là gì | PTQ ("reconstruction, không train toàn model") | → tương đương S1, PTQ theo đúng định nghĩa lab đang dùng | Train |

**Kết luận**: AQLM nằm ĐÚNG cùng hạng với "S1 sequential" mà lab đã coi là PTQ suốt từ Bài 8.
Không vi phạm ràng buộc "không train" của user — miễn là dùng ĐÚNG scope: tối ưu codebook
bằng calib data offline, KHÔNG chạm vào corpus train/dev-loss/nhiều-nghìn-bước. Nếu user vẫn
muốn tuyệt đối zero-gradient thì dùng biến thể "AQLM chỉ k-means, bỏ bước gradient-refine"
(rẻ hơn, chất lượng kém hơn — cờ tùy chọn `--no-grad-refine`, đo A/B).

## 1. Vì sao đây là hướng PTQ CÒN LẠI, không phải lặp lại việc đã âm tính

Toàn bộ Bài 0–10 dùng **scalar hoặc ternary quantization theo nhóm** (mỗi trọng số quantize
riêng, chỉ chia sẻ 1 scale/nhóm 32-64). Bài 2 (`exp_c_incoherence.py`) đã thử **Hadamard
incoherence THẬT** (xoay trọng số, không phải fake sign-flip) — có tác dụng (−2.2% đến −2.8%
sai số) nhưng chỉ đẩy ngưỡng dùng-được từ ~4 bit xuống ~3 bit, KHÔNG chạm 1.58. Kết luận cũ:
*"SOTA của nó là 2-bit trên model 7B+, không phải 0.6B"* — nhưng lab MỚI DÙNG NỬA kỹ thuật
(rotation), CHƯA GHÉP với phần còn lại của QuIP#/QTIP: **vector/trellis quantization sau khi
xoay**. AQLM còn không cần rotation — nó thắng bằng cơ chế khác hẳn: **mã hóa CHUNG nhiều
trọng số bằng 1 index** (additive multi-codebook) thay vì mỗi trọng số 1 giá trị độc lập.

Đây là khác biệt CHẤT, không phải mức độ: scalar quantization (mọi thứ lab đã thử) không thể
khai thác TƯƠNG QUAN giữa các trọng số trong cùng nhóm; codebook/VQ khai thác được (một
"pattern" hay gặp trong nhóm 8 trọng số được lưu 1 lần trong codebook, dùng lại nhiều nơi).
Đây chính xác là lý do R-D của VQ tốt hơn scalar về lý thuyết (đã ghi nhận ở bảng ý G của
`RESEARCH_NOVEL_SUBBIT_PLAN.md`, nhưng bị loại ở đó vì so với hướng TRAIN — ở đây ta so với
PTQ scalar, phép so sánh khác).

## 2. Thiết kế: AQLM trước (rẻ, không cần trellis kernel), QTIP là dự phòng nếu cần hơn

### 2.1 AQLM — cấu hình đề xuất cho vòng thăm dò

- Nhóm g=8 trọng số liên tiếp (theo hàng, đúng chiều input — giống cách chia group-64 hiện
  tại nhưng nhỏ hơn để phù hợp beam search chi phí hợp lý).
- M=2 codebook, mỗi codebook 2^8=256 codeword (8-bit index/codebook) → **2 byte/group-8 =
  2 bit/trọng số** (khớp đúng dải "2-2.5 bit" lab tự ước lượng). Có thể thử M=1 (1 bit/trọng
  số, cực rẻ nhưng chất lượng chắc chắn kém, dùng làm điểm neo dưới) và M=3 (3 bit, điểm neo
  trên gần vùng "3 bit dùng được" lab đã dự đoán).
- Khởi tạo: residual k-means trên trọng số THẬT (không cần activation) — CHẠY ĐƯỢC CPU, rẻ.
- Tinh chỉnh: beam search gán mã (rẻ, không gradient) → tùy chọn gradient-refine codebook
  (vài trăm bước Adam trên MSE tái tạo, KHÔNG phải KD loss — xem §0) — so A/B có/không bước
  này để đo đóng góp thật, tách bạch theo đúng văn hóa ablation của lab.
- Calibration: dùng LẠI bộ calib mixwj đã có (`exp_r_qat_lite.CAL_*` + dev vi/ja) — không
  cần data mới.

### 2.2 Áp cho lớp nào

Giống mọi thí nghiệm PTQ trước (Bài 5 mixed-precision): **FFN gate/up/down** trước (60%
linear, "chỗ ternary chịu tốt nhất" theo README) — đây cũng là nơi VQ có nhiều tương quan
giữa trọng số nhất (theo văn liệu AQLM, FFN nén tốt hơn attention). Attention/o_proj giữ
scalar ternary hoặc int4 như hiện tại (Bài 2 đã chỉ ra đây là nơi nhạy — "mạnh nhất ở
down_proj/o_proj" cho Hadamard, ngược lại nghĩa o_proj CẦN Hadamard/cẩn trọng hơn, không
phải nơi để thử nghiệm mới đầu tiên).

### 2.3 QTIP — CHỈ nếu AQLM cho thấy tín hiệu tốt nhưng chưa đủ

QTIP = incoherence processing (Hadamard — **lab ĐÃ CÓ code**, `exp_c_incoherence.py`, chỉ
cần verify nó xoay đúng cả 2 phía weight+activation runtime, không chỉ weight) + trellis-coded
quantization (TCQ, decoder có trạng thái, cần viết MỚI — không có sẵn trong lab). TCQ tốt
hơn VQ về lý thuyết (chi phí tuyến tính theo chiều thay vì mũ) nhưng phần kernel decode phức
tạp hơn AQLM nhiều (không chỉ là tra bảng — cần hàm decode trellis "bit-shift" như paper mô
tả). **KHÔNG code TCQ ở vòng này** — chỉ dự phòng nếu AQLM đạt tín hiệu tốt và cần bit thấp
hơn nữa mà AQLM riêng không đạt.

## 3. Thước đo & tiêu chí PASS/FAIL (ghi trước, theo đúng kỷ luật lab)

So sánh trực tiếp với mốc PTQ tốt nhất lab đã có ở CÙNG hoặc GẦN bit-budget (Bài 8/9):

| bpw AQLM | Mốc scalar-PTQ gần nhất để so | Tiêu chí PASS |
|---:|---|---|
| ~2.0 (M=2,g=8) | ternary dense fix-pack+SEQUENTIAL 1.94bpw → PPL vi **594** (kỷ lục PTQ scalar) | AQLM PPL vi < 594 ở bpw TƯƠNG ĐƯƠNG hoặc thấp hơn → PTQ vượt được trần scalar |
| ~1.0 (M=1,g=8) | exp_l thang sub-1-bit: mọi cấu hình <1.9bpw đều PPL >60k trên 0.6B | AQLM PPL vi < 10.000 đã là thắng lớn (chưa cần "dùng được", chỉ cần chứng minh VQ phá được trần scalar) |
| ~3.0 (M=3,g=8) | int4-g32 (không train) → PPL vi 86, "gần như miễn phí" | Không kỳ vọng thắng int4 (khác hạng bit) — mốc này chỉ để vẽ đường cong bpw-vs-PPL đầy đủ |

**Kill criteria (dừng, không đầu tư QTIP)**: nếu AQLM ở ~2bpw không đánh bại PPL 594 (mốc
scalar-PTQ hiện tại) → VQ KHÔNG có lợi thế thật trên model 0.6B này (có thể vì model quá nhỏ,
không đủ tương quan nhóm để codebook khai thác — đúng "định luật kích thước" Bài 6/7/9 lặp
lại lần nữa ở dạng khác) → đóng hướng này, viết phát hiện âm tính vào README (Bài 20), quay
về chấp nhận trần 4-bit/PTQ-scalar-1.94bpw làm giới hạn thực dụng.

**Mở rộng nếu PASS**: đo geo6 (6 miền, tái dùng `eval6()`/probes trong `exp_r_qat_lite.py`)
không chỉ PPL vi/ja đơn lẻ — tránh lặp lại bài học "eval chỉ vi/ja = thiên vị" (Bài 11).

## 3b. KẾT QUẢ THẬT — Bước 0 và Bước 1 (04/08, testbed đổi sang OLMoE-1B-7B, không phải 0.6B)

> Đổi testbed theo yêu cầu user: dùng MoE nhẹ nhất có thể để thử nghiệm nén, KHÔNG dùng
> 30B-A3B/Laguna 118B. Chọn **OLMoE-1B-7B** (Allen AI, 7B tổng/1B active, 64 expert/layer,
> expert FFN chỉ 1024×2048) — MoE mở nhỏ nhất còn được duy trì tốt. Tải tensor qua HTTP
> range-read (đọc header safetensors + range-read đúng tensor cần, KHÔNG tải cả shard/model —
> kỹ thuật giống Kimi-K3-in-C "sampled tensors over HTTP range reads").

**Bước 0** (`exp_ad_aqlm_toy.py`, 4 tensor riêng, k-means thuần không refine): AQLM
**2,062bpw → err 0,325** vs ternary-Lloyd tốt nhất **2,585bpw → err 0,384**. AQLM thắng ở bit
THẤP HƠN. → PASS, đi Bước 1.

**Bước 1** (`exp_ae_aqlm_full.py`, pool 16 expert cùng 1 bộ codebook chung, 33,5M trọng số
thật, ablation 4 bước):

| Cấu hình | err | bpw |
|---|---:|---:|
| (a) k-means-only | 0,3323 | 2,0625 |
| **(b) +beam-search** | **0,3195 ← tốt nhất trong lần chạy này** | 2,0625 |
| (c) +gradient-refine (lr=1e-2) | 0,3250 (TỆ HƠN b) | 2,0625 |
| (d) +1 vòng alternating | 0,3219 (phục hồi 1 phần, vẫn kém b) | 2,0625 |
| ternary-Lloyd g8 | 0,3834 | 2,585 |
| ternary-Lloyd g32 | 0,4262 | 1,835 |

**Đọc kết quả trung thực**: AQLM (mọi biến thể, cả bản kém nhất) thắng ternary-Lloyd. Nhưng
gradient-refine — bước mà paper AQLM gốc dùng để nâng chất lượng — lại làm TỆ HƠN beam-search
đơn thuần trong lần chạy này; nhiều khả năng lr=1e-2 chưa tinh (đúng bẫy Bài 4: "LR cao đầu
tiên tệ hơn cả PTQ ngây thơ"), CHƯA đủ để kết luận refine vô dụng — cần thử lr thấp hơn
(1e-3) trước khi đóng câu hỏi này. Công thức thực dụng tạm dùng: **k-means + beam-search,
bỏ refine** cho tới khi tinh lại được.

## 4. Quét bit-budget 0,3 → 1,56bpw + TRỘN kỹ thuật (chỉ đạo mới của user 04/08 — chạy chuẩn từ nay)

> User: *"nghiên cứu thử nghiệm ưu tiên cho 0.3 bpw tới 1.56bpw. luôn chạy theo hướng từ nhỏ
> tới lớn này lên các ý tưởng hoặc trộn lẫn các ý tưởng để làm sao đạt được mức bit này nhưng
> vẫn giữ được chất lượng model."* Ghi thành quy tắc chuẩn (xem memory `feedback-ptq-only-
> bpw-sweep`): MỌI thí nghiệm PTQ từ nay quét đủ **5 mốc bpw chuẩn của lab** (đúng ladder
> `CANON` trong `exp_ab_subbit_curriculum.py`, đúng các mốc README Bài 9/10/13/15 đã dùng),
> đi từ mốc THẤP NHẤT (khó nhất) lên, và ở MỖI mốc thử KẾT HỢP kỹ thuật, không chỉ 1 kỹ
> thuật đơn lẻ:

| bpw mục tiêu | Cấu hình AQLM gần nhất (M,K,g) | Kỹ thuật cộng thêm để thử TRỘN (theo thứ tự ưu tiên) |
|---:|---|---|
| **0,331** | M=1,K=32,g=16 (0,3125) | Hadamard incoherence TRƯỚC k-means (Bài 2, `exp_c_incoherence.py` — có sẵn) + highway columns (giữ FP vài kênh outlier, đo không đoán, giống `detect_highway` trong exp_ab) |
| **0,474** | M=1,K=128,g=16 (0,4375) | + sequential block-wise (BRECQ-lite — đòn LỚN NHẤT lab từng đo, ×10 ở Bài 8) áp SAU khi codebook đã fit, dùng activation calib thật |
| **0,699** | M=2,K=16,g=8 (1,0) hoặc M=1,K=64,g=8 (0,75) | + Wanda-style mask (ưu tiên nhóm quan trọng dùng codebook mịn hơn, nhóm còn lại thô hơn — mixed-precision Ở CẤP CODEBOOK, chưa ai thử) |
| **1,023** | M=2,K=16,g=8 (1,0) | Như 0,699 + đo riêng đóng góp mỗi kỹ thuật (ablation, không gộp mù) |
| **1,566** | M=2,K=64,g=8 (1,5) | Đích = khớp khung TQ33 (2:4 ternary+scale) — nếu AQLM thắng ở đây, cân nhắc khả năng đóng gói thay thế 2:4 ternary trong runner TQ33 hiện có |

**Nguyên tắc trộn (tránh gộp mù — đúng văn hóa ablation Bài 10 "quy công/tội")**: mỗi mốc bpw,
LUÔN đo riêng từng kỹ thuật cộng thêm (giống bảng Bước 1 ở trên: (a)→(b)→(c)→(d)) trước khi
báo "kết hợp X+Y tốt" — nếu không tách được đóng góp, không biết cái nào thật sự có công.

**Kill/pivot per mốc**: nếu ở 1 mốc bpw, dù trộn hết mọi kỹ thuật khả dụng mà vẫn KHÔNG thắng
ternary-Lloyd cùng bpw (hoặc thắng nhưng err vẫn ở vùng "chết" theo chuẩn Bài 9: so sánh
tương đối, chưa có PPL/chất lượng sinh thật ở bước này) → ghi nhận mốc đó "chưa tìm được tổ
hợp thắng", đi tiếp mốc CAO hơn, không cố đấm ăn xôi ở 1 mốc.

### KẾT QUẢ THẬT — quét 5 mốc (04/08, `exp_af_bpw_sweep.py` + fix baseline `exp_ah_nm_ternary_fix.py`)

⚠️ Lần chạy đầu (`exp_af`) so AQLM với ternary DENSE — SAI, vì ternary dense có sàn cứng
log2(3)=1,585bpw, không biểu diễn được 4/5 mốc mục tiêu (đều <1,585bpw), nên code tự rơi về
group=64 (~1,71bpw) ở MỌI mốc — so sánh không công bằng (ternary luôn dùng nhiều bit hơn).
**Đã sửa bằng ternary N:M SPARSE đúng CANON** (`exp_ah`) — bảng đúng:

| bpw mục tiêu | AQLM (err @ bpw thật) | ternary N:M CANON (err @ bpw thật) | Kết quả |
|---:|---|---|---|
| 0,331 | 0,8572 @ 0,3125 | 0,9062 @ 0,331 (mask 1:32) | **AQLM thắng** |
| 0,474 | 0,7938 @ 0,4375 | 0,8505 @ 0,474 (mask 1:16) | **AQLM thắng** |
| 0,699 | 0,6695 @ 0,750 | 0,7710 @ 0,698 (mask 1:8) | **AQLM thắng** |
| 1,023 | 0,5905 @ 1,000 | 0,6682 @ 1,021 (mask 1:4) | **AQLM thắng** |
| 1,566 | 0,4344 @ 1,500 | 0,5045 @ 1,564 (mask 2:4) | **AQLM thắng** |

**AQLM (thuần k-means+beam, KHÔNG calib/activation) thắng ternary N:M sparse-Lloyd ở CẢ 5
mốc**, cùng hoặc ít bit hơn. Đây là kết quả sạch, đáng tin — công thức AQLM hiện tại (không
cần calib) đã là cải tiến thật so với scalar N:M sparse trên toàn dải 0,3-1,56bpw.

### Thử "cứu" mốc 0,3bpw bằng Wanda-calib (04/08, `exp_ag_wanda_calib.py`) — KẾT QUẢ ÂM, nghi bug

Theo yêu cầu user ("cách cứu với calib để đạt 0.3bit"): tải GGUF Q4_K_M OLMoE (~4GB, đồng ý
của user), dựng `OlmoeForCausalLM` 1-layer thật bằng `transformers`, forward pass calib text
EN thật qua embedding+attention+router, tính Wanda importance `|W|·mean(|activation|)` cho
16 expert đang test, so với mask thuần `|W|` ở mốc 1:32 (0,33bpw).

**Kết quả lần 1: 0/16 expert cải thiện** — Wanda-calib đều TỆ HƠN |W|-thuần. Nghi CÓ BUG:
công thức Wanda gốc dùng **L2-norm `‖X‖₂`**, tôi lại dùng **mean(|activation|)** — sai công
thức tổng hợp.

**Đã sửa (`exp_ai_wanda_aqlm.py`, 04/08): dùng đúng L2-norm + tăng calib 20→30 đoạn (token/
expert tăng ~43→~139 trung bình) — CHẠY LẠI, KẾT QUẢ VẪN ÂM, dứt điểm: 0/16 expert, Wanda-L2
thua magnitude-thuần ở MỌI expert (vd expert có nhiều calib nhất — 910 token — vẫn thua rõ
0,939 vs 0,901).** Đây KHÔNG còn là bug — là phát hiện thật, có cơ chế hợp lý: Wanda được
thiết kế cho mức thưa VỪA PHẢI (~50%), không phải mức CỰC ĐOAN 1/32 (giữ 3% trọng số) đang
thử ở đây. Ở mức cực đoan này, việc PHẢI ép trọng số được giữ về ±scale (ternary) khiến sai
số làm-tròn tuyệt đối trở thành yếu tố chi phối hơn hẳn "chọn đúng kênh quan trọng theo
activation" — magnitude thô (ưu tiên giữ trọng số TO, vốn ít bị méo tương đối khi ép về
±scale) thắng vì đo đúng cái THẬT SỰ quyết định sai số ở regime này.

**Gate tự động dừng đúng thiết kế** (`n_better_gate1 < 50%` → không làm Bước 2 weighted-AQLM,
tránh lãng phí thời gian đầu tư kỹ thuật vào importance-weighting khi tín hiệu cơ bản đã âm).

**Kết luận cho câu hỏi "cứu 0,3bpw bằng calib"**: KHÔNG, ít nhất không theo cách Wanda-style
activation-importance ở mức 1/32. Không đóng hẳn cửa calib nói chung — chỉ đóng cửa NHÁNH
CỤ THỂ này (Wanda cho mask N:M cực đoan). Hướng calib còn chưa thử: calib có thể vẫn có ích
ở mức ÍT cực đoan hơn (1:8, 1:4 — nơi Wanda gốc được thiết kế/validate), hoặc dùng calib theo
cách khác hẳn (vd. GPTQ-style: điều chỉnh trọng số CÒN LẠI để bù sai số do trọng số bị bỏ,
không phải chọn trọng số nào bỏ).

### Mở rộng: đưa calib vào CHÍNH AQLM (không chỉ mask ternary) — cũng ÂM, đã bác bỏ giả thuyết riêng

User hỏi đúng: Gate 1 chặn Bước 2 dựa trên suy luận từ ternary-mask, nhưng AQLM là cơ chế
KHÁC (biểu diễn liên tục qua codeword, không ép 0 tuyệt đối) — giả thuyết hợp lý là Wanda có
thể vẫn có ích cho AQLM dù không có ích cho ternary. **Đã bỏ gate, chạy trực tiếp
(`exp_ai_wanda_aqlm.py` Bước 2, weighted k-means đúng công thức per-point — xem chứng minh
centroid trong code): KẾT QUẢ CŨNG 0/16 — AQLM+Wanda thua AQLM-thường ở MỌI expert** (vd
0,856 → 0,864-0,879, luôn tệ hơn). **Giả thuyết "AQLM khác ternary nên có thể ăn calib" bị
bác bỏ bằng số đo, không phải suy luận** — cùng cơ chế thất bại (làm lệch hướng "quan trọng"
khỏi thứ THẬT SỰ quyết định sai số ở mức bit cực thấp: độ lớn thô của trọng số) áp dụng cho
cả hai cách biểu diễn.

**Kết luận cuối cùng, dứt điểm**: ở mức 0,3bpw trên MoE nhỏ (OLMoE), calibration (Wanda-style,
cả cho chọn mask VÀ cho chính AQLM) không giúp — pure-magnitude/pure-weight vẫn là chuẩn tốt
nhất đã tìm được ở mức bit này. Không thử thêm biến thể Wanda nào nữa ở 0,3bpw trừ khi có ý
tưởng THẬT SỰ khác cơ chế (không phải chỉnh tham số/công thức tổng hợp của cùng ý tưởng).

### Kiểm tra tổng quát hóa (04/08, `exp_aj_generalization.py`) — kết quả chính đã VỮNG

Mọi kết quả AQLM-thắng-ternary phía trên chỉ đo trên **16/64 expert, CHỈ layer 0** — mở rộng
lên **TOÀN BỘ 64 expert × 2 layer (0 và 8, đại diện đầu/giữa model)** ở 2 mốc biên
(0,331 và 1,566bpw):

| Layer | bpw | AQLM (64 expert) | ternary N:M (64 expert) | Kết quả |
|---|---:|---:|---:|---|
| 0 | 0,331 | 0,8575 | 0,9063 | **AQLM thắng** |
| 0 | 1,566 | 0,4335 | 0,5040 | **AQLM thắng** |
| 8 | 0,331 | 0,8575 | 0,9073 | **AQLM thắng** |
| 8 | 1,566 | 0,4320 | 0,5046 | **AQLM thắng** |

**4/4 thắng, số liệu gần như giống hệt giữa layer 0 và layer 8** (0,8575 cả hai layer ở mốc
thấp; 0,4335 vs 0,4320 ở mốc cao) — phát hiện chính KHÔNG phải may mắn của mẫu 16 expert hay
đặc thù layer đầu; vững qua quy mô đầy đủ và qua độ sâu model. Đây là điểm dừng hợp lý để coi
"AQLM (không calib) thắng ternary N:M sparse trên toàn dải 0,3-1,56bpw" là kết luận CHÍNH đủ
tin cậy của lần thăm dò này.

## 4b. Thăm dò rẻ TRƯỚC khi code đầy đủ (đúng thứ tự "toy trước, tốn tiền sau" của lab)

1. **Bước 0 ($0, ~30 phút, numpy CPU)**: chạy AQLM (k-means thuần, KHÔNG gradient-refine) trên
   ĐÚNG 1 tensor `down_proj` của 1 layer (tensor đã dùng xuyên suốt Bài 3/4 để so sánh xuyên
   các thí nghiệm) ở g=8/M=2. So sai số tái tạo ||Ŵ−W|| với ternary-Lloyd cùng bit-budget đo
   trong Bài 0 (bảng rate-distortion). Nếu AQLM KHÔNG thắng ternary-Lloyd ngay ở mức tái tạo
   trọng số đơn giản nhất (chưa cần sequential/reconstruction gì) → tín hiệu xấu sớm, cân
   nhắc dừng trước khi code toàn bộ pipeline.
2. **Bước 1 ($0, vài giờ CPU)**: nếu bước 0 tích cực, code AQLM đầy đủ (k-means+beam+gradient-
   refine tùy chọn) áp cho TOÀN BỘ FFN của 1-2 layer, đo PPL cục bộ kiểu Bài 4 (layer-wise,
   không cần sequential đầy đủ ngay).
3. **Bước 2**: nếu bước 1 vẫn tích cực, ghép với sequential block-wise (BRECQ-lite, đã có
   `s1_sequential`-style code trong `exp_r_qat_lite.py`/`exp_k_fixpack_sequential.py`) áp
   toàn model, đo theo bảng §3.

Không nhảy thẳng bước 3 — đúng bài học "bẫy: bản LR cao đầu tiên tệ hơn PTQ ngây thơ" (Bài 4)
và toàn bộ triết lý "kỳ vọng ghi trước + cổng hủy giữa chừng" của lab.

## 5. Rủi ro & giới hạn

1. **0.6B là "ca khó nhất"** (định luật kích thước, lặp lại xuyên suốt Bài 6/7/9/17) — AQLM/
   QTIP có SOTA công bố trên 7B+; hoàn toàn có thể KHÔNG chuyển giao xuống 0.6B, đúng dự đoán
   thận trọng sẵn có trong README ("2 bit rủi ro ở 0.6B").
2. **Beam search + k-means chi phí compute cao hơn scalar quantize nhiều** — cần đo thời gian
   thật ở bước 0 trước khi ước lượng có áp được toàn model 0.6B (196 tensor) trong thời gian
   hợp lý trên CPU máy A hay không (không có GPU-kernel AQLM sẵn, có thể cần viết hoặc dùng
   thư viện tham khảo — cần khảo sát thêm nếu bước 0 tích cực).
3. **Kernel suy luận CHƯA THIẾT KẾ**: nếu AQLM thắng về chất lượng, vẫn cần kernel decode
   multi-codebook cho runner C (khác hẳn LUT ternary 33-pattern của TQ33) — việc kế, không
   nằm trong scope thăm dò chất lượng này.
4. **Không lặp lại sai lầm Bài 14 (B1a/B1a2)**: nếu ghép AQLM với sequential/absorber, đích
   optimize phải là `Wfp_gốc − Q` theo đúng chiều đã học được (không mài về `orig−Q`, không
   đọc trực tiếp ô "residual" chưa qua `quant()` — xem lý do kỹ thuật đầy đủ ở Bài 14).

## 6. Bản đồ file dự kiến

| File | Vai trò | Trạng thái |
|---|---|---|
| `exp_ad_aqlm_toy.py` (mới) | Bước 0: AQLM k-means thuần trên 1 tensor, so ternary-Lloyd | chưa code |
| `exp_ae_aqlm_full.py` (mới) | Bước 1-2: AQLM đầy đủ (+gradient-refine tùy chọn) + sequential | chưa code |

Không đụng file PTQ hiện có (`exp_h/i/k/l/...`) — hướng này độc lập, so sánh KẾT QUẢ với
chúng, không sửa chúng.
