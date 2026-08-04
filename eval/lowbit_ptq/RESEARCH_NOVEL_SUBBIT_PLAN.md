# KẾ HOẠCH NGHIÊN CỨU MỚI: Rollout-KD × Bit-Curriculum cho dải 0.33→1.56 bpw

**Ngày**: 2026-08-04 · **Trạng thái**: thiết kế xong + toy-verify xong ($0, máy B) + code
Modal SẴN SÀNG (chưa chạy — 4 ví đều cạn, cần user nạp & xác nhận) · **Người đề xuất**: phiên
Claude 04/08 chiều · **Tiền đề**: toàn bộ README Bài 0–19, RESEARCH_TQ33_OUTLIER_FIX.md,
exp_r/exp_w/exp_v, memory lab.

> Đây là đề xuất THĂM DÒ. Mọi con số kỳ vọng đều ghi trước và có cổng hủy (kill criteria).
> Không hứa đạt "Q4-quality @0.3bpw" — mục tiêu PoC được định nghĩa đo được ở §5.

---

## 0. Tóm tắt 10 dòng

1. **Chẩn đoán mới**: "lời nguyền dưới 2 bit" không nằm ở trọng số (PTQ đã vét — Bài 0–10,
   H1/H2 âm) mà ở **HÀM MỤC TIÊU**: mọi KD của lab là teacher-forced; student chưa từng được
   train trên trạng thái do CHÍNH NÓ sinh — nơi nhiễu lượng tử đẩy nó vào attractor lặp.
   PPL teacher-forced mù với lỗi này (Bài 12/15 đã thấy: "sinh-mở đòi hỏi cao hơn PPL nhiều").
2. **Chọn hướng**: **E. Rollout-KD (on-policy/DAgger)** làm chủ lực + **D̃. bit-curriculum
   qua họ mask N:M LỒNG NHAU** 1:32⊂1:16⊂1:8⊂1:4⊂2:4 (0.331→1.566bpw — đúng dải user muốn,
   đích cuối đúng khung TQ33/F0) làm trục phụ; **C-lite highway** là cờ tùy chọn.
3. Đã verify $0 hôm nay: cơ chế exposure-bias TỒN TẠI và rollout-KD thu hẹp nó (toy G3:
   gap 0.535→0.377, +15.8đ); masked-STE là BẮT BUỘC cho curriculum (G2 trên tensor thật:
   full-STE gây shock ×10.8 khi grow); và một phát hiện phản-trực-giác từ probe model thật:
   **teacher FP "đồng lõa" với loop trong-ngữ-cảnh** → rollout-KD phải CẮT loss tại cửa-vào-loop.
4. Kế hoạch Modal: P0 probe $1 (trên ckpt GEN4 có sẵn) → 4 arm 2×2 (~$28) với cổng hủy
   sau từng bước. Đường lean $15. Code sẵn: `exp_ab_subbit_curriculum.py` + `modal_qat_subbit.py`.

---

## 1. Tổng hợp bằng chứng: lời nguyền thực chất là gì

Xếp lại toàn bộ facts của lab thành 3 tầng:

| Tầng | Sự thật đã chứng minh | Hệ quả |
|---|---|---|
| Thông tin (trọng số) | PTQ trần ~4bit; ternary 43% werr; H1/H2: KHÔNG còn dư thừa ẩn; QAT-STE −60% output err; F0: phải train-trong-khung | Muốn <2bit phải TRAIN — đã làm (GEN4/exp_w) |
| Định dạng | TQ33 1.5bpw lossless + kernel AVX2 19.3GB/s; free-lunch format đã vét (gauge/perm/mask) | Khung đích đã có, KHÔNG cần thiết kế lại |
| **Hành vi** | 0.6B@1.56 KD xong PPL ×4.6 vẫn **loop**; 30B S1 sampling "đúng ngữ pháp cục bộ, vô nghĩa toàn cục"; outlier-fix đúng toán vẫn 0/3 domain; routing lật ở gần-hòa | **← nút thắt CHƯA CÓ vũ khí. Đây là chỗ đánh.** |

Điểm mấu chốt chưa ai khai thác: **mọi loss từng dùng (KL teacher-forced, CE, MSE
reconstruction) đều đo trên ngữ cảnh VÀNG**, trong khi hiện tượng chết đo trên ngữ cảnh
**TỰ SINH**. Hai phân bố trạng thái này khác nhau, và khoảng cách đó TĂNG theo độ dài sinh
(compounding). Đây là exposure bias kinh điển (DAgger/imitation learning), bị nhiễu lượng
tử hóa khuếch đại: mỗi bước sai một chút → ngữ cảnh trượt khỏi manifold → model (vốn chỉ
được ép khớp teacher TRÊN manifold) sai nhiều hơn → rơi vào attractor tự-nhất-quán (loop).

**Bằng chứng mới đo hôm nay ($0, máy B — `exp_ac_divergence_probe.py`, 16 prompt × 48 tok):**

| Arm | KL(T‖S) self-context theo bucket vị trí | KL teacher-forced | distinct-2 | Hành vi |
|---|---|---|---|---|
| FP (sanity) | 0 / 0 / 0 / 0 | 0 | 0.86 | mạch lạc |
| int4-g32 (control tốt) | 0.30 / 0.23 / 0.26 / 0.20 | 0.25–0.28 | 0.88 | self ≈ TF, PHẲNG ✓ |
| ternary 2:4 PTQ thô (vùng chết) | **7.4** / 5.1 / 4.6 / 4.4 | 12.4–13.8 | **0.57** | loop rác "瓢瓢瓢…" |

Ba bài học từ bảng này (định hình thiết kế):
1. **Chế độ khỏe** (int4): self-KL ≈ TF-KL, phẳng — không compounding. Chế độ chết: self-KL
   cao nhất ở **CỬA VÀO** (bucket 0-8) rồi *giảm* — vì một khi ngữ cảnh đã thành loop,
   **chính teacher FP cũng dự đoán tiếp loop** (induction head). Teacher "đồng lõa" trong-ngữ-cảnh.
2. Hệ quả thiết kế: tín hiệu học của rollout-KD nằm ở **vùng cửa-vào-loop** (KL lớn = gradient
   lớn); phần đuôi đã-loop cho gradient ≈ 0 và phí compute → **loss phải cắt sau điểm-vào-loop**
   (đã cài: `loop_entry()`/`--roll-loopcut` trong exp_ab).
3. Thước "self/TF ratio + shape" phân biệt được 3 chế độ (khỏe / đang trượt / đã sụp) — dùng
   làm cổng P0 và metric xuyên suốt.

---

## 2. Phản biện hạt giống A–D + ý mới

| Ý | Cơ sở từ bằng chứng lab | Vấn đề/lỗ hổng | Phán quyết |
|---|---|---|---|
| **A. Router margin loss** | Mạnh nhất về bằng chứng (finding routing 30B: 96.4% lệch ở 3 hạng thấp nhất; fix có lợi vẫn lật ở layer sâu) | KHÔNG test được trên 0.6B (dense, không MoE) → không PoC rẻ được; chỉ dùng ở pha 30B | **GIỮ cho pha 30B** — toy G5 đã verify loss đúng chiều (flip −28%, MSE sạch +7.5%, instability −11%) |
| **B. Slimmable/multi-bit đồng thời** | Khớp chữ "curriculum ngược" | Văn liệu slimmable cho thấy mức THẤP NHẤT thường *kém hơn* train riêng, không hơn; nhân chi phí mỗi bước ×số mức; mục tiêu lab là MỘT model 1.5bpw chứ không phải họ model | **LOẠI** (ghi lại lý do, không đốt tiền) |
| **C. Outlier highway bake-in** | Kênh massive CỐ ĐỊNH có thật (45/48 layer @30B); vá hậu kỳ đã âm tính → thử bake-in từ đầu là bước logic kế | Trên 0.6B phải ĐO lại kênh (30B là kênh 0, 0.6B báo cáo ~48/52 — không suy chéo); tác dụng có thể nhỏ vì weight-side chỉ là 1 phần | **CỜ TÙY CHỌN** trong exp_ab (`--highway`, tự đo kênh, +~0.05bpw) — không phải trục chính |
| **D. N:M ramp dần** | Khớp triết lý grow-depth đã thắng ở dự án dịch; gradual-pruning là văn liệu vững | Bản gốc D là "xiết dần" (dense→sparse); user muốn NGƯỢC (0.3→1.56). Cả hai đều có nguy cơ: mask đổi giữa chừng = vi phạm luật "mask refresh ×2.2" của lab | **CHỌN dạng sửa D̃**: mask họ LỒNG NHAU cố định từ đầu (tournament), chuyển stage chỉ THÊM ô — grow không refresh. Curriculum 0.33→1.56 đúng ý user, mỗi nấc trùng thang bpw lab đã có mốc (0.62/0.70/1.02/1.56) → đọc được giá trị curriculum trực tiếp |
| **E. Rollout-KD (on-policy/DAgger) — ý mới** | Khớp CHÍNH XÁC chữ ký lỗi (PPL ổn, sinh sụp; loop); exp_w đã có sẵn máy KL top-K+lseT để tái dụng | Teacher đồng lõa với loop trong-ngữ-cảnh (đo hôm nay) → cần loop-cut; chi phí sinh rollout ×~1.7/bước; nếu lỗi thực chất là capacity thì vô ích → cần P0 làm cổng | **CHỌN LÀM CHỦ LỰC** |
| **F. Divergence-curve như thước đo hạng nhất — ý mới** | Đã validate trên 3 chế độ (bảng §1) | Chỉ là thước, không phải phương pháp | **DÙNG** làm cổng P0 + metric mọi arm |
| **G. VQ/codebook (AQLM-style) thay ternary** | R-D của VQ > scalar về lý thuyết | F0 nói QAT dịch chuyển trọng số cho vừa khung → lợi R-D của codebook giảm mạnh khi đã train-trong-khung; phá toàn bộ đầu tư kernel TQ33; đổi cả stack | **LOẠI cho vòng này** (ghi nhận là hướng dài hạn nếu E/D̃ đều thất bại) |

**Vì sao E×D̃ là tổ hợp đáng tiền nhất**: E đánh vào tầng HÀNH VI (nút thắt thật, chưa có vũ
khí), D̃ đánh vào ĐƯỜNG ĐI tối ưu (rẻ, an toàn nhờ nested, tự cho ra 5 mốc frontier so sánh
được với GEN4). Hai trục trực giao → thí nghiệm 2×2 tách bạch công/tội từng trục — đúng văn
hóa ablation của lab (bảng quy công/tội Bài 10).

---

## 3. Thiết kế kỹ thuật (đã code xong, đã toy-verify)

### 3.1 Quantizer: họ mask lồng nhau + masked-STE (file `exp_ab_subbit_curriculum.py`)

- **Thang**: `1:32 (0.331bpw) → 1:16 (0.474) → 1:8 (0.699) → 1:4 (1.023) → 2:4 (1.566)`
  (kế toán đúng công thức exp_r: payload×log2(3) + entropy mask + scale f8/g64; khớp mốc lab
  từng đo — verify G1). Đích cuối = **đúng gia phả TQ33**: 2:4 ternary + scale f8-grid g64
  (+bias — runner cần thêm 1 phép cộng/hàng nếu đóng gói, xem §7 việc kế).
- **Mask cố định TOÀN BỘ từ phút 0** bằng tournament trên Wanda importance (|W|·xnorm):
  top-2/4 → top-1/4 → winner từng cặp nhóm khi M nhân đôi. Bảo đảm cấu trúc
  mask(k) ⊆ mask(k+1); chuyển stage CHỈ THÊM ô. → tôn trọng tuyệt đối luật lab
  "mask refresh = thủ phạm ×2.2" (Bài 10) trong khi vẫn có curriculum.
- **Masked-STE** (grad = 0 trên ô pruned): toy G2 đo trên tensor THẬT Qwen3-0.6B
  (layers.2.down_proj): full-STE để ô pruned random-walk |Δ|=0.22 (LỚN HƠN ô kept 0.13 —
  tái lập cơ chế B1a2 "ô pruned là nhiễu" của Bài 14 trong bối cảnh curriculum), làm grow-shock
  ×10.8 và **85% ô mới bật ngay ±1 (nhiễu thuần)**; masked-STE: drift đúng 0, shock ×4.7,
  chỉ 0.8% ô mới bật → **soft-start tự nhiên** (ô mới nằm im ở 0 tới khi gradient thật đẩy
  vượt s/2). Đây là mảnh ghép khiến curriculum an toàn.
- **Scale**: per-(row,g64), STE trên lưới f8 (đúng exp_r); refit đóng-form (Lloyd 3 vòng)
  tại MỖI lần chuyển stage, ĐÓNG BĂNG trong stage (đúng recipe GEN4/EfficientQAT: chỉ train
  W+bias, lr 2e-5).
- **Highway (cờ)**: đo absmax/kênh hidden ở input q/k/v/gate/up qua calib; kênh > 32× median
  ở ≥60% layer → cột đó giữ FP (loại khỏi mask/quant, grad thật), +16bit×k/C ≈ 0.05bpw @k=6.
  KHÔNG hardcode kênh (bài học "phải đo lại, không suy 30B→0.6B").

### 3.2 Loss: TF-KD + λ·Rollout-KD với loop-cut

```
L = (1−λ)·[KL_fullvocab(T‖S | ngữ cảnh vàng, temp 2) + 0.1·CE] + λ·KL_topK+lseT(T‖S | ngữ cảnh TỰ SINH)
```
- Nhánh TF: nguyên văn công thức exp_r v4 (fix batchmean, temp 2, CE 0.1) — không đổi gì.
- Nhánh rollout: mỗi `--roll-refresh 4` bước, student TỰ SINH 64 token (sampling t=1.0,
  top-p 0.95) từ pool 256 prompt (đoạn 24-token từ KD-mix); teacher chấm **top-64 + lseT
  full-vocab** (nguyên máy exp_w v2 — "KL đúng cả khối đuôi"); loss chỉ tính vùng tự sinh,
  **CẮT sau điểm-vào-loop + 2 token** (`loop_entry`: cùng 4-gram xuất hiện 3 cửa sổ liền kề;
  lý do ở §1).
  Buffer dùng lại 4 bước (semi-on-policy kiểu GKD) → chi phí ~×1.7/bước thay vì ×3.
- λ: 0 trong warmup, ramp tuyến tính lên 0.5 trong 30% đầu stage; **chỉ bật khi bpw stage
  ≥ 0.6** (`--roll-min-bpw`) — student 0.33bpw sinh toàn rác, DAgger trên rác thuần là phí
  (ưu tiên TF dựng khung trước).
- Chọn best theo geo6 (giữ so sánh được với GEN4); battery chạy ở best VÀ cuối mỗi stage
  (nếu rollout thắng về hành vi nhưng thua nhẹ geo6, số liệu vẫn hiện ra — không bị nuốt).

### 3.3 Toy-verify đã chạy (4/4 PASS — `exp_aa_subbit_toys.py`, kết quả trong `exp_aa_results.json`)

| Gate | Kiểm cái gì | Kết quả then chốt |
|---|---|---|
| G1 | Họ mask lồng nhau + bpw + masked-STE + highway đúng toán | 5/5 stage nested ✓, bpw khớp mốc lab ±0.005 ✓, grad ô pruned = 0 ✓ |
| G2 | Transition-shock trên tensor thật | full-STE: drift ô pruned 0.22, shock ×10.8, 85% ô mới = nhiễu ±1; masked: 0 / ×4.7 / 0.8% |
| G3 | Cơ chế rollout-KD end-to-end (LM tí hon ternary 1:8, task copy-chain trên manifold 8-pattern, oracle chấm mọi prefix) | TF-KD: TF-acc 1.000 nhưng tự sinh 0.465 (curve sụp 0.87→0.32 — compounding); rollout-KD: 0.623 (**+15.8đ**, gap −30%, bucket cuối +56%) không mất TF-acc |
| G5 | Router margin (cho pha 30B) | flip-rate dưới nhiễu −28% (0.083→0.059), MSE sạch +7.5%, output-instability −11% |

Trung thực về G3: đã phải thiết kế lại 2 lần — (i) task ngoặc: off-manifold quá hẹp (1 chiều
depth) → rollout vô dụng; (ii) copy prompt uniform-random: trạng thái rollout CÙNG PHÂN BỐ
với trạng thái vàng → DAgger không có gì để dạy. Chỉ khi manifold vàng CÓ CẤU TRÚC (8 pattern)
thì exposure gap xuất hiện và rollout-KD ăn. **Hai lần hỏng này chính là điều kiện biên của
phương pháp**: rollout-KD chỉ ăn khi (a) phân bố trạng thái tự sinh ≠ phân bố train, và
(b) lỗi không phải thuần capacity. LLM thật thỏa (a) rõ ràng (text có cấu trúc, lỗi phá cấu
trúc); (b) là câu hỏi mở → chính là cái P0 phải đo trước khi tiêu tiền.

### 3.4 Thước đo & tiêu chí thành công (PRE-REGISTERED — không đổi sau khi thấy số)

Mỗi arm đo: (1) PPL 6 miền + geo6 + val-100 vi (nối tiếp GEN4); (2) **behavior battery**
22 prompt × validator tự động (số học exact / fact contains / code CHẠY THẬT so kết quả /
format), greedy VÀ sampling t=0.7, kèm **loop-rate** (distinct-2 + 4-gram lặp 3 lần);
(3) **divergence curve** self vs TF (exp_ac).

| Câu hỏi | Tiêu chí PASS | Nếu FAIL |
|---|---|---|
| P0: cơ chế có ở model QAT thật? | ckpt gen4_n4 (1.56bpw): loop-rate@t0.7 ≥ 30% VÀ self-KL bucket 0-8 ≥ 2× TF-KL cùng bucket (chữ ký attractor/cửa-vào) | Nếu battery gen4 đã khá + không loop → nút thắt không phải coherence@0.6B → dừng chuỗi 0.6B, cân nhắc thẳng 30B |
| E ăn không? (roll vs base, cùng 4000 bước, cùng 1.56bpw) | pass-rate battery +≥8đ tuyệt đối HOẶC loop-rate −≥30% tương đối; geo6 không tệ hơn >10% | E chết trên LLM thật (dù toy pass) → báo cáo âm tính, giữ lại loop-cut/divergence như công cụ đo; cân nhắc pivot G (codebook) |
| D̃ ăn không? (curr vs base, cùng TỔNG 4000 bước) | geo6 HOẶC battery tốt hơn base ở đích 1.56bpw; các mốc giữa (0.70/1.02) ≥ GEN4.1 cùng bpw | D̃ âm tính → đóng nhánh curriculum, chỉ giữ E |
| Mục tiêu mở rộng (stretch) | battery pass-rate @1.56bpw ≥ 80% pass-rate của FP teacher trên cùng battery | Không đạt cũng KHÔNG coi là fail toàn cục (xem §6 giới hạn capacity) |
| Kill giữa chừng | stage 1:32 sau 400 bước KD mà val-100 vi > 3000 → bỏ stage, vào thẳng 1:16 (ghi log) | — |

---

## 4. Kết quả đã có hôm nay ($0) — tóm tắt

1. `exp_aa_subbit_toys.py` — 4/4 gate PASS (bảng §3.3). Thiết kế masked-STE + nested mask
   + loop-cut + router-margin đều được kiểm bằng số trước khi tốn tiền.
2. `exp_ac_divergence_probe.py` — chạy thật trên Qwen3-0.6B (bảng §1): instrument hoạt động,
   3 chế độ tách bạch, và phát hiện "teacher đồng lõa với loop" (định hình lại loss).
3. `exp_ab_subbit_curriculum.py` — trainer đầy đủ (S1 compact 2-pass port từ exp_r + curriculum
   + rollout + battery + divergence), smoke CPU trên máy B.
4. `cloud/modal_qat_subbit.py` — app Modal 5 hàm (probe_p0 / arm / arms_all idempotent /
   gate_ab / status), đúng quy ước modal_qat_lite (volume `qat-lite-vol`, `--detach`,
   chạy chuỗi TRONG container — bài học Bài 13).

## 5. Kế hoạch chạy Modal (CHỜ user nạp ví + xác nhận — tuyệt đối không tự chạy)

| Bước | Lệnh (profile trituekstns) | Giá ước* | Cổng quyết định |
|---|---|---|---|
| P0 | `modal run --detach cloud/modal_qat_subbit.py::probe_p0` | ~$1 / 25ph | bảng §3.4 dòng 1 |
| A1 base | `...::arm --tag ab_base` | ~$5 / 2.5h | mốc |
| A2 roll | `...::arm --tag ab_roll` | ~$8-9 / 4h | E ăn? → quyết A4 |
| A3 curr | `...::arm --tag ab_curr` | ~$5-6 / 3h | D̃ ăn? |
| A4 cr | `...::arm --tag ab_cr` | ~$9-10 / 4.5h | gộp có cộng hưởng? |
| (hoặc trọn gói) | `...::arms_all` (idempotent theo tag, preempt-safe) | ~$28±8 | — |
| Gate bổ sung | `...::gate_ab --ckpt ab_cr.pt` | ~$0.7 | — |

\* Neo giá từ lịch sử thật: GEN4 = 5000 bước L40S ≈ $6/bậc (Bài 15) ⇒ ~$1.2/1k bước TF-KD;
arm rollout ×~1.7 (đo lại hệ số thật ở A2 — nếu >2.2 thì giãn `--roll-refresh` 4→8).
Điều kiện: ckpt `qat_gen4_n4.pt`/`qat_gen4_o1.pt` còn trên volume `qat-lite-vol` (Bài 15 ghi
là còn, "chế độ 4G chưa tải về") — P0 tự bỏ qua nếu thiếu.

**Kiểm novelty (đã làm, 04/08 — web search):**
- Cơ chế "KD trên chuỗi student tự sinh" ĐÃ TỒN TẠI: **GKD** (Agarwal et al. 2023,
  [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)) — nêu đúng vấn đề train-inference
  distribution mismatch, đã vào cả [TRL GKDTrainer](https://huggingface.co/docs/trl/gkd_trainer).
  → Tin tốt: phương pháp đã được chứng minh ổn định ở quy mô thật (giảm rủi ro kỹ thuật);
  tin ràng buộc: KHÔNG được tuyên bố "on-policy KD" là phát minh của lab.
- **LLM-QAT** dùng data do TEACHER sinh (data-free) — KHÁC on-policy (không sửa exposure bias
  của student).
- Prior gần nhất: "What Makes Low-Bit QAT Work for Reasoning LLMs? A Systematic Study"
  ([arXiv:2601.14888](https://arxiv.org/abs/2601.14888), 01/2026) — KD là objective robust
  cho QAT thấp-bit, PTQ-init tốt, có kết quả 2-bit trên chính Qwen3-0.6B (MATH-500). Nên đọc
  kỹ TRƯỚC khi chạy arms (miễn phí, có thể chỉnh recipe).
- CHƯA thấy prior nào ghép: on-policy-KD × sub-2bpw × chữa loop/coherence × loop-cut do
  teacher-collusion (phát hiện đo thật của lab, §1) × nested-mask curriculum trong khung
  ternary N:M tự chế. Mảnh mới của lab nằm ở TỔ HỢP + 2 cơ chế loop-cut/nested-masked-STE.

## 6. Rủi ro & giới hạn (trung thực)

1. **Capacity wall 0.6B**: BitNet-style ternary chỉ ~ngang FP từ cỡ ~3B trở lên; 0.6B là ca
   khó nhất lab tự chọn. Khả năng thật: rollout-KD giảm loop rõ (hành vi "đọc được") nhưng
   pass-rate battery vẫn xa FP/Q4. Khi đó giá trị của vòng này = CƠ CHẾ đã chứng minh + công
   thức mang lên 30B (nơi capacity đủ — LoRA-KD 30B đã "sinh đúng ngữ pháp+fact" ở greedy).
2. **Teacher-collusion**: loop-cut xử lý phần đuôi, nhưng nếu attractor của student KHÔNG
   phải loop 4-gram (vd. lặp cấu trúc câu dài, chuỗi số) detector bỏ sót → loss loãng.
   Việc kế nếu thấy: detector entropy-based (distinct-n cửa sổ trượt).
3. **Curriculum có thể vô ích hoặc hại**: prior văn liệu cho "reverse curriculum" yếu (DSD
   là analogy gần nhất, nhưng DSD là dense→sparse→dense). Nested design làm nó RẺ và AN TOÀN
   để thử, không làm nó ĐÚNG. Đó là lý do nó là trục phụ có ablation riêng (A3), không trộn
   vào chủ lực.
4. **Rollout distribution shift theo stage**: student đổi chất lượng qua stage → phân bố
   rollout đổi → so sánh giữa stage không thuần túy. Chấp nhận (mục tiêu là model cuối).
5. **1 seed, battery 22 prompt**: đủ cho hiệu ứng lớn (±10đ), không đủ cho hiệu ứng nhỏ.
   Nếu A2−A1 < 8đ: coi là "chưa kết luận", không phải "âm tính" — cần thêm seed (thêm $).
6. **Bias trong khung TQ33**: ckpt exp_ab có bias (như exp_r). Đóng gói TQ33 cần runner cộng
   bias/hàng (~vài dòng C, chưa làm). KHÔNG ép hậu kỳ bỏ bias (F0: mất ×4000).
7. **Số liệu toy ≠ LLM**: G2/G3/G5 chứng minh cơ chế và bắt lỗi thiết kế, không dự báo độ lớn
   hiệu ứng trên 0.6B. Độ lớn thật chỉ có sau A1/A2.

## 7. Pha 2 — 30B-A3B (thiết kế, CHƯA code, chờ kết quả 0.6B)

Nếu E thắng trên 0.6B (cổng §3.4): ghép vào chuỗi 30B hiện có `exp_v (S1) → exp_w (LoRA-KD)`:
1. **Rollout-KD cho exp_w**: exp_w hiện cache teacher-logits TRƯỚC (top-K+lseT) nên không
   on-policy được. Sửa: 2 GPU trong 1 container (`gpu="A100-80GB:2"` — student+grad ở GPU0,
   teacher bf16 61GB ở GPU1), rollout 48-64 token, cùng loop-cut. Ước $25-40/2k bước.
   (Phương án rẻ hơn nếu thiếu VRAM: teacher re-cache theo chu kỳ trên rollout mới — mất
   on-policy một phần, thử sau.)
2. **Router margin loss (hạt giống A)**: thêm vào loss exp_w, train CẢ trọng số router
   (12.6M params — nhỏ): `L_margin = E[relu(m − (z_(k) − z_(k+1)))]` với m = 0.3·std(z),
   trọng số 0.03 (đúng công thức đã verify G5: flip −28% không hại chất lượng sạch). Kỳ vọng
   ghi trước: giảm tỉ lệ lệch-routing-vs-oracle (thước có sẵn: `check_routing_vs_oracle.py`)
   và giảm khuếch đại rel-err layer sâu trong RESEARCH_TQ33_OUTLIER_FIX.
3. Đích đóng gói: TQ33 6.88GB hiện có + bias-add trong runner.

## 8. Những gì KHÔNG làm và vì sao (để người sau khỏi lặp)

- **Slimmable multi-bit đồng thời (B)**: xem bảng §2 — chi phí ×số mức, prior văn liệu ngược
  chiều mục tiêu "1 model tốt nhất ở mức thấp".
- **VQ/codebook (AQLM/QTIP-style)**: đổi cả stack format+kernel; lợi R-D bị F0/QAT ăn mòn;
  chỉ mở lại nếu E và D̃ đều âm tính.
- **Outlier-fix hậu kỳ thêm**: đã âm tính có hệ thống (RESEARCH_TQ33_OUTLIER_FIX) — mọi vá
  activation lúc suy luận KHÔNG sửa được thiếu-KD. Highway ở đây là bake-in lúc TRAIN, khác
  về bản chất, và cũng chỉ là cờ phụ.
- **Train thẳng 0.3bpw làm ĐÍCH DEPLOY**: frontier lab (GEN4.1: 0.62bpw geo6 613 sau full KD)
  cho thấy 0.3-0.5bpw trên 0.6B còn cách "dùng được" rất xa; 0.33bpw trong kế hoạch này là
  ĐIỂM XUẤT PHÁT curriculum, không phải đích.

## 9. Bản đồ file

| File | Vai trò | Trạng thái |
|---|---|---|
| `eval/lowbit_ptq/exp_aa_subbit_toys.py` (+`exp_aa_results.json`) | 4 toy gate thiết kế | ĐÃ CHẠY, 4/4 PASS |
| `eval/lowbit_ptq/exp_ac_divergence_probe.py` (+`exp_ac_results.json`) | probe exposure-bias model thật; dùng lại cho P0 Modal | ĐÃ CHẠY local (fp/int4/ptq24) |
| `eval/lowbit_ptq/exp_ab_subbit_curriculum.py` | trainer chính (curriculum × rollout × highway + battery) | Code xong, smoke máy B |
| `cloud/modal_qat_subbit.py` | app Modal (P0 + 4 arm + gate), idempotent, --detach | SẴN SÀNG, CHƯA CHẠY |
| File này | kế hoạch + tiêu chí pre-registered | — |

Không sửa/xóa bất kỳ file/ckpt nào có sẵn; exp_ab chỉ IMPORT từ exp_r.
