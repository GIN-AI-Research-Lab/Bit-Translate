# PLAN KD JA→VI — Distill thầy Haiku để vượt Google chiều ja→vi (lập 2026-07-20)

> Thay thế lộ trình data-niche của PLAN_RANKUP §2-3 cho mục tiêu ja→vi.
> User đã chốt: **ja→vi là sản phẩm chính**, vi→ja hạ xuống mức duy trì.

## 0. Vì sao pivot — chẩn đoán sau 3 vòng data + scale

Số liệu gate step30000 (hardbench 4-way, 10 giám khảo):

1. **Data vá niche không tăng ròng.** Vòng 3a (idiom/slang/ctx, 5000 step LR-restart 1.2e-4)
   chỉ ĐẢO CHỖ điểm ja→vi: thanhngu +0.62, zeropronoun +0.58 NHƯNG keigo −0.78,
   nguphap −0.72, hop −0.55. TB ja→vi đứng yên: 2.40 (19100) → 2.37 (30000).
2. **Gap lớn nhất với Google là năng lực lõi, không phải niche:** nguphap −2.0,
   caudai −1.95, hop −1.9. Niche data không chữa được cái này.
3. **Corpus gốc đã vắt kiệt:** ~3.9 tỉ token train trên mix ~1 tỉ token (đa phần
   OpenSubtitles/CCMatrix nhiễu, register phụ đề) = nhiều epoch trên cùng nguồn nhiễu.

**Sự thật chiến lược chưa khai thác:** trên chính hardbench của ta, **Haiku thắng Google
CẢ 10/10 domain ja→vi** (và thắng theo câu 128/55/17). Ta có API truy cập một hệ dịch
ja→vi TỐT HƠN Google với giá $1/$5 per MTok:

| domain ja→vi | 292M@30k | Google | Haiku |
|---|---|---|---|
| caudai      | 1.70 | 3.65 | 4.75 |
| hoithoai    | 2.20 | 3.70 | 4.95 |
| hop         | 1.75 | 3.65 | 4.40 |
| it_deep     | 2.60 | 4.10 | 4.65 |
| keigo       | 1.95 | 2.80 | 3.75 |
| nguphap     | 1.95 | 3.95 | 4.35 |
| slang       | 1.90 | 3.05 | 3.95 |
| solieu      | 3.40 | 4.50 | 4.70 |
| thanhngu    | 2.85 | 3.60 | 4.15 |
| zeropronoun | 3.35 | 3.75 | 4.40 |

⇒ Bài toán đổi từ "gom data mà Google không có" thành **sequence-level knowledge
distillation** (Kim & Rush 2016 — con đường đã kiểm chứng của mọi model dịch nhỏ tốt:
distilled NLLB, Opus-MT): thầy dịch cả corpus nguồn, trò train trên output thầy thay vì
reference nhiễu. Trò giữ 85-95% chất lượng thầy trên domain hẹp; 85% × 4.41 ≈ 3.75 >
Google 3.67. Đây là lộ trình SỐ HỌC đầu tiên tới "vượt Google toàn tuyến ja→vi".
KD đồng thời chữa bệnh #3: model được train trên VI sạch nhất quán thay vì VI phụ đề.

Lưu ý ToS: dùng output Haiku train model dịch chuyên hẹp (không cạnh tranh LLM) — đã
duyệt hướng Haiku-làm-thầy ở PLAN_RANKUP; CLAUDE.md dặn kiểm tra điều khoản — giữ nguyên.

## 1. Ba đợt

| Đợt | Việc | Chi phí | Kết quả |
|---|---|---|---|
| **0** | Ổn định LR + checkpoint averaging + chẩn đoán trần (chi tiết §2) | ~$5 GPU, 1-2 ngày | Chọn checkpoint nền + kết luận nhiễu-LR vs trần-sức-chứa |
| **1** | Xây corpus KD: Haiku dịch ja→vi 300k câu JA nhắm domain yếu (Batch API) | ~$60-150 | Corpus (JA thật, VI-Haiku) đã lọc LaBSE |
| **2** | Train mix nghiêng ja→vi 70-75%, LR thấp decay dài; đo harness cũ | ~$10-20 GPU | Mục tiêu sóng 1: lật zeropronoun/thanhngu/keigo; TB ja→vi 2.37 → 3.2-3.5 |

Song song 0 param (app-side, PLAN_RANKUP §8.4 — quyết định "sản phẩm thắng Google"):
placeholder số/mã + quy đổi 万/億 (xóa gần hết gap solieu −1.1), tách câu tại 。！？.

**KHÔNG làm nữa:** sinh idiom/slang kiểu vòng 3a; BT phục vụ vi→ja (841k cặp đã sinh:
cất release `bt-vong3b`, KHÔNG trộn vào mix — nguồn JA của nó tái dùng cho KD Đợt 1);
bàn scale >292M (chỉ mở lại nếu KD cũng bão hòa — lúc đó mới có bằng chứng trần thật).

## 2. ĐỢT 0 — chi tiết (đang chạy)

**Mục tiêu:** trả lời 1 câu hỏi — vụ 5 domain tụt ở step30000 là (A) nhiễu LR-restart
chưa hội tụ lại, hay (B) trần cạnh tranh sức chứa thật ở 292M? Đồng thời chọn checkpoint
nền tốt nhất cho Đợt 2. KHÔNG data mới, không tiền lớn.

### W0.1 — Train +2500 step LR êm (Modal L40S, ~2h, ~$5)

- Resume last.pt@30000 (data = premix vòng 3a, GIỮ NGUYÊN). Chạy được trên **tài khoản
  Modal mới** — `::setup` tải mọi thứ từ GitHub release, không phụ thuộc Volume cũ.
- LR: warmup 50 step → đỉnh **6e-5** → cosine về 3e-5 tại 32500 (TB ~4.5e-5 — bằng nửa
  đỉnh 1.2e-4 của Phase 1, không có cú sốc restart). Milestone mỗi **500 step** → 5 bản
  30500…32500 phục vụ averaging.
- Autosave lên release `autosave-scale300m` với tên file RIÊNG `w0_aa/w0_ab` — không đè
  bộ `p_aa/p_ab` step30000 lành.

```bash
# tài khoản Modal mới — chuẩn bị 1 lần:
python3 -m pip install modal && python3 -m modal setup
TK=$(gh auth token); python3 -m modal secret create github-token GH_TOKEN=$TK
python3 -m modal run cloud/modal_train_wave0.py::setup       # nạp Volume ~4GB từ release
# chạy:
python3 -m modal run --detach cloud/modal_train_wave0.py::train
python3 -m modal run cloud/modal_train_wave0.py::status      # xem tiến độ
```

### W0.2 — Checkpoint averaging (0 đồng)

Trung bình 5 milestone 30500…32500 (`scripts/avg_checkpoints.py`) — mẹo NMT chuẩn,
thường +0.2-0.5 chrF. Chạy ngay trên Modal sau khi train xong + upload cả 2 ứng viên
lên release **`wave0-lowlr`** (mỗi file ~1.2GB, model-only):

```bash
python3 -m modal run cloud/modal_train_wave0.py::publish
# -> release wave0-lowlr: avg_wave0.pt + step32500.pt
```

### W0.3 — Eval local (máy có bitnet.cpp, ~1h)

Cho TỪNG ứng viên {avg_wave0, step32500} (lệnh mẫu cho avg):

```bash
gh release download wave0-lowlr -R trituenguyen97/Bit-Translate -p avg_wave0.pt -D checkpoints/
python scripts/convert_to_gguf.py --ckpt checkpoints/avg_wave0.pt --out dist/w0avg_f32.gguf
llama-quantize dist/w0avg_f32.gguf dist/w0avg_i2s.gguf I2_S 1        # số 1 BẮT BUỘC
python eval/run_hardbench.py dist/w0avg_i2s.gguf 292m_w0avg
python eval/eval_one.py dist/w0avg_i2s.gguf 292m_w0avg ja2vi 100
python eval/eval_one.py dist/w0avg_i2s.gguf 292m_w0avg vi2ja 100
python eval/run_probe64.py dist/w0avg_i2s.gguf 292m_w0avg
```

Đọc sớm bằng chrF theo domain (so `eval/hardbench_292m_step30000.jsonl`) để CHỌN bản
tốt hơn đưa đi judge — nhưng KHÔNG kết luận bằng chrF (bài học 16/07: chrF đánh giá
thấp gap thật).

### W0.4 — Judge 4-way CHỈ cho bản thắng (~1h)

```bash
python eval/build_judge_panels_4way.py eval/hardbench_haiku.jsonl \
       eval/judge_4way_w0 eval/hardbench_292m_w0avg.jsonl 292M
# chấm 10-15 giám khảo Claude như protocol cũ ->
python eval/aggregate_judge_4way.py eval/judge_4way_w0/judges eval/judge_4way_w0
```

### W0.5 — KẾT QUẢ THẬT (2026-07-20 tối) — kết luận C, chưa lường trước

Không chạy được `::publish` (cần Modal, user chưa auth ở máy phụ) nên chỉ eval được
checkpoint step32500 THÔ (chưa average) — tải trực tiếp `w0_aa/w0_ab` trên release
`autosave-scale300m` (bản backup cuối của hàm `::train`, đủ dùng). Kết quả judge 10
giám khảo (protocol y hệt step30000) so bằng "hệ tĩnh" Google/Haiku/Fable (bản dịch
CỐ ĐỊNH, chỉ đổi giám khảo giữa 2 lượt) làm mốc sàn nhiễu:

| Hệ | acc ja→vi step30000 | acc ja→vi wave0 | Δ |
|---|---|---|---|
| Google (tĩnh) | 3.67 | 3.56 | −0.11 (nhiễu) |
| Haiku (tĩnh) | 4.41 | 4.27 | −0.14 (nhiễu) |
| Fable (tĩnh) | 4.88 | 4.97 | +0.09 (nhiễu) |
| **292M** | **2.37** | **1.84** | **−0.53** |

Sàn nhiễu thật ±0.1-0.15; 292M lệch −0.53 = gấp 3-4 lần → **suy giảm THẬT, không phải
nhiễu giám khảo**. Tụt ĐỀU cả 10/10 domain ja→vi (it_deep −1.15, thanhngu −1.15,
zeropronoun −0.75...) — khác hẳn kiểu "đảo chỗ" của vòng Phase1 25000→30000. Loss
train giảm (~1.0-1.1, thấp hơn mọi mốc trước) trong khi judge held-out tụt = chữ ký
overfit. **Kết luận C (chưa có trong bảng dự kiến ban đầu): train thêm trên CÙNG data
vòng3a (dù LR êm, không restart sốc) vẫn có hại — corpus đã vắt kiệt từ vòng train
trước, không còn tín hiệu mới để học đúng.**

**Quyết định: dùng `step30000` làm nền cho Đợt 2, KHÔNG dùng wave0/step32500/avg.**
Không cần chạy `::publish` (averaging) nữa — 5 milestone đều nằm trên cùng quỹ đạo
overfit, average không cứu được sai lệch lớn cỡ này. Bài học áp dụng ngược cho Đợt 2:
đo judge ở các mốc trung gian trong lúc train KD, đừng chạy cố định số step rồi mới
kiểm tra — corpus KD nhỏ hơn nhiều (13-19k câu) nên đến điểm "vắt kiệt" nhanh hơn.

## 3. ĐỢT 1 — corpus KD

### 3.1 Tuyển thầy — XONG (2026-07-20), kết quả vượt kỳ vọng

Tận dụng key free sẵn có (Gemini ×2, DashScope, Zhipu) thay vì trả tiền Haiku ngay:
chạy hardbench200 qua 7 ứng viên (`eval/run_hardbench_api.py`,
`scripts/run_teacher_bench.sh`), rồi **judge mù 15 giám khảo** (5 panel × 3 người,
rubric acc/nat 0-5 y hệt protocol cũ) so 2 ứng viên đầu bảng chrF vs Google/Haiku
(`eval/build_judge_panels_custom.py` — bản tổng quát không hardcode 4 hệ cũ).

| Hệ | acc ja→vi | acc vi→ja | %acc≥4 | vs Google ja→vi |
|---|---|---|---|---|
| Google | 3.22 | 3.40 | 40.5% | mốc |
| Haiku | 4.10 | 4.31 | 79.0% | +0.88 |
| qwen-plus | 4.72 | 4.47 | 90.0% | +1.50 |
| **gemini-flash-lite-latest** | **4.90** | **4.79** | **96.0%** | **+1.68** |

**gemini-flash-lite THẮNG Google ở 10/10 domain ja→vi** (kể cả domain khó nhất:
keigo 4.87 vs 2.73, thanhngu 4.80 vs 2.63, hoithoai 4.90 vs 2.63), thắng cả Haiku.
qwen-plus theo sát, cũng thắng Google 10/10. Đối đầu theo câu: gemini-lite vs
Google 170 thắng/21 hòa/9 thua (n=200); vs Haiku 129/62/9. Xác nhận bằng tay (không
phải ảo giác chỉ số): cả hai dịch đúng kính ngữ, đúng nghĩa bóng thành ngữ, đúng
ngữ pháp phức tạp (「わけではない」phủ định kép).

**Quyết định ban đầu** (trước khi pilot): dùng cả 4 Qwen + Gemini flash-lite,
ước ~50-70k câu. **Pilot 2026-07-20 (300 câu/model + audit lớp 3 phân tầng) ĐẢO
NGƯỢC quyết định này:**

| Model | Rule-filter đạt | Lỗi nghĩa thật (mẫu audit) | Quyết định |
|---|---|---|---|
| gemini-flash-lite | 99.7% | 0/30 (0%) | **GIỮ** — thầy chính |
| qwen-plus | 100% | 0/30 (0%) | **GIỮ** — thầy chính |
| qwen-flash | 99.0% | 11/100 (11%, mẫu mở rộng n=100) | **LOẠI** — false-friend thuật ngữ IT lặp lại có hệ thống (イメージ→"hình ảnh" thay vì Docker image, 枯れている→"lỗi thời" thay vì "đã ổn định/chín muồi" — ĐẢO hàm ý tích cực→tiêu cực; 1 ca đảo phủ định nghiêm trọng) |
| qwen-turbo | 97.3% | 4/30 (13%) | **LOẠI** — đảo nghĩa (貫く"tuân thủ"→"xuyên thủng"), bỏ sót nội dung |
| qwen-max | 98.0% | 5/30 (16.7%) | **LOẠI** — false-friend (エージェント→"đại lý" thay vì AI agent), lẫn ký tự Cyrillic, artifact "Bản dịch:" sót lại, TỆ NHẤT dù là tier đắt nhất |

Áp đúng luật `eval/QUALITY_GATE.md` ("mode nào lỗi >5% thì đọc FULL mode đó") —
3 model Qwen còn lại đều vượt xa 5%, đọc full hàng chục nghìn câu không thực tế
→ loại hẳn thay vì cố dùng.

**Roster thầy CUỐI CÙNG: chỉ gemini-flash-lite + qwen-plus.** Ràng buộc quota:
qwen-plus ước dịch được ~10-15k câu (giới hạn 1M token output), Gemini 500 req/
ngày/key × 2 key = 1000 câu/ngày. Tổng khả dụng **~17-22k câu trong ~1 tuần, 0
đồng** — nhỏ hơn ước tính ban đầu (50-70k) nhưng error rate ~0% thay vì 7-17%.

### 3.2 Pipeline sinh + lọc (đã dựng 2026-07-20)

- `scripts/gen_kd_corpus.py`: dịch hàng loạt ja→vi từ nguồn JA 1 câu/dòng, resume
  theo số dòng đã có, không dùng draft 292M (dịch thẳng từ JA thật — tránh neo theo
  chất lượng yếu 2.37 của model đang train).
- Nguồn: `data/synthetic/kd/ja_pool.txt` = 841k câu của `bt-vong3b` (tái dùng, đã
  xáo trộn seed 20260720) — filter_ja_mono.py đã lọc rác nguồn từ trước (BT vòng 3b).
- **QC 4 lớp (tái dùng `eval/QUALITY_GATE.md`, bỏ lớp đọc seed vì đây dịch câu thật
  không sinh từ gloss) — ĐÃ PILOT + HIỆU CHỈNH XONG 2026-07-20, thông số dưới đây là
  bản CUỐI đã kiểm chứng, không phải dự kiến:**
  1. Rule filter code (`scripts/filter_kd_corpus.py`): rò ngôn ngữ (JA/Hán giản thể lẫn
     vào vế Việt), tỉ lệ độ dài, dedup, refusal/leftover `<think>`. ĐÃ BỎ check
     "..."/"anh chị" (gây báo động giả trên corpus blog kỹ thuật — "..." xuất hiện tự
     nhiên trong code/công thức/lời ngập ngừng; "anh/chị" là xưng hô chuẩn tiếng Việt).
     Đạt 97-100%/model.
  2. **LaBSE ≥0.55** (`scripts/labse_score.py`) — KHÔNG PHẢI 0.80. Pilot cho thấy 0.80
     giết oan ~26% bản dịch ĐÚNG (câu ngắn/kỹ thuật Việt hóa tự nhiên tự nhiên có LaBSE
     thấp dù đúng nghĩa). Ở 0.55 giữ 99,6%. LƯU Ý: LaBSE KHÔNG bắt được lỗi nhầm thuật
     ngữ tinh vi (vd "リモート"→"làm việc từ xa" thay vì "máy chủ remote" vẫn scored
     0.622, "Pod"→"podcast" scored 0.705 — cả hai TRÊN ngưỡng). Lỗi loại này CHỈ bắt
     được bằng lớp 3.
  3. Đọc mẫu phân tầng (Claude, có prompt priming ví dụ lỗi cụ thể) — ~30-100 câu/model,
     model nào lỗi >5% loại hẳn (không đọc full — đã thử "khử lỗi" bằng cách nhờ model
     khác verify, THẤT BẠI 0/5 recall, xem §3.1). gemini-lite/qwen-plus: 0% lỗi.
  4. Pilot 300 câu/model đã xong — roster CUỐI: chỉ gemini-lite + qwen-plus (§3.1).

### 3.3 Mở rộng (sau khi G Đợt 2 sóng 1 xác nhận KD ăn)

Nếu cần thêm khối lượng: đăng ký thêm tài khoản DashScope quốc tế (mỗi tài khoản
1M token/model mới) hoặc trả tiền Haiku Batch API cho phần vượt quota free — chỉ
làm sau khi có bằng chứng KD cải thiện judge acc thật ở sóng 1.

## 4. ĐỢT 2 — train KD (phác thảo)

- Mix: **70-75% ja→vi** (KD mới + replay ja→vi cũ) / 25-30% vi→ja duy trì; replay tổng
  ≥65-70%. Binarize bằng pipeline premix hiện có (`scripts/pack_premix.py`).
- LR thấp (≤6e-5), decay dài; +4-6k step từ checkpoint nền Đợt 0.
- Gate sóng 1 (harness y hệt): **zeropronoun & thanhngu & keigo thắng Google thật**
  (judge acc cao hơn); TB ja→vi 2.37 → 3.2-3.5; không domain nào tụt >0.3; FLORES
  ja→vi không giảm >1 chrF. Đạt → sóng 2 mở rộng KD lên 841k+, nhắm nốt
  nguphap/caudai/hop/it_deep/hoithoai tới khi TB ja→vi ≥ 3.7 (mốc Google).

## 5. Nhánh phụ đang cân nhắc — 100M/150M CHỈ ja→vi, from-scratch (2026-07-20, chưa bắt đầu)

Câu hỏi: bỏ hẳn vi→ja, train riêng model nhỏ hơn CHỈ ja→vi có đủ data không?
**KHÔNG phải "chỉ dùng data KD"** (13-19k câu quá ít để học ngôn ngữ từ đầu, thiếu
~1-2 bậc độ lớn) — mà là giữ **nửa ja→vi của premix hiện có** (base 5,4M cặp +
vòng1/2/3a, ước ~11,7M câu ≈ 500M token) + cộng KD mới đè lên trên. Đây là quy mô đủ
để train from-scratch có ý nghĩa, không phải thí nghiệm thiếu data.

Ước tính chi phí (quy đổi tuyến tính từ tốc độ ĐÃ ĐO THẬT của 292M trên Modal L40S —
26.000 tok/s — CHỈ tham khảo, scaling thực tế có thể lệch):

| Model | tok/s ước tính | 3 epoch (1,5 tỷ tok) | 4 epoch (2 tỷ tok) | 5 epoch (2,5 tỷ tok) |
|---|---|---|---|---|
| 100M | ~76.000 | ~5,5h | **~7,3h** | ~9,1h |
| 150M | ~50.600 | ~8,2h | ~11h | ~13,7h |

Rẻ bất ngờ so 292M (Phase1: ~42h GPU cho 3,9 tỷ token). Đáng thử nếu 100M đơn hướng
thắng được 292M song hướng ở ja→vi — vừa nhẹ 3x vừa nhanh lặp vòng data hơn.
⚠️ Bài học Đợt 0: KHÔNG chạy cố định số epoch rồi hy vọng — đo judge ở từng epoch,
dừng khi held-out bắt đầu tụt (chữ ký overfit đã thấy ở wave0).

**CHƯA làm gì cho nhánh này** — cần: (1) script tách phần ja→vi từ premix hiện có,
(2) quyết định có đáng làm SONG SONG với Đợt 1 KD hay để sau khi có kết quả Đợt 2.

## 6. Việc user / trạng thái tính đến 2026-07-20 tối

1. ~~Chạy 3 lệnh modal Đợt 0~~ ✅ XONG — kết quả: dùng step30000, bỏ wave0 (§W0.5).
2. ~~Chốt ngân sách Đợt 1~~ ✅ ĐỔI HƯỚNG — dùng gemini-lite+qwen-plus free thay Haiku
   trả phí, trần ~19k câu/tuần (không phải 300k).
3. **Còn phải quyết:** chạy Đợt 1 quy mô lớn ngay (13-19k câu, script sẵn
   `scripts/gen_kd_corpus.py`), hay thử nhánh 100M/150M (§5) trước/song song, hay đăng
   ký thêm tài khoản DashScope để tăng trần KD trước — xem STATUS.md mục "VIỆC TIẾP THEO".
4. Nếu chạy Đợt 1: JA nguồn đã sẵn `data/synthetic/kd/ja_pool.txt` (841k câu, đã xáo
   trộn) — chỉ cần chạy `gen_kd_corpus.py` với START/COUNT không chồng lấn cho 2 model,
   qua rule filter + LaBSE (ngưỡng đã chốt 0.55) + đọc mẫu phân tầng trước khi mix.
