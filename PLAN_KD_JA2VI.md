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

### W0.5 — Ma trận quyết định (điền xong Đợt 0)

So judge acc ja→vi với step30000 (keigo 1.95 / nguphap 1.95 / hop 1.75;
thanhngu 2.85 / zeropronoun 3.35):

| Quan sát | Kết luận | Hệ quả cho Đợt 2 |
|---|---|---|
| keigo/nguphap/hop hồi ≥ +0.3 VÀ thanhngu/zeropronoun giữ (≥ −0.2) | **A: nhiễu LR** — trần chưa chạm | Nền = bản tốt hơn {32500, avg}. Đợt 2 LR thấp (≤6e-5) decay dài, tự tin đổ KD |
| Không hồi, hoặc hồi nhưng domain vòng 3a tụt lại tương đương | **B: cạnh tranh sức chứa thật** | Đợt 2: replay ≥70%, mix nghiêng ja→vi 75/25 NGAY, KD **thay thế** (không cộng thêm) phần OpenSubtitles nhiễu nhất trong mix; nếu KD cũng bão hòa → lúc đó mới bàn scale |
| avg_wave0 > step32500 rõ (≥+1 chrF TB) | averaging ăn — dùng làm chuẩn đóng gói mọi vòng sau | — |

**Đầu ra Đợt 0:** 1 checkpoint nền + 1 kết luận A/B + điền `eval/capacity_log.md` dòng
vòng 3a (đã có đủ số 19100/30000/w0).

## 3. ĐỢT 1 — corpus KD (phác thảo, chốt chi tiết sau Đợt 0)

- **Nguồn JA** (ưu tiên domain gap lớn): 841k in-domain đã có (tái dùng nguồn của
  `bt-vong3b`) + harvest thêm nhắm: nguphap (câu chứa mẫu N2-N1), caudai (30-60 token),
  hop (biên bản họp/business mail — corpus BSD), hoithoai. Wrapper overnight sẵn.
- **Haiku dịch ja→vi qua Batch API** (~50% giá — xác nhận khi triển khai). Ước
  ~$40-50/100k câu giá thường → khởi điểm **300k câu ≈ $60-150**. KHÔNG đưa draft của
  292M cho thầy sửa — dịch thẳng (draft chất lượng 2.37 neo kéo thầy).
- Lọc: LaBSE ≥0.8 + length-ratio + dedup (pipeline cũ). Eyeball 50 mẫu (user).
- Mở rộng lên 841k+ CHỈ sau khi Đợt 2 sóng 1 chứng minh KD ăn.

## 4. ĐỢT 2 — train KD (phác thảo)

- Mix: **70-75% ja→vi** (KD mới + replay ja→vi cũ) / 25-30% vi→ja duy trì; replay tổng
  ≥65-70%. Binarize bằng pipeline premix hiện có (`scripts/pack_premix.py`).
- LR thấp (≤6e-5), decay dài; +4-6k step từ checkpoint nền Đợt 0.
- Gate sóng 1 (harness y hệt): **zeropronoun & thanhngu & keigo thắng Google thật**
  (judge acc cao hơn); TB ja→vi 2.37 → 3.2-3.5; không domain nào tụt >0.3; FLORES
  ja→vi không giảm >1 chrF. Đạt → sóng 2 mở rộng KD lên 841k+, nhắm nốt
  nguphap/caudai/hop/it_deep/hoithoai tới khi TB ja→vi ≥ 3.7 (mốc Google).

## 5. Việc user

1. Chạy 3 lệnh modal Đợt 0 (máy có modal token) — W0.1/W0.2.
2. Eyeball 50 câu bản thắng Đợt 0 khi có kết quả.
3. Chốt ngân sách Đợt 1 (đề xuất khởi điểm 300k câu ≈ $60-150 qua Batch API).
