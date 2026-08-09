# TỔNG KẾT VÒNG V8 — targeted-KD tiếp từ v7a_avg (18L/152,1M)

> Ngày: 2026-08-09 · Train trên Modal **thaovyh2t** (L40S) · Step 8750 → **16750** (+8000)
> Trạng thái: **train XONG, đã nghiệm thu chấm mù**. Chưa chốt deploy — chờ quyết định (soup / v8 / retrain).

---

## 0. TL;DR (đọc cái này trước)

- **Trên 64 câu v7a TỪNG SAI (đích của vòng KD): v8 THẮNG ÁP ĐẢO 39–7** (hoà 18), cắt "bad" từ **37 → 11**, sửa dứt điểm 31/64. **KD đã làm đúng việc.**
- **Trên 150 câu phổ thông: v8 hơi KÉM** — thua đối đầu 24–34 (hoà 92), thêm 3 câu "bad", có **7 câu regression thật**. Cái giá ~2% của việc KD hơi nặng tay.
- **chrF-proxy nói v8 tốt hơn (+0.44); chấm mù LLM nói ngược trên phổ thông.** Tin chấm LLM (đo nghĩa-vs-nguồn), không tin chrF (đo trùng chuỗi với Google).
- **Chưa nên deploy thẳng v8.** Việc nên làm tiếp: **model-soup v7a_avg⊕v8** (rẻ) → nếu không ăn cả hai đầu thì **retrain mix KD nhẹ hơn**.

---

## 1. Cấu hình train (thông số đầy đủ — để tái lập / continue)

| Tham số | Giá trị |
|---|---|
| Kiến trúc | decoder-only BitNet b1.58, d_model **768**, n_layers **18**, n_heads **12**, d_ff **2048** |
| Vocab / max_seq | 32001 / 256 |
| Params | **152,1M** |
| Điểm xuất phát | `v7a_avg.pt` (averaged milestone, **step 8750**) |
| Lịch LR | RESTART: đỉnh **5e-5**, min **5e-6**, warmup **200**, `lr-anchor 8750`, cosine |
| Effective batch | max-tokens **8192** × grad-accum **16** = **131 072 tok/step** (= y hệt run nhà 2048×64) |
| label smoothing | 0.1 |
| Data mix | base `bin_v7g` **15 881 846 seq** + **KD V8 ×8 oversample = 104 800 seq** → tổng **15 986 646 seq / 951M tok**; dev **3 878** |
| Phần cứng | Modal **L40S**, `BITNET_OPT=off`, `torch.compile(dynamic=True)` |
| Tốc độ / wall-clock | **~1,17 s/step**, ~47 000 tok/s → **~2h35m** cho 8000 step (compile lần đầu ~2min) |
| Chi phí | ~$7–10 / $30 (ước tính; xem hoá đơn chính xác ở modal.com/settings/usage) |

**CLI train (nguyên văn):**
```
python3 scripts/train.py --d-model 768 --n-layers 18 --n-heads 12 --d-ff 2048 \
  --vocab-size 32001 --max-tokens 8192 --grad-accum 16 --max-seq 256 \
  --compile --fixed-shapes --pad-multiple 32 --label-smoothing 0.1 \
  --bin-dir data/bin --lr 5e-05 --min-lr 5.00e-06 --warmup 200 --lr-anchor 8750 \
  --max-steps 16750 --save-every 250 --milestone-every 1000 --log-every 10 --dev-every 500
```
Trình điều khiển: `cloud/modal_train_v8.py` (hàm `train_v8`, `eval_1200`, `status`).

---

## 2. Kết quả eval tự động (proxy — theo dõi tiến triển)

### 2.1 dev_loss (held-out, tin cậy) — GIẢM ĐƠN ĐIỆU

| step | 9000 | 10000 | 11000 | 12000 | 13000 | 14000 | 15000 | 16750 |
|---|---|---|---|---|---|---|---|---|
| dev_loss | 2.1112 | 2.1088 | 2.1011 | 2.0907 | 2.0775 | 2.0625 | 2.0504 | **2.0368** |

Tổng **−0,0744**, không chững, không bật lại → không thoái hoá dù đã LR-restart.

### 2.2 chrF vs Google (proxy — CẢNH BÁO: mâu thuẫn với chấm LLM, xem §3)

| Mốc | chrF toàn bộ (v8 / v7a / Δ) | chrF 64 câu (v8 / v7a / Δ) |
|---|---|---|
| step 10000 | 49.19 / 49.39 / **−0.20** | 42.20 / 40.19 / +2.01 |
| step 12000 | 49.77 / 49.39 / **+0.38** | 40.16 / 40.19 / −0.03 |
| step 14000 | 49.80 / 49.39 / **+0.41** | 42.96 / 40.19 / +2.77 |
| **16750 (final)** | **49.83 / 49.39 / +0.44** | **42.13 / 40.19 / +1.94** |

---

## 3. NGHIỆM THU CHẤM MÙ LLM (số thật — Sonnet 5, blind A/B, đã giải mã key)

Phương pháp: mỗi câu hiện JA nguồn + Google (tham khảo) + 2 bản dịch **A/B randomize** (grader không biết hệ nào). Chấm: chính xác nghĩa (phủ định, kính ngữ, số, chủ-vị…) → độ tự nhiên. 5 subagent song song. File: `eval/v8_grade/`.

### 3.1 Tập 64 câu v7a TỪNG SAI (trọng tâm)

| | v8 | v7a |
|---|---|---|
| Thắng đối đầu | **39** | 7 (hoà 18) → **v8 85%** |
| good / ok / bad | **31 / 22 / 11** | 9 / 18 / 37 |

→ v8 sửa dứt điểm (`good`) **31/64 = 48%**, cắt bad 37→11. Theo domain (v8/v7a/tie): life 6/0/0 · news 9/1/1 · work 6/1/3 · tech 6/2/2 · travel 6/2/6 · orig_mix 6/1/6.

### 3.2 Tập 150 câu random (no-regression)

| | v8 | v7a |
|---|---|---|
| Thắng đối đầu | 24 | **34** (hoà 92) → v8 **41%** |
| good / ok / bad | 118 / 25 / **7** | 126 / 20 / 4 |

→ **7/150 câu v8 làm hỏng thêm** (v8 bad, v7a không bad). Ví dụ: 編み物 "đan len"→"đan lát"; お義母さん "mẹ chồng"→"mẹ"; nhầm chủ-vị; đảo lý do trong câu. Chi tiết đầy đủ ở `grade_merged.json` khoá `rand150`.

### 3.3 Vì sao chrF ≠ chấm LLM

chrF đo **trùng chuỗi ký tự với Google** → v8 tình cờ giống lời văn Google hơn ⇒ +0.44. Chấm LLM đo **đúng nghĩa so với tiếng Nhật gốc** → phát hiện v8 trôi nhẹ trên phổ thông. **Chấm LLM là thước đo nghiệm thu.**

---

## 4. BA ĐƯỜNG ĐI TIẾP (chi tiết + recipe)

### (soup) Trộn trọng số v7a_avg ⊕ v8 — RẺ NHẤT, THỬ TRƯỚC
Chỉ là trung bình có trọng số state_dict, **chạy được trên CPU laptop** (~vài giây). Thường thu hồi regression mà giữ phần lớn cú sửa.
```bash
# tạo soup ở 3 tỉ lệ (alpha = trọng số cho v8)
python eval/v8_grade/soup_checkpoints.py --v7a E:/Bit-Translate-data/checkpoints_v7a/v7a_avg.pt \
    --v8 E:/Bit-Translate-data/checkpoints_v8/v8_step16750.pt --alpha 0.3 --out soup_a03.pt
# lặp cho --alpha 0.5 và 0.7
```
Rồi eval lại (nên chạy trên Modal L40S cho nhanh):
```bash
# upload soup lên volume, rồi:
modal run cloud/modal_train_v8.py::eval_1200 --ckpt soup_a05.pt --tag soupA05
# và chấm mù lại tập-64 + tập-150 y quy trình §3 (dùng lại build_blind_grade.py + merge_grade.py)
```
Chọn alpha **ăn cả hai đầu**: giữ v8-win trên 64, kéo tie/no-regression trên 150.

### (2) Deploy thẳng v8 — 0 chi phí
Nhận cú sửa tập-64 mạnh nhất, chịu ~2% tax phổ thông. Chỉ nên nếu ưu tiên tuyệt đối là dẹp ca khó và tax được coi là chấp nhận được.

### (3) Train lại với mix KD NHẸ hơn — TỐN, phải có GPU (Modal)
**KHÔNG chạy được trên laptop** (cần GPU). Nguyên nhân regression = KD ×8 oversample hơi nặng. Sửa: hạ oversample xuống **×3–4** (hoặc tăng tỉ lệ base replay) trong `scripts/mix_and_binarize.py`, re-binarize, rồi chạy lại `train_v8`. Model vẫn đủ KD sửa ca khó nhưng "nền" phổ thông đậm hơn → bớt trôi. Chi phí ~$8–10, ~2,5h.

---

## 5. BẢN ĐỒ FILE (ở đâu, lấy thế nào)

### Trong repo (đã có, pull là thấy)
| File | Nội dung |
|---|---|
| `eval/TONGKET_V8.md` | tài liệu này |
| `eval/v8_grade/grade_merged.json` | **chi tiết từng câu** đã giải mã (winner, v8_q, v7a_q, why, domain) — tập-64 & tập-150 |
| `eval/v8_grade/v8_final.jsonl` | 1199 bản dịch của v8 (step 16750) |
| `eval/v8_grade/verdicts/*.json` | verdict thô 5 subagent (ẩn danh A/B) |
| `eval/v8_grade/items/*.json` | input chấm (A/B randomize) |
| `eval/v8_grade/grade_*_key.json` | key giải mã A→hệ nào |
| `eval/v8_grade/soup_checkpoints.py` | script model-soup (CPU OK) |
| `eval/v8_grade/build_blind_grade.py`, `merge_grade.py` | dựng bộ chấm + gộp/giải mã |
| `cloud/modal_train_v8.py` | train + eval_1200 + status trên Modal |

Bản dịch tham chiếu (đã có từ vòng trước, ở scratchpad / cần thì hỏi): `v7a_1200.jsonl`, `google_1200.jsonl`, `combined_1200.txt`, `v7a_errors.jsonl`.

### Checkpoint (lớn, KHÔNG trong repo)
| Checkpoint | Vị trí |
|---|---|
| **v8 step16750** (ứng viên deploy) | Modal `vija-v8-vol:checkpoints_v8/last.pt`; đang lưu bền về `E:/Bit-Translate-data/checkpoints_v8/v8_step16750.pt` (Máy A) |
| v8 milestones 10k/12k/14k(/16k) | Modal `vija-v8-vol:checkpoints_v8/stepNNNNN.pt` |
| **v7a_avg** (bản deploy HIỆN TẠI, step 8750) | `E:/Bit-Translate-data/checkpoints_v7a/v7a_avg.pt` (Máy A) + GitHub Release. ⚠️ Đã bị GHI ĐÈ trên volume (last.pt giờ = v8) |
| src model / tokenizer | `src/bitnet.py`, `tokenizer/spm_vija_32k.model` (trong repo) |

Lấy checkpoint từ Modal:
```bash
modal volume get vija-v8-vol checkpoints_v8/last.pt ./v8_step16750.pt
```

---

## 6. LƯU Ý KHI LÀM TRÊN LAPTOP (Máy B — KHÔNG CUDA)

- **Soup averaging: chạy tốt trên CPU** (chỉ cộng tensor, vài giây).
- **Eval 1199 câu trên CPU: CHẬM** (~1–2h ở `OMP_NUM_THREADS=5 MKL_NUM_THREADS=5`). Nên đẩy eval lên Modal L40S (~4 phút/lần).
- **Train (kể cả option 3): KHÔNG làm được trên laptop** — bắt buộc Modal.
- Nhớ `OMP_NUM_THREADS=5 MKL_NUM_THREADS=5` (laptop nhiều luồng làm CHẬM đi — đo thật 21 tok/s ở 12 luồng).
- Xem/đọc `grade_merged.json` để mắt thường kiểm 7 câu regression + 11 câu v8 vẫn bad — quyết định chất lượng cuối là của người (CLAUDE.md §9).
