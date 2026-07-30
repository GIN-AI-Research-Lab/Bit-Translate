# TỔNG KẾT MODEL — vòng V7A (2026-07-30)

> Mọi con số dưới đây là **đo thật**, kèm nguồn. Chỗ nào chưa đo thì ghi rõ **CHƯA ĐO**
> chứ không suy diễn. **v7a_avg thay v6 làm bản deploy chính.**

## 1. Model là gì

| | |
|---|---|
| Kiến trúc | **18 lớp, d_model 768, FFN 2048, 12 heads** (không đổi so v6) |
| Tham số | **152,1M** — BitNet b1.58, weight ternary (−1/0/+1), from-scratch (QAT + STE) |
| Vocab | SentencePiece 32.001 (UNIGRAM), chung VI+JA, thẻ hướng `>>vie<<` / `>>jpn<<` |
| `max_seq` | 256 token (input + output cộng lại) |
| Chiều tối ưu | **ja→vi** |
| Checkpoint | **`v7a_avg.pt`** = trung bình 4 mốc step 8000–8750 |
| File deploy | **`v7a_avg_i2s.gguf` — 77,56 MB**, RAM khi chạy ~125 MB |
| Tốc độ | **~350 tok/s** @ 6–8 luồng CPU laptop (Core Ultra 5 225H, cắm sạc) |

**Cách train vòng này** (chi tiết: `PLAN_V7A.md`, log: `logs/overnight_v7a.log`):

1. **Data**: +623.007 câu KD mới (Gemini Live API, 6 key × 20 session; dịch đạt 98% rồi
   dừng), nhắm lexicon đóng + câu dài đào theo độ dài + ghép liền kề (PLAN_V7A §2-3).
   Sau QC còn 614.120 câu (3,9%), gộp corpus cũ → **15.921.937 cặp** → `bin_v7g`
   (15,89M seq / 946M token). **Dev đã lọc rò: 3.886 câu sạch** (trả nợ 12,56% rò vòng 6).
2. **Gate trần scale** (4.000 step, lr-restart 5e-5, Modal tritue12): long-OOD chấm mù
   **80% ≥ ngưỡng 78%** ⇒ **18L CHƯA ĐẾN TRẦN**, dev cuối gate vẫn giảm −0,011/1000 step
   → chạy tiếp 18L, hoãn grow 24L (`logs/KETQUA_V7A_GATE.txt`).
3. **Train tiếp 4000 → 8750**: hết credit tritue12 giữa chừng → **failover sang tài khoản
   Modal `nguyentuanngai`**, resume từ 7500. Step 9000 không kịp save (container chết ngay
   sau log). Dev (dev sạch): 2,0617@4000 → 2,0579@8000 → **2,0569@8500**;
   `v7a_avg` dev (thang ls=0) **0,8435** vs gate_avg 0,8535 (`logs/v7a_final.log`).

---

## 2. Chất lượng — bench 200 câu chấm mù, bản DEPLOY thật đứng đầu bảng

Bench chính: **200 câu Opus 5 tự sinh** (10 domain × 20, 100 ngắn / 100 dài, 68 câu khó),
4 hệ xáo mù chấm **trong CÙNG một phiên**, thang acc/nat ∈ {0,1,2}, use ∈ {0,1}.
Hệ `v7a i2s` là **bản GGUF i2_s deploy thật** (77,56MB) — không phải bản PyTorch.
Nguồn: `eval/judge_200b_claude/KETQUA.md`.

| Hệ | acc==2 | nat==2 | use |
|---|---:|---:|---:|
| **v7a i2s (deploy)** | **78,5%** | **76,5%** | 91,5% |
| Google Translate | 69,0% | 60,0% | 92,0% |
| v7 gate (step 4000) | 74,5% | 74,5% | 92,0% |
| v6 baseline | 75,5% | 74,0% | 92,0% |

- **use ("dùng được") hòa 4 hệ ~92%** — bench phổ thông đã trần, McNemar v7a vs Google
  p=1,000. Muốn thấy khác biệt use phải bench miền hiếm (bài học `ban-do-mien-3-truc`).
- **acc==2: v7a đứng đầu, +9,5 điểm so Google. nat==2: +16,5 so Google** — Google nhiều
  câu dịch máy lộ rõ, chấm mù vẫn nhận ra.
- **Bản deploy i2_s giữ nguyên chất lượng** — xác nhận fix tokenizer (ISSUES #4) đã đóng
  gap PyTorch↔GGUF.

### Ngắn vs dài — câu dài là chỗ V7A ăn tiền

| acc==2 | ngắn | dài |
|---|---:|---:|
| v7a i2s | 84% | **73%** |
| google | 67% | 71% |
| gate | 84% | 65% |
| v6 | 84% | 67% |

- **Câu ngắn: 3 model nhà đè Google** (84% vs 67%; nat ~90% vs 65%).
- **Câu dài: v7a +6..+8 điểm so v6/gate** — đúng mục tiêu train của vòng. Google vẫn
  "an toàn" nhất ở câu dài (use 95%): hiếm sập hẳn nhưng chất lượng đỉnh thấp hơn.
- Câu khó (n=68): v7a **75,0%** vs v6 67,6 / gate 64,7 / Google 64,7.

### Đối chiếu thứ hai — judge Gemini (phiên khác, chỉ tham khảo chéo)

`eval/judge_v7a_opus/` (opus100 OOD): v7a 87% = v6-pt 87%, gate 81%, Google 89%.
`eval/judge_v7a_rand/` (rand200 in-domain): v7a 79%, v6-pt 83%, Google 75,5% — chênh
v7a/v6 KHÔNG có ý nghĩa (McNemar 8/16, p≈0,15). Số quyết định là bảng Opus 4 hệ cùng
phiên ở trên (bench duy nhất có cả v6 lẫn gate lẫn bản deploy thật trong một phiên).

### ⚠️ Caveat của phép đo chính

Người chấm là Opus 5 (song ngữ, không phải người Việt bản xứ), chấm mù nhãn hệ nhưng cũng
là model đã sinh câu nguồn. Bench gồm domain phổ thông — không slang/game/y khoa hiếm.

---

## 3. Điểm mạnh / điểm yếu

**Mạnh** (từ bench 200b chấm mù):
- Câu ngắn đè Google: acc 84% vs 67%, tự nhiên hơn hẳn (nat 89% vs 65%).
- Câu dài cải thiện thật so v6/gate (+6..+8 acc) — data câu-dài + dev sạch có tác dụng.
- Câu khó +7..+10 điểm so mọi hệ còn lại.
- Deploy = PyTorch về chất lượng, 77,56MB, offline, 0đ/câu.

**Yếu — còn mở, xem `ISSUES.md` #5 (mới) và #1**:
- **Cụm kính ngữ thư tín cố định**: ご査収 / お納めください / ご清栄 / 取り急ぎ
  (id182/184/187/189 giết nhiều hệ cùng lúc; id187 cả 4 hệ fail).
- **Nghĩa phụ của động từ thường theo ngữ cảnh**: 当たる = "trút giận" (id67),
  漏れかける = "SUÝT lộ" (id169) — cả 4 hệ cùng fail; lỗi class B (nghĩa theo ngữ cảnh).
- Domain: `art_ent` use 80% (Google 95%), `formal_letter` 80% (v6 90%).
- Câu dài vẫn dưới câu ngắn 11 điểm (73 vs 84) — ISSUES #1 chưa đóng.

---

## 4. Cách chạy deploy ĐÚNG — bắt buộc tokenize ngoài

**KHÔNG để llama-cli/llama-server tự tokenize** — tokenizer dự án là UNIGRAM, llama.cpp
chạy greedy-merge kiểu BPE sẽ xé vụn từ, mất ~16 điểm (nguyên nhân + xác minh:
**`ISSUES.md` #4**). Đường đúng:

1. Chạy `llama-server -m v7a_avg_i2s.gguf -t 8 -c 256` (bitnet.cpp/WSL, một lần).
2. Mỗi câu: tokenize bằng **sentencepiece + `normalize_for_model`** (y hệt lúc train),
   POST `/completion` với **MẢNG token ids**: `[2, 4] + sp.encode(...) + [3]`
   (`<s>` `>>vie<<` … `</s>`), `temp 0`.
3. Code mẫu đã chạy được: `run_gguf` trong `scripts/translate_bench.py`.

| Cấu hình tối thiểu | |
|---|---|
| CPU | bất kỳ, có **AVX2**; không cần GPU |
| RAM / đĩa | ~125 MB / 78 MB |
| Luồng | **6–8, đặt cứng** (`-t 8`) — 14 luồng sụp 1.100× (đo ở TONGKET_V6 §4) |
| Nguồn | laptop phải cắm sạc |
| Mạng | không cần — offline hoàn toàn |

Gói deploy đóng sẵn cho tool dịch (project khác): `D:/Bit-Translate-data/dist/v7a_deploy_pkg/`.

Quantize lại từ nguồn (nếu cần): `llama-quantize v7a_avg_f32.gguf v7a_avg_i2s.gguf I2_S 1`
(số `1` bắt buộc — đa luồng ghi hỏng i2_s).

---

## 5. Trạng thái ngân sách & đường đi tiếp

- **Modal HẾT TIỀN**: tritue12 = $0; `nguyentuanngai` lố hạn mức $3 free sang tiền thật
  (đã stop, không train trả phí nữa).
- Đường train miễn phí duy nhất còn lại: **máy A desktop** (3060 Ti 8GB + 48GB RAM, WSL2,
  CLAUDE.md §7) — chậm hơn L40S ước 2-4×. KD data qua Gemini Live API vẫn gần như miễn phí.
- **Grow 24L**: điều kiện kích hoạt theo PLAN_V7A §1 (long-OOD ≤ 75% VÀ dev phẳng)
  **HIỆN CHƯA THỎA** — 18L chưa trần, chưa grow.
- Đã cấm/đã bác, đừng đề xuất lại (PLAN_V7A + ISSUES): position-offset RoPE, tách câu
  inference, ghép cặp ngẫu nhiên, beam/rerank, data-free FP16→ternary, "cân lại độ dài data".

---

## 6. Nguồn số liệu

| số | file |
|---|---|
| Bench chính 200 câu, 4 hệ cùng phiên | `eval/judge_200b_claude/KETQUA.md` (+ `judges/`, `eval/judge_200b/panel_key.json`) |
| Câu nguồn + bản dịch 4 hệ | `eval/bench_opus200b.jsonl`, `eval/bench_200b_{v7ai2s,google,gate,v6}.jsonl` |
| Verdict gate 18L | `logs/KETQUA_V7A_GATE.txt`, `eval/judge_gate_opus/` |
| Judge Gemini đối chiếu | `eval/judge_v7a_opus/RESULT.json`, `eval/judge_v7a_rand/RESULT.json` |
| Dev-loss + bench PyTorch cuối | `logs/v7a_final.log`, `eval/bench_v7aopus.jsonl`, `eval/bench_v7arand.jsonl` |
| Log train 3 phase | `logs/train_v7a_gate.log`, `train_v7a_p2.log`, `train_v7a_p3.log` |
| Pipeline KD/merge/binarize | `logs/overnight_v7a.log`, `logs/kd_v7a.log` (gitignore) |
| Vị trí checkpoint/model | `ARTIFACTS.md` (mục VÒNG V7A) |
| Issue còn mở | `ISSUES.md` #1 (câu dài), #5 (kính ngữ thư tín + nghĩa phụ động từ) |
