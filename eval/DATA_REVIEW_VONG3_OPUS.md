# THẨM ĐỊNH DATA VÒNG 3 — 292M VI↔JA (Opus review, 2026-07-18)

> Giám định độc lập bởi Claude Opus qua Agent, đối chiếu PLAN_RANKUP_292M §2/§8,
> PLAN_BUOC5 §4.1/§5.1/§9, eval/capacity_log.md với số liệu thật trên đĩa.

## 0. Số liệu thực đã kiểm chứng trên đĩa

| Hạng mục | Trạng thái thật 2026-07-18 |
|---|---|
| Từ điển idiom/slang 2 chiều (`vong3_gloss_pool.jsonl`) | **1.247 mục** — 206 gốc VIỆT + 1.041 gốc NHẬT; 106 mục có `lit`. Đích ~2k ⇒ ~62% |
| Câu ngữ cảnh idc (ja→vi) | đang gen (~20% lúc review) |
| Câu ngữ cảnh idcv (vi→ja) | đang gen |
| Cặp tương phản ct | đang gen |
| Code-gen §4.1 (số liệu 万/億/loại từ/phủ định) | **CHƯA có script** |
| Code-gen §8.1 (nối câu ngắn → câu ghép 25-60 tok) | **CHƯA có script** |
| Code-gen `ctx ||| src` từ OpenSubtitles/TED (§8.3, §9#1) | **CHƯA có script** |
| Glossary-injection `[term=訳語]` samples | **CHƯA có script** |
| って+mệnh lệnh / あくまで (lỗi phát hiện 18/07) | **KHÔNG thuộc category data nào đang có** |

## 1. VERDICT — THIẾU cho gate G1

Chưa đủ để tuyên G1, nhưng phần idiom/slang đúng hướng cho 2/5 domain:

| Domain G1 (ja→vi) | Data vòng 3 phủ? | Đánh giá |
|---|---|---|
| thanhngu | ✅ phủ mạnh (idc + ct) | Khả năng thắng cao — Google yếu nhất (2.5-3.0) |
| slang | ✅ phủ (422 slang JA + slang VIỆT) | Khả năng thắng cao (Google 3.3) |
| hoithoai | ⚠️ gián tiếp — KHÔNG có data `ctx\|\|\|` chuyên trị zero-pronoun | 110M đã hơn Google (56.5 vs 52.3) → giữ được nhưng không có đòn mới |
| hop | ❌ không data mới | Vòng 2 từng làm họp ja→vi tụt −4.4 (T2) — phó thác 292M, là cá cược |
| it_deep | ❌ không data mới | Tiềm năng có (3 câu từng thắng Google đều IT ja→vi) nhưng không đòn mới |

**Ba lý do thiếu cứng:**
1. **probe64 ja→vi chrF ≥52.2 là nghẽn thật** (đang 48.7, đã tụt từ 51.2). Idiom là memorization — gần như không nhích chrF probe64 tổng. Rủi ro số 1 của G1.
2. Lỗi discourse/quotative mới (って+mệnh lệnh mất tính mệnh lệnh, あくまで nghĩa đen) **không nằm trong category nào** — lỗ hổng ngữ pháp chức năng, không phải từ điển.
3. hop + it_deep **không có một cặp data mới nào**.

## 2. GAP xếp theo tác động/chi phí

| # | Thiếu gì | Bao nhiêu | Tác động |
|---|---|---|---|
| G-1 | Hoàn tất gen idiom context | ~10-12k thô → 7-9k sạch | Nền thanhngu+slang — phải xong |
| G-2 | **Cặp `ctx \|\|\| src` code-gen** OpenSubtitles/TED | 30-50k, 0đ | Trực tiếp hoithoai G1 + zero-pronoun (+0.3-0.6 judge kỳ vọng) |
| G-3 | **Discourse/quotative markers** (って mệnh lệnh, あくまで, からって…) | 500-1k | Ăn thẳng lỗi mới 18/07 |
| G-4 | Số liệu 万/億/loại từ/niên hiệu + phủ định N2-N1 code-gen | 10-15k, 0đ | Lỗi số deterministic + phủ định; nâng probe64 |
| G-5 | Nối câu ngắn → câu ghép 25-60 tok code-gen | 100-150k, 0đ | Thuốc chính vi→ja câu dài → **G2, không phải G1** |
| G-6 | Glossary-injection `[term=訳語]` | 5-10k, 0đ | Vượt Haiku Phase 3 + term khách |
| G-7 | Harvest idiom 1.247 → ~2k | +750 mục | Xem cảnh báo trần trước khi nở |
| G-8 | BT 500k-1M + thầy-trò | Phase 2 | Cần 292M qua gate — hoãn |

**Mấu chốt:** 4 hạng mục **0 đồng** (G-2/4/5/6) chưa có script — tác động cao/chi phí 0 đang bỏ trống.

## 3. KẾ HOẠCH ĐỢT KẾ — tách 2 vòng train (không trộn 2 bệnh 1 đơn thuốc)

### Vòng 3a — G1 (ja→vi), làm TRƯỚC
| Hạng mục | Cách sinh | SL | Ưu tiên |
|---|---|---|---|
| Hoàn tất idc/idcv/ct + filter + Haiku review + LaBSE | LLM free | 7-9k sạch | **P0** (đang chạy) |
| Cặp `ctx \|\|\| src` cùng block hội thoại | **CODE mới** | 30-50k | **P0** |
| Discourse/quotative markers (seed JLPT N2-N1) | LLM free + Haiku review | 500-1k | P1 |
| Số liệu + phủ định template | **CODE** §4.1 | 10-15k | P1 |
| Glossary-injection (mẫu có/không gợi ý) | **CODE** | 5-10k | P2 |
| Harvest idiom → 2k mục | LLM free | +750 | P2 (sau đo trần) |

### Vòng 3b / Phase 2 — G2 (vi→ja), SAU gate 292M
| Hạng mục | Cách sinh | SL | Ưu tiên |
|---|---|---|---|
| Nối câu → câu ghép 25-60 tok, oversample dài | CODE §8.1 | 100-150k | P2 |
| BT 500k-1M (ép ≥40% câu 30-60 tok) | 292M infer local | 500k-1M | P3 |
| Thầy-trò error-targeted | 292M dịch + Haiku sửa (qua Claude Code) | 30-50k | P3 |

## 4. CẢNH BÁO TRẦN SỨC CHỨA (đối chiếu capacity_log)

- 110M bật cả 4 tín hiệu T1-T4 chỉ với 13.858 cặp synth vòng 2; glossary term-trần kẹt 28%.
- 292M ≈ 2,66× params — nhiều chỗ hơn nhưng không vô hạn. Luật:
  1. Vòng đầu 292M: nhét trọn ~1.247 (tối đa 2k) mục × 6-8 câu ≈ **10-14k cặp idiom là hợp lý** — đừng nở quá.
  2. Sau vòng đó **bắt buộc đo lại glossary term-trần**: vượt ~35-40% ⇒ được nở tiếp; đứng ~28% ⇒ DỪNG nhồi, chuyển app-side glossary-injection (§8.4#2).
  3. **KHÔNG nhồi idiom + nối câu + số liệu cùng 1 vòng** (~130k+ synth vượt ngưỡng mix 35%, nhiễu tín hiệu đo trần) — lý do tách 3a/3b.
  4. Chống dương tính giả (§5.1): idiom không ăn điểm ⇒ eyeball 10 cặp trước khi kết tội trần (data mẫu hiện trông sạch).

## TL;DR

- G1 chưa chắc bằng data hiện có: chắc 2/5 domain, hoithoai mong manh, hop+it_deep trống; probe64 ja→vi ≥52.2 là nghẽn idiom không giải được.
- Việc rẻ nhất & thiếu nhất: 4 script code-gen 0 đồng — ưu tiên `ctx|||` + số liệu NGAY.
- Cần category discourse-marker riêng (~500-1k cặp) cho lỗi って/あくまで.
- Trần: vòng đầu 292M ≤ ~14k cặp idiom, đo glossary-rate rồi mới nở; tách data G1/G2 thành 2 vòng.
