# Tổng kết data VÒNG 3a (2026-07-18) — phục vụ gate G1 (PLAN_RANKUP §2)

## Kho data mới (sau 4 lớp lọc: rule filter → Haiku review → Fable phán xử → chờ LaBSE lúc mix)

| Hạng mục | File | Số lượng SẠCH | Nguồn sinh |
|---|---|---|---|
| Câu idiom/slang (idc/idcv/ct/dm) | **`filtered_labse.jsonl`** | **6.615** (idc 4.518, idcv 1.205, ct 497, dm 395) | DashScope + Gemini; GLM loại cả batch; LaBSE ≥0.70 |
| Cặp ngữ cảnh `ctx \|\|\| src` | `ctx_pairs.jsonl` | **50.000** (25k/chiều, MỘT CHIỀU — có field `dir`) | CODE từ cặp liền kề OpenSubtitles/TED |
| Số liệu 万/億/loại từ/niên hiệu + phủ định | `numeric_negation.jsonl` | **5.457** | CODE template |
| Glossary-injection `[term=訳語]` | `glossary_inject.jsonl` | **4.232** (3.177 ja2vi / 1.055 vi2ja, MỘT CHIỀU) | CODE, phủ 1.006/1.193 mục |
| Từ điển idiom/slang 2 chiều | `../gen/vong3_gloss_pool.jsonl` | **1.247 mục** (206 gốc VI) | harvest 3 provider + Haiku full review 827 mục + Fable cứu 16 mục VI |
| Blacklist review | `../../eval/vong3_review_blacklist.jsonl` | 139 cặp | Haiku 30 panel |

Tổng cộng **~66.3k record mới** cho vòng 3a (≈78k sequence sau khi nhân chiều).

### Ngưỡng LaBSE 0.70 — quyết định có căn cứ (2026-07-18)
Phân bố điểm 7.319 cặp: ≥0.70=90,4% | ≥0.75=79,3% | ≥0.80=59,8% — dốc mạnh vì idiom
dịch NGHĨA BÓNG lệch mặt chữ. Eyeball band 0.70-0.75 (808 cặp): 8/8 mẫu dịch đúng
→ 0.80 (chuẩn mining) giết oan 40%; band <0.70 lẫn cặp VI tự thêm ngữ cảnh (dạy bịa)
→ chốt 0.70 (dải cho phép của CLAUDE.md là 0.7-0.8). Loại 704 cặp (9,6%).

## Chất lượng theo nhóm (Haiku review, sau blacklist)

| Nhóm | Đạt | Ghi chú |
|---|---|---|
| idcv gemini-2.5 | 100% | |
| idc qwen-flash / qwen-turbo | 99%/98% | confirm round: lỗi NGHĨA thật chỉ 2% |
| idc gemini-flash | 97% | |
| idcv qwen-max | 96% | confirm: lỗi nghĩa 2% |
| dm qwen-plus | 87% | mode mới (って mệnh lệnh, あくまで…), 0 lỗi trong 45 mẫu review |
| ct qwen-plus | 54% | **được đọc FULL 690 cặp** — bệnh chính: bịa kanji nghĩa đen cho thành ngữ VIỆT (冷如錢, 食べ如艦, SML…) → 111 cặp loại; 576 cặp sống là data tương phản quý |
| idc glm-4.7-flash | 0% | bỏ cả batch (lỗi 10% sample, khối lượng nhỏ, thay được) |

## Bài học sinh ct (cho lần sau)
Prompt ct cho mục gốc VIỆT phải CẤM bịa cách viết kanji "nghĩa đen" — nghĩa đen của thành ngữ Việt
phải diễn đạt bằng tiếng Nhật tự nhiên (mô tả), không phải dịch từng chữ thành kanji giả.

## Trạng thái pipeline (cập nhật cuối 2026-07-18)
1. ~~Glossary-injection~~ ✅ XONG (`gen_glossary_inject.py` → 4.232 mẫu).
2. ~~mix_and_binarize hỗ trợ dir + LaBSE~~ ✅ XONG: direction `"record"` per-dòng, cấm tự
   đảo chiều (`seqs_for` ja2vi/vi2ja/both); 4 nguồn vòng 3a đã đăng ký trong SOURCES;
   LaBSE vòng 3 chấm offline (`labse_score_vong3.py`, chạy được CPU) → `filtered_labse.jsonl`.
3. **CHƯA CHẠY: lệnh mix cuối** — chờ GPU rảnh (LaBSE cho nguồn vòng 1-2 chạy GPU nhanh hơn)
   và chờ run 292M hiện tại xong gate (không ghi đè `data/bin/train.*` khi train đang chạy):
   `python scripts/mix_and_binarize.py`   (new-frac 0.30 mặc định, synth ≤35% ✓)
4. KHÔNG trộn thêm data nối-câu/BT vào vòng này (để dành vòng 3b/G2).
5. Sau train + gate: đo **glossary term-rate** trước khi harvest từ điển tiếp lên 2k mục.
