# QUALITY GATE — quy trình review data BẮT BUỘC trước khi train

> Nguyên tắc: **không có cặp câu nào vào `data/bin` mà chưa qua đủ 4 lớp dưới đây.**
> Áp dụng từ Vòng 2 (2026-07-16). Lặp lại y hệt cho mọi vòng sau.

## 4 lớp lọc (theo thứ tự)

| Lớp | Ai làm | Bắt gì | Công cụ |
|---|---|---|---|
| **1. Rule filter** | code | rỗng / trùng (ja,vi) / sót ký tự Nhật trong vế Việt / vế Nhật không có chữ Nhật / tỉ lệ độ dài bất thường | `mix_and_binarize.py` (tự động) + script gộp |
| **2. Claude đọc GLOSS/SEED** | Claude | nghĩa SAI từ gốc (1 gloss sai đầu độc ~22 câu sinh ra) — bịa idiom, nghĩa ngược, nghĩa lệch | đọc **100%** danh sách seed/gloss trước khi sinh câu |
| **3. Claude đọc CẶP CÂU** | Claude | tiếng Nhật bịa cách dùng / gãy ngữ pháp; tiếng Việt dịch literal, lệch nghĩa, sai chiều phủ định, sai register | mẫu phân tầng **25/mode/provider**; mode nào lỗi >5% → đọc **toàn bộ** mode đó; kết quả ghi blacklist |
| **4. LaBSE ≥ 0.8** | code (GPU/CPU) | cặp lệch nghĩa ngữ nghĩa mà mắt thường sót | bước mix trên node (`mix_and_binarize.py`) |

Blacklist tích luỹ: `eval/vong2_review_blacklist.jsonl` (+ file tương ứng mỗi vòng) — mỗi mục ghi **lý do** để tái sử dụng làm bộ nhớ lỗi của provider.

## Nhật ký review Vòng 2 (cập nhật liên tục)

| Batch (provider × mode) | Cỡ | Cách review | Kết quả |
|---|---|---|---|
| flash-lite: pb/ht/hop/ps/dn | 3.362 | 125 mẫu + **full dn (220)** | xoá 19 (`結構`="cấu trúc" bịa, `冷たい` IT bịa, lệch nghĩa) |
| 2.5-flash: pk/g2/id | 597 | full id (66) + full g2 (125) + 100 mẫu pk | ✅ sạch, 0 xoá — polarity phủ định đúng 100% mẫu |
| DashScope qwen-plus: pk/g2/id | ~1.800 | 60 mẫu | ✅ sạch (id/pk/g2 chuẩn) |
| DashScope qwen-plus: **dn** | 980 | mẫu lỗi ~5% → **đọc FULL 980** | **xoá 120 (12%)** — bịa hệ ẩn dụ (`濡れる` cho IT, `道` ghép, `食べる` thơ), mismatch nghĩa (`三十路→"ba chín"`, `甘いコーヒー→"ít ngọt"`) |
| Zhipu glm-4.7-flash: ht/hop/ps | 1.078 | 60 mẫu → lỗi ~35%! | ❌ **CÁCH LY TOÀN BỘ** (lẫn tiếng Trung vào vế Việt, romaji, bịa thêm nội dung) → sinh lại 48/52 task bằng qwen-turbo (+890 cặp, mẫu 30 sạch ✓) |
| **Gloss harvest (654)** | 654 | **Claude đọc 100%** | **xoá 178** (bịa cụm 水/金/động vật, padding 気が気でない), **sửa 31 nghĩa** → 476 sạch |
| idh qwen-turbo | 1.657 | 78 mẫu đã đọc | ~90% đạt; turbo thỉnh thoảng gãy ngữ pháp JA → tăng cỡ mẫu + tin vào LaBSE lớp 4; các cặp JA-gãy phát hiện được → blacklist |
| idh qwen-plus | ~2.500 | 60 mẫu (chờ xong) | — |

## Bài học provider (để chọn model cho vòng sau)
- **Zhipu GLM-4.7-Flash: CẤM sinh cặp có vế TIẾNG VIỆT** — rò tiếng Trung vào target ("Hán hoá" đúng nghĩa đen), bịa thêm nội dung. Chỉ dùng nếu target là tiếng Nhật + lọc gắt, hoặc bỏ hẳn.
- **Mode dn (đa nghĩa) là mode độc nhất**: MỌI model (kể cả qwen-plus) đều bịa cách dùng khi bị ép vắt "nghĩa hiếm" của 1 từ (~12% lỗi). Vòng sau: giảm số nghĩa yêu cầu/từ (chỉ nghĩa PHỔ BIẾN), hoặc chấp nhận đọc full 100% mode này.
- **Model nhỏ/lite bịa cách dùng tiếng Nhật** khi bị ép sinh "nghĩa hiếm" (dn) hoặc danh sách dài (harvest) → mode kiểu đó PHẢI dùng model mạnh (qwen-plus/2.5-flash) + Claude đọc 100% seed.
- **Phủ định kép**: 2.5-flash + thinking làm chuẩn — giữ công thức này.
- **qwen-turbo**: được cho volume câu đơn giản; KHÔNG giao mode cần tiếng Nhật tinh tế.
- Câu Nhật GÃY nguy hiểm nhất ở chiều **vi→ja** (model học sinh tiếng Nhật hỏng) — ưu tiên bắt lớp 3.
