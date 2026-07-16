# Phân tích ja→vi 48 câu × 8 ngữ cảnh — model step-19000 vs Claude (2026-07-16)

> Bộ câu: `eval/probe_javi48.jsonl`; output model: `eval/probe_javi48_step19000.jsonl` (i2_s 68MB, greedy).
> Chấm tay từng câu (Claude làm giám khảo + bản dịch chuẩn): **OK** = dùng được ngay / **P** = hiểu được nhưng lệch / **FAIL** = sai nghĩa hoặc vô nghĩa.

## Bảng tổng: OK/P/FAIL theo ngữ cảnh (6 câu/ngữ cảnh)

| Ngữ cảnh | OK | P | FAIL | Nhận xét |
|---|---|---|---|---|
| Số liệu/ngày giờ | 3 | 0 | 3 | 万 đơn (5万→50.000 ✓), thứ+giờ ✓; NHƯNG 億+万 compound, 令和, mất thứ Sáu |
| Phủ định | 3 | 1 | 2 | わけではない cơ bản ✓, しきれない ✓; **phủ định KÉP ないことはない → dịch NGƯỢC nghĩa** |
| Câu dài | 3 | 3 | 0 | Khá bất ngờ — câu 3 mệnh đề về công ty/chi phí/triển lãm dịch tốt; chỉ rối khi mệnh đề lồng sâu |
| IT sâu | 2 | 1 | 3 | Term phổ biến (memory leak, timeout, conflict) ✓; **term hiếm (rebase, staging-câu phức) hallucinate** |
| Thân mật | 2 | 2 | 2 | Khẩu ngữ やばい/寝坊した/引っ越し vỡ |
| Chủ ngữ ẩn/đa nghĩa | 2 | 3 | 1 | Sai NGÔI ("liên lạc với anh ấy" thay vì "tôi sẽ liên lạc"); 結構です mất nghĩa từ chối |
| Keigo | 1 | 2 | 3 | 恐れ入りますが → "Tôi rất sợ"; ご査収 → "thanh toán"(!); mẫu localization quen thì ✓ |
| Công thức văn hoá | 1 | 1 | 4 | お疲れ様 → "Chào anh" ✓ (data Vòng 1!); nhưng お世話に→"chăm sóc anh", よろしく→"Xin hãy làm ơn", ごちそうさま/お先に literal |
| **TỔNG** | **17 (35%)** | **13 (27%)** | **18 (38%)** | Claude chuẩn ~100% làm mốc trần |

## Taxonomy lỗi còn lại (xếp theo tần suất × độ nguy hiểm)

1. **Công thức văn hoá + keigo dịch literal** — 8 lần (nặng nhất). `お世話になっております→"Tôi luôn chăm sóc anh"` (NGƯỢC vai), `よろしく→"Xin hãy làm ơn"`, `ご査収→"thanh toán"`. → **Vòng 2 phrasebook** (đã plan §3.1) — xác nhận đúng hướng, tăng ngân sách phần này.
2. **Phủ định kép ないことはない/なくもない → dịch NGƯỢC nghĩa** — 3 lần. `お金がないことはない (không phải không có tiền) → "Không có tiền"`. Nguy hiểm nhất về nghĩa. → **KÉO từ Vòng 3 LÊN Vòng 2** (data sinh bằng code + LLM rẻ).
3. **Katakana/term hiếm hallucinate** — 4 lần. `リベース→"dính cơ sở dữ liệu"`, `データベース→"deta"`, câu staging→sách vở(!). Đỡ hơn 14000 nhưng chưa hết. → bổ sung glossary đợt 2: term git/infra hiếm (rebase, hotfix, rollout, canary…) + câu ví dụ.
4. **Số liệu compound & lịch Nhật** — 3 lần. `3億2000万→"300 triệu"` (mất 20 triệu), `令和6年→"2011"`, mất "thứ Sáu". → Vòng 3 như plan (template số sinh bằng code, phủ 億+万 compound, 令和 map).
5. **Zero-pronoun sai ngôi** — ~4 lần. `連絡します→"liên lạc với anh ấy"` (đúng: "tôi sẽ liên lạc"). → giữ lộ trình sau 3 vòng (đổi format input có ngữ cảnh); tạm thời Vòng 2 persona giúp một phần.
6. **Khẩu ngữ/slang** — 2-3 lần (やばい, 寝坊, それな). → Vòng 2 hội thoại (đã plan).
7. **Mệnh đề lồng sâu xáo trật tự** — 3 lần mức P (không FAIL). → Vòng 3 nối câu ghép (đã plan).

## Vòng 1 đã chữa được gì (thấy rõ trong 48 câu)
- IT term phổ biến + code-switch tự nhiên: memory leak, timeout, conflict, pull request, test, release, index.
- 納期→"thời hạn bàn giao", 議事録→"biên bản họp" (glossary BrSE ăn).
- 万 đơn quy đổi đúng (50.000/600.000/80.000/150.000), giờ+thứ đúng.
- わけではない cơ bản + しきれない + まだ〜ていない.
- Câu dài 2-3 mệnh đề mạch lạc hơn hẳn mô tả lỗi cũ ("câu Nhật ra cụt/vụn").

## Điều chỉnh plan rút ra
- **Vòng 2 (giữ + bổ sung):** phrasebook công thức (tăng ~1.5k câu, thêm biến thể trong-email/trong-họp/trong-chat), hội thoại khẩu ngữ (thêm slang list ~100 từ: やばい・まじ・それな・寝坊…), persona xưng hô, **+ phủ định kép (kéo từ Vòng 3)**, **+ glossary đợt 2 term dev-workflow hiếm**.
- **Vòng 3 (giữ):** số liệu compound 億万 + 令和 + thứ/ngày, câu phức lồng sâu.
- Mốc so sánh: chạy lại đúng bộ 48 câu này sau Vòng 2 (`eval/probe_javi48.jsonl` cố định làm regression).
