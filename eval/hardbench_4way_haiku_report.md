# BENCHMARK 4 HỆ — 110M vs Google vs Claude Haiku 4.5 vs Claude Fable (2026-07-17)

Bổ sung **Claude Haiku 4.5** vào harness hardbench200 + FLORES, chấm lại cả 4 hệ
trong CÙNG một lượt trọng tài mù (nên số 110M/Google ở đây hơi lệch so
`hardbench_report.md` ngày 16/07 — phương pháp 4-bản/câu + 2 tiêu chí thay vì 3-bản/câu).

## 1. Phương pháp (khác gì bản 16/07)

- Haiku dịch mù (subagent Claude Haiku 4.5, không thấy ref) 200 câu hardbench + 200 câu FLORES.
- Trọng tài: 15 giám khảo Claude (5 panel × 3 người), mỗi câu thấy src + ref + 4 bản dịch
  xáo thứ tự A-D (seed cố định 20260717), chấm 2 điểm nguyên 0-5 **độc lập từng bản**:
  - `acc` — độ chính xác nghĩa (0=vô nghĩa … 3=đúng ý chính có lỗi đáng kể … 5=chuẩn dịch giả)
  - `nat` — độ tự nhiên/văn phong đích (0=không thành câu … 3=lộ dịch máy … 5=như bản xứ, đúng sắc thái)
- Caveat giữ nguyên như report cũ: ref + trọng tài cùng dòng Claude → điểm của
  Fable là "trần trên"; Haiku cùng gia đình cũng có thể được ưu ái nhẹ về giọng văn.
  Khoảng cách lớn tới mức bias không đổi thứ hạng.
- File: `eval/hardbench_haiku.jsonl`, `eval/flores_haiku.jsonl`,
  `eval/hardbench_4way_scores.json` (điểm TB từng câu), script
  `eval/build_judge_panels_4way.py` + `eval/aggregate_judge_4way.py` (tái lập được).

## 2. Chất lượng — hardbench200 (câu khó, trọng tài chấm mù, n=200)

| Hệ | acc vi→ja | acc ja→vi | nat vi→ja | nat ja→vi | % dùng được (acc≥4) |
|---|---|---|---|---|---|
| **110M step23000** | 1.22 | 1.90 | 1.56 | 2.10 | **6.0%** |
| Google Translate | 3.88 | 3.73 | 3.24 | 3.17 | 60.0% |
| **Claude Haiku 4.5** | **4.69** | **4.45** | **3.81** | **3.52** | **92.5%** |
| Claude (Fable) | 5.00 | 5.00 | 4.97 | 4.96 | 100.0% |

- Đối đầu theo câu (acc TB): **110M vs Google 5 thắng/16 hòa/179 thua** (khớp 3/20/177 hôm 16/07 → chấm lại ổn định).
- **Haiku vs Google: 122 thắng/53 hòa/25 thua** — Haiku vượt Google rõ rệt trên câu khó.
- Haiku vs Fable: 0 thắng/115 hòa/85 thua — Haiku kém Fable chủ yếu ở độ tự nhiên (nat 3.5-3.8 vs ~5.0), acc thì khá sát.
- Điểm đáng chú ý: **nat của Haiku (3.5-3.8) chỉ nhỉnh hơn Google (3.2) một bậc** —
  Haiku dịch ĐÚNG nghĩa gần như mọi câu nhưng văn còn lộ giọng dịch; Fable mới đạt "như người viết".

### Theo domain (acc, chọn lọc)

- Haiku thắng Google đậm nhất đúng chỗ Google yếu: **thành ngữ** (ja→vi 4.17 vs 2.97; vi→ja 4.67 vs 2.50), **slang vi→ja** (4.53 vs 3.30), **hội thoại** (4.5-4.7 vs 3.2-3.5), **zeropronoun vi→ja** (4.47 vs 3.47) — tức là các lỗi bắc-cầu-tiếng-Anh và thiếu ngữ dụng của Google.
- 110M vẫn giữ đúng "đảo mạnh" cũ: solieu/zeropronoun/hop ja→vi (2.1-2.3) và vẫn sập ở caudai/slang/thanhngu vi→ja (0.6-0.7).

## 3. Chất lượng — FLORES-200 n=100 (thước trung lập, chrF, ref dịch giả chuyên nghiệp)

| Hệ | vi→ja | ja→vi |
|---|---|---|
| 110M step23000 | 21.5 | 42.3 |
| Google | **42.6** | 53.5 |
| **Claude Haiku 4.5** | 37.2 | 51.1 |
| Claude (Fable) | 40.2 | **54.1** |

Trên văn tin tức/wiki trang trọng chấm sát ref, Haiku THẤP hơn Google một chút —
ngược với hardbench. Cùng một mẫu như Fable ngày 16/07: LLM ăn đứt ở câu đòi tri
thức ngữ dụng, còn NMT lớn bám ref chữ-đối-chữ tốt hơn ở văn formal.
(chrF hardbench của Haiku: vi→ja 33.0 / ja→vi 41.5 — tham khảo, ref cùng giọng Claude.)

## 4. Tốc độ & dung lượng & chi phí

| | 110M 1.58-bit (i2_s) | Google Translate | Claude Haiku 4.5 | Claude (Fable/Opus) |
|---|---|---|---|---|
| Dung lượng model | **69 MB** file, ~114 MB RAM | cloud (N/A) | cloud (N/A) | cloud (N/A) |
| Chạy ở đâu | **CPU local, offline** | cần internet | cần internet (API) | cần internet (API) |
| Tốc độ đo được | **~140 tok/s** sinh chữ, ~912 tok/s đọc prompt (Core Ultra 5 225H, 4-6 luồng, đo 2026-07-17); repo từng đo **344-465 tok/s** trên Ryzen 5600X | ~tức thời (web) | API cloud, model nhanh nhất dòng Claude | chậm hơn Haiku, chất lượng cao nhất |
| Chi phí / 1M token | **0đ** (điện CPU) | free (web) / có phí API | $1 in / $5 out | $10 in / $50 out (Fable); Opus $5/$25 |
| Riêng tư | **dữ liệu không rời máy** | gửi lên Google | gửi lên Anthropic | gửi lên Anthropic |

Ghi chú tốc độ: trên máy đo hôm nay, 4-6 luồng là tối ưu (140 tok/s); 8 luồng tụt còn 102,
12 luồng còn 21 (E-core + tranh băng thông làm hại — đúng tính chất memory-bound của BitNet).
200 câu hardbench chạy hết **21s** qua llama-server (đo 16/07).

## 5. Kết luận

1. **Xếp hạng chất lượng trên câu khó: Fable > Haiku > Google >> 110M.**
   Haiku 4.5 — model RẺ NHẤT của Anthropic — đã đủ vượt Google Translate rõ rệt
   (92.5% vs 60% câu dùng được), củng cố kết luận: khoảng cách của LLM với NMT
   nằm ở tri thức ngữ dụng, không phải kích cỡ hạ tầng dịch.
2. **110M không cạnh tranh chất lượng tổng quát — nhưng là hệ DUY NHẤT offline/0đ/69MB.**
   Niche của nó (đã có bằng chứng từ 16/07): câu ngắn công sở IT ja→vi, nơi nó bám/thắng Google.
3. **Mốc mới cho vòng lặp data:** nếu cần "đối thủ giá rẻ" để so hoặc để sinh data/chấm điểm,
   Haiku là baseline hợp lý ($1/$5 per MTok) — dịch đúng nghĩa 4.45-4.69/5 trên chính bộ đề khó của ta.
   Có thể dùng Haiku sinh back-translation / data vòng 3 rẻ hơn nhiều so với Fable/Opus.
4. Gate 292M sắp tới: so với bảng §2 (cùng harness 4 hệ đã lưu key + script, tái lập được).

## 6. Tái lập

```bash
# 1) Haiku dịch (đã làm bằng subagent, output đã lưu): eval/hardbench_haiku.jsonl, eval/flores_haiku.jsonl
# 2) Dựng panel mù 4 hệ:
python eval/build_judge_panels_4way.py eval/hardbench_haiku.jsonl <panels_dir>
# 3) 15 giám khảo Claude chấm (5 panel x 3), ghi judge_p{0..4}_j{1..3}.jsonl
# 4) Tổng hợp:
python eval/aggregate_judge_4way.py <judges_dir> <panels_dir>
```
