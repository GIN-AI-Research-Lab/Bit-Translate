# Kết quả chấm mù 200 câu (bench_opus200b) — Opus 5 tự sinh câu + tự chấm

- **Ngày chấm**: 2026-07-30, chấm trong CÙNG một phiên (tránh trôi thang điểm giữa phiên).
- **Quy trình**: 200 câu Nhật do Opus 5 sinh (10 domain × 20 câu, 100 ngắn / 100 dài, 68 câu khó).
  4 hệ dịch → xáo trộn thành cột A–D theo `judge_200b/panel_key.json` (Opus 5 KHÔNG nhìn key khi chấm).
  Mỗi bản dịch chấm `[acc, nat, use]`: acc/nat ∈ {0,1,2}, use ∈ {0,1}.
- **4 hệ**: `v7ai2s` (v7a_avg chạy **i2_s GGUF deploy thật**, 77,56MB), `google` (Google Translate),
  `gate` (v7 step 4000), `v6` (baseline v6_avg).
- **Caveat**: người chấm là Opus 5 (song ngữ, không phải người Việt bản xứ); tuy chấm mù nhãn hệ,
  nhưng cùng là model đã sinh câu nguồn. Bench này gồm domain phổ thông (không slang/game/y khoa hiếm).

## Tổng (n=200)

| Hệ | acc==2 | nat==2 | use | acc TB (0–2) | nat TB |
|---|---|---|---|---|---|
| **v7a i2s (deploy)** | **78,5%** | **76,5%** | 91,5% | **1,72** | **1,74** |
| Google Translate | 69,0% | 60,0% | 92,0% | 1,65 | 1,57 |
| v7 gate (4000) | 74,5% | 74,5% | 92,0% | 1,69 | 1,72 |
| v6 baseline | 75,5% | 74,0% | 92,0% | 1,71 | 1,71 |

- **use (dùng được)**: 4 hệ hòa tuyệt đối ~92%. McNemar v7ai2s vs google: +11/−12, p=1,000.
- **acc==2 (đúng hoàn toàn)**: v7a i2s dẫn đầu, hơn Google **+9,5 điểm**.
- **nat==2 (tự nhiên)**: v7a i2s hơn Google **+16,5 điểm** — Google nhiều câu dịch cứng, dịch máy lộ rõ
  (chấm mù vẫn nhận ra, khớp với ấn tượng khi chưa mở key).

## Ngắn vs dài

| | acc==2 ngắn | acc==2 dài | nat==2 ngắn | nat==2 dài | use ngắn | use dài |
|---|---|---|---|---|---|---|
| v7a i2s | 84% | **73%** | 89% | **64%** | 92% | 91% |
| google | 67% | 71% | 65% | 55% | 89% | **95%** |
| gate | 84% | 65% | 90% | 59% | 92% | 92% |
| v6 | 84% | 67% | 87% | 61% | 93% | 91% |

- **Câu dài là chỗ V7A ăn tiền**: acc dài 73% vs gate 65% / v6 67% (+6..+8 điểm) — đúng mục tiêu train.
- Google vẫn "an toàn" nhất ở câu dài (use 95%): hiếm khi sập hẳn, nhưng chất lượng đỉnh thấp hơn.
- Câu ngắn: cả 3 model nhà đè Google (acc 84% vs 67%, nat ~90% vs 65%).

## Theo độ khó

| | easy (37) | med (95) | hard (68) |
|---|---|---|---|
| v7a i2s acc==2 | 91,9% | 75,8% | **75,0%** |
| google acc==2 | 81,1% | 67,4% | 64,7% |
| gate acc==2 | 91,9% | 74,7% | 64,7% |
| v6 acc==2 | 86,5% | 76,8% | 67,6% |

Câu khó: v7a hơn gate/google ~10 điểm acc — gate→v7a_avg (thêm ~5000 step + data dài) cải thiện rõ ở đây.

## Domain yếu của v7a (use%)

- `art_ent` 80% (google 95%) — nghệ thuật/giải trí vẫn hụt.
- `formal_letter` 80% (v6 90%) — kính ngữ thư tín: cụm cố định vẫn là bẫy.
- Còn lại 90–95%, ngang hoặc hơn Google.

## 3 câu CẢ 4 HỆ đều fail (bẫy chung)

1. **id67** つい子どもに当たってしまう — "trút giận lên con" → cả 4 dịch sai 当たる.
2. **id169** 情報が漏れかけた — "SUÝT lộ thông tin" → cả 4 dịch thành "đã lộ" (mất かけた).
3. **id187** 心ばかりの品ですがお納めください — cả 4 hỏng お納めください (thanh toán/nộp/tặng cho tôi).

Cụm kính ngữ cố định giết nhiều hệ cùng lúc: id182 ご査収 (chỉ google đúng), id184 取り急ぎご報告まで
(chỉ v7a đúng), id189 ご清栄 (chỉ google tạm được).

## Kết luận

1. **Bản deploy i2_s GGUF (77,56MB, ~350 tok/s CPU) giữ nguyên chất lượng**: v7a i2s đứng ĐẦU bảng
   acc/nat — xác nhận lần cuối tokenizer fix đã đóng gap PyTorch↔deploy.
2. **v7a ≥ Google Translate trên bench này**: hòa use, thắng rõ acc (+9,5) và nat (+16,5).
3. **V7A cải thiện đúng chỗ nhắm**: câu dài acc +6..+8 so với v6/gate; câu khó +7..+10.
   use tổng không nhích (đã ~92% trần bench phổ thông) — muốn thấy khác biệt use phải bench miền hiếm.
4. **Còn yếu**: cụm kính ngữ thư tín cố định, nghĩa phụ của động từ thường (当たる, 納める, 漏れかける)
   — đúng loại lỗi B (nghĩa từ theo ngữ cảnh), không phải thiếu thuật ngữ.
