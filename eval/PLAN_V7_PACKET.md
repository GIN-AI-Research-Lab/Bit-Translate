# HỒ SƠ CHỐT VÒNG 7 — model dịch ja→vi 152M, BitNet 1.58-bit

Bạn là chuyên gia MT độc lập được thuê để PHẢN BIỆN. Mặc định là lập luận dưới đây CÓ LỖI.
Chỉ rõ SỐ LIỆU NÀO bác KẾT LUẬN NÀO. Nếu một kết luận không có số liệu chống lưng, nói thẳng.

## 0. Bối cảnh cố định
Transformer enc-dec 18 lớp, d_model 768, **152M tham số**, weight ternary (BitNet b1.58),
`max_seq=256` (input+output CỘNG LẠI, vì generate giữ cả hai trong một chuỗi).
Data train 100% là output LLM thầy (gemini-flash-lite, audit acc 4,90/5, thắng Google 10/10 miền).
Ngân sách vòng 7: **~$9 = ~9.000 step trên L40S**. Không có tiền cho phương án đắt hơn.

## 1. Bench 200 câu — judge MÙ, thang 0/1/2 ĐÚNG NGHĨA, % = tỉ lệ đạt 2đ
Cân bằng nhân tạo: 10 đặc trưng × 20 câu, 100 ngắn / 100 dài. CẢ 3 HỆ CHẤM CÙNG PHIÊN.
⚠️ Judge là Gemini, mà data train v6 do gemini-flash-lite sinh → có thể thiên vị v5/v6.

| | Google | v5 | v6 | McNemar v6 vs Google |
|---|---:|---:|---:|---|
| NGẮN (n=100) | 78% | 80% | **81%** | p=0,710 (ngang nhau) |
| DÀI (n=100) | 87% | 55% | **64%** | p=0,001 (Google hơn thật) |
| TỔNG (n=200) | 82% | 68% | 72% | p=0,025 |

v6 vs v5: TỔNG p=0,144 · dài p=0,124 · ngắn p=1,000 → **KHÔNG đạt ý nghĩa thống kê**.

Theo đặc trưng (n=20 mỗi ô → KTC ±20, chỉ đọc theo hướng):

| đặc trưng | v5 | v6 | Google | mật độ data v6 |
|---|---:|---:|---:|---|
| plain | 70% | **90%** | 85% | – |
| negation | 75% | **95%** | 90% | **6,25×** |
| slang | 50% | 60% | 80% | **5,17×** |
| idiom | 65% | **75%** | 60% | 0,88× (loãng) |
| clause | 55% | 65% | 90% | 1,11× |
| question | 70% | 70% | 85% | – |
| katakana | 70% | 70% | 90% | 1,04× |
| zeropron | 65% | 60% | 85% | 1,07× |
| keigo | 70% | 65% | 75% | 0,98× |
| number | 85% | 75% | 85% | 0,94× |

## 2. Đường cong loss DEV (held-out 4.435 cặp, lần đầu dự án có)
1 epoch = 6.549 step. Corpus 15,28M cặp / 858M token.

| step | epoch | dev_loss | Δ/1000 | train Δ/1000 |
|---:|---:|---:|---:|---:|
| 2000 | 0,31 | 2,1846 | – | – |
| 6000 | 0,92 | 2,1505 | −0,0099 | −0,0071 |
| 10000 | 1,53 | 2,1221 | −0,0067 | −0,0068 |
| 12000 | 1,83 | 2,1069 | −0,0076 | −0,0062 |
| 14000 | 2,14 | 2,0946 | −0,0061 | −0,0057 |
| 16000 | 2,44 | 2,0896 | **−0,0025** | **−0,0126** |
| 16400 | 2,50 | 2,0886 | **−0,0025** | – |
| **avg7** | – | **2,0761** | – | – |

LR cosine: 8e-5 → step12000 2,0e-5 → step16000 8,1e-6.
Trung bình 7 checkpoint cuối mua được −0,0125 dev loss, MIỄN PHÍ.

## 3. Độ phủ vị trí token trong data train (15.284.511 chuỗi, dài TB 56,2 token)
| vị trí | số chuỗi phủ tới đó | % |
|---:|---:|---:|
| 64 | 4.342.143 | 28,41% |
| 128 | 1.481.337 | 9,69% |
| 160 | 734.086 | 4,80% |
| 224 | 58.740 | **0,38%** |
| 256 | 685 | **0,00%** |

Trên 120.000 câu biên bản Quốc hội (văn phong dài nhất dự án có), tổng token JA+VI:
trung vị 80 · p90 152 · p99 220 · **chỉ 0,10% vượt 256** · 0,00% vượt 320.

Nguồn câu dài KHÔNG khan hiếm: `kokkai_ja.txt` 2M dòng đầu có 7,9% câu ≥150 ký tự,
1,3% ≥200 ký tự. CC-100 còn 392,8M câu. Mọi script đào đều chọn theo KHUÔN MẪU
(mệnh đề lồng, slang, phủ định), CHƯA LẦN NÀO chọn theo ĐỘ DÀI.

## 4. Những đòn ĐÃ THỬ VÀ ĐÃ CHẾT (đừng đề xuất lại)
- **Rerank/oracle**: oracle chọn bản tốt nhất trong 9 bản sinh ra chỉ +5,2 chrF;
  LaBSE-rerank TỆ hơn greedy ⇒ "chất lượng KHÔNG giấu trong weights, đòn bẩy decoding chết".
- **Self-edit `>>fix<<`** (2 lượt dịch+sửa): 1 sửa đúng 1 sửa hỏng, chỉ pattern-match,
  không cứu được lỗi thiếu kiến thức (lớp lỗi chủ đạo).
- **CoT/thinking token**: loại từ đầu (model quá nhỏ, phá latency).
- **Beam search**: vô ích. **Trung bình checkpoint**: có tác dụng (+0,7 chrF, nay xác nhận
  bằng dev loss −0,0125).
- **Giả thuyết "KD bị pha loãng"** (train lại mix KD đậm 35%): +0,05 acc rồi tụt vì overfit → BỊ BÁC.
- **Vòng 3** bơm data tổng hợp theo chủ đề: hardbench +18 mà bench thật ĐỨNG YÊN.
- **Vòng 5** bơm 2,45M câu Quốc hội: bench TED đứng yên (p=0,755) nhưng Quốc hội held-out
  nhảy 14%→95% ⇒ bench đo sai miền.

## 5. Phân loại nguyên nhân lỗi (59 câu v5 hỏng mà Google dịch được)
cấu trúc câu 37% là nguyên nhân CHÍNH nhưng chỉ là nguyên nhân DUY NHẤT ở 2/59 câu.
từ vựng thông thường 22% chính / 1 câu duy nhất. thuật ngữ chuyên ngành 3% chính / 0 duy nhất.
**81% câu hỏng có ≥2 nguyên nhân** (2 nguyên nhân 61%, 3 nguyên nhân 20%).

## 6. KẾ HOẠCH VÒNG 7 ĐANG ĐỀ XUẤT — HÃY CÔNG PHÁ

**Data**: giữ `bin_v6` + thêm cặp DÀI ghép từ các cặp NGẮN có sẵn (nối 3-4 cặp bằng
`、`/`そして`, phần Việt nối tương ứng). Chi phí quota KD = **$0**, nguồn vô hạn từ 15,3M cặp.

**Train**: một chu kỳ LR cosine MỚI (8e-5 → thấp) trên corpus đó, 9.000 step = ~$9.

**Lý lẽ**: (a) khoảng cách lớn nhất với Google là câu dài (−23, p=0,001); (b) nguyên nhân
đo được là vị trí 224 chỉ 0,38% corpus; (c) chi phí data $0 nên toàn bộ tiền dồn vào GPU.

**Cơ sở tin LR-restart còn ăn**: chính v6 khởi đầu từ `v5_avg` (đã anneal xuống LR thấp,
dev đã phẳng), nâng LR lại 8e-5, và dev loss vẫn xuống 2,18 → 2,09 trên corpus mà 85,9%
là data CŨ của v5. Điểm yếu: v6 có thêm 14,1% data mới nên KHÔNG tách được đóng góp của
LR-restart với data mới.

**Kiểm sớm**: dev loss mỗi 500 step + 120 câu Quốc hội held-out. Nếu 3.000 step đầu mà
dev không xuống dưới 2,0761 (mốc avg7 hiện tại) thì DỪNG, không đốt hết $9.

**Đã quyết KHÔNG làm**: bơm thêm slang (5,17× rồi chỉ +10); bơm negation/plain/idiom
(đã vượt Google); nới `max_seq` (chỉ 0,10% câu cần, mà vùng 160-256 còn chưa lấp);
bơm zeropron bằng câu đơn (22,5% mật độ mà vẫn hỏng ⇒ thiếu NGỮ CẢNH câu trước).

## 7. CÂU HỎI
**A.** Chỗ nào trong §6 sai hoặc thiếu bằng chứng? Số liệu nào bác nó?
**B.** Với $9, có phương án nào kỳ vọng CAO HƠN §6 không? Nêu cụ thể + cách biết sớm là ăn.
**C.** "Ghép cặp ngắn thành cặp dài" có rủi ro gì mà số liệu ở đây chưa lộ ra?
**D.** Sai sót phương pháp nghiêm trọng nào cả dự án vẫn chưa nhận ra?
