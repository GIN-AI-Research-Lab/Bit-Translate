# Phân loại nguyên nhân 100 câu fail ja→vi (model 100M v3, acc ≤ 2)

*2026-07-21. Thí nghiệm theo đề xuất reviewer ngoài: "phân loại 100 câu fail theo
nguyên nhân thật sự để xác định nút thắt trước khi đổ thêm data".*

## Nguồn dữ liệu
- 66 câu từ `eval/judge_4way_v3` (hardbench, chỉ lấy chiều ja→vi vì v3 là model
  single-direction) + 34 câu từ `eval/judge_flores_v3` (FLORES, câu tin tức chuẩn).
- Fail = acc trung bình 2 judge ≤ 2.0. Người phân loại: Claude (đọc trực tiếp
  từng cặp src/ref/hyp), mỗi câu gán 1 nhãn **nguyên nhân chính**.
- File chi tiết từng câu kèm nhãn: `eval/error_taxonomy_100.json`.

## Kết quả

| Nhãn | Số câu | Mô tả |
|---|---|---|
| **SEM — sai nghĩa / vỡ cấu trúc** | **61** | parse sai câu, gán nhầm ai-làm-gì, chọn sai nghĩa của từ đa nghĩa |
| NAME — tên riêng / số liệu | 12 | tên người bịa ("Bết-lê-len", 胡錦濤→"Tập Cận Bình", Miller→"gương", Mike→"micro"), số sai (五時→8 giờ, 2泊3日→"3 ngày 3 đêm") |
| OMIT — bỏ sót vế/ý | 8 | rơi hẳn một mệnh đề (thường ở câu ghép dài) |
| TERM — thuật ngữ | 7 | katakana/IT/tài chính (見積もり→"điểm thưởng", プロジェクター→"máy bay chuyên nghiệp", 猥褻物取締法→"Đạo luật Chống Khủng bố") |
| KEIGO — keigo / register / slang | 6 | 召し上がる, いたしかねます, 見送らせていただく hiểu ngược; slang やばい/だるい dịch mặt chữ |
| HALL — hallucination / thoái hóa | 5 | bịa nội dung không có trong src, lặp vô hạn ("giải thưởng, giải thưởng…") |
| FLU — tiếng Việt không tự nhiên | 1 | nghĩa đúng nhưng diễn đạt vụng |

Hai pattern con đáng chú ý bên trong SEM:
- **Đảo vai / đảo phủ định: 13/100** — passive bị lật chủ thể (泣かれて→"tôi khóc cho
  con", 海賊に襲われ→"tên cướp bị tấn công", cảnh sát *phát nổ* bom), phủ định bị lật
  (外出しております→"đang có mặt", パスする→"sẽ đi nhậu", 必ず…してください→"không
  cần"). Đây là lỗi **hiểu ngược 180°** — nguy hiểm nhất với người dùng.
- **Idiom dịch mặt chữ: 9/100** — 目から鱗, お茶を濁す, 二の足を踏む, 朝飯前… Chỉ 9%,
  KHÔNG phải lớp lỗi chủ đạo.

## Kết luận — đúng như giả thuyết reviewer

**~74% câu fail (SEM 61 + HALL 5 + OMIT 8) là lỗi năng lực biểu diễn ngữ nghĩa**,
không phải thiếu kiến thức miền: model vỡ trận với câu ghép ≥2 mệnh đề, kính ngữ
lồng vai, passive/causative — dù mọi từ vựng trong câu đều là từ phổ thông đã có
thừa trong 11,5M câu train.

Chỉ **~25%** (TERM 7 + KEIGO 6 + NAME 12) là lớp lỗi mà data nhắm đích
(glossary, phrasebook, keigo drill, meeting-domain) có thể chạm tới — và ngay cả
NAME phần lớn là lỗi *phiên âm katakana/copy số* mang tính hệ thống, không phải
thiếu entry.

Hệ quả cho kế hoạch:
1. Củng cố quyết định **dừng data nhỏ giọt cho 100M** (đã chốt ở STATUS.md) —
   thêm 32k hay 100k câu meeting/idiom chỉ tác động ≤ ~25% khối lỗi.
2. Với **Phương án A (KD toàn phần)**: kỳ vọng hợp lý là KD dạy được *phong cách
   parse + tránh đảo vai* qua 200k-11M câu dịch mượt của thầy — tức là đánh đúng
   khối SEM, không phải đánh khối kiến thức. Gate +0,3 acc vẫn là thước đo đúng.
3. Lỗi đảo vai/phủ định (13%) là ứng viên metric riêng: đếm tỉ lệ câu bị lật
   chủ thể/phủ định trên bộ probe cố định — rẻ hơn judge toàn phần, nhạy hơn chrF.
4. Nếu KD toàn phần không phá được khối SEM → bằng chứng mạnh rằng trần nằm ở
   **sức chứa 100M**, chuyển hẳn câu hỏi sang 292M/scale (khớp gate sức chứa
   PLAN_BUOC5 §5.1).
