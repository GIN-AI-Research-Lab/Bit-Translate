# Pilot audit thầy Sonnet 5 — 300 câu, Opus 4.8 chấm mù (2026-08-08)

> Cơ chế: 3 subagent Sonnet (subscription, KHÔNG tốn API key/tiền — ~200k subagent token)
> dịch `eval/pilot300_teacher_audit.ja` (64 câu v7a từng sai + 236 câu phủ đều 6 miền).
> Opus 4.8 đọc JA + bản Sonnet, chấm acc(0-2)/nat(0-2). Glossary hint áp cho 53/300 câu.

## KẾT QUẢ: ĐẠT — Sonnet 5 làm thầy tốt, thắng dứt khoát trên đúng chỗ v7a yếu

| Nhóm | n | acc==2 | nat==2 | Ghi chú |
|---|---|---|---|---|
| **64 câu v7a từng SAI** | 64 | **64/64** | **64/64** | gồm CẢ 22 ca acc0 thảm hoạ của v7a |
| 236 câu còn lại (phủ miền) | 236 | 236 | 236 | 0 hồi quy, kể cả câu news dài + idiom |

**Sonnet sửa đúng 100% các lớp lỗi mà v7a chết:**
- **Đảo cực/phủ định (v7a acc0):** 思わない?→"cậu không thấy vậy sao?" (id124), じゃないか→"chẳng cải thiện chút nào cả!" (id339), 非婚化→"không kết hôn" (id1087), 入れないで→"không cho trứng vào" (id846), 聞こえなくなった→"không nghe được nữa" (id731), 雨も降ってきた→"trời đổ mưa" (id989), 充電の減りが早い→"hết pin nhanh" (id693). Tất cả đúng cực.
- **False-friend katakana:** ケース→"ốp lưng" (id694), フォロー→"theo dõi" (id676), デフレ→"**giảm phát**" (id1085 — ca acc0 ngược nghĩa nguy hiểm nhất). Đúng hết.
- **Sinh rác của v7a:** 機嫌いい→"tâm trạng tốt" (id546), のど乾いた→"khát nước" (id581), 玉石混交→"vàng thau lẫn lộn" (id1077), 立ち直れなかった→"lấy lại tinh thần" (id634). Sonnet không rác.
- **Thành ngữ (v7a dịch mặt chữ/garbled):** 焼け石に水→"muối bỏ bể" (id1091), 灯台下暗し→"ngay dưới chân đèn lại tối" (id1002), 郷に従う→"nhập gia tùy tục" (id1148), 捨てる神…→"cửa này đóng thì cửa khác mở" (id1198). Chuẩn nghĩa.
- **PIN_VI glossary: 5/5 render đúng** (デフレ→giảm phát, ケース→ốp lưng, 過疎→thưa dân, テレワーク→làm việc từ xa, ワイヤレス充電→sạc không dây).
- **Naturalness giữ nguyên guardrail:** khẩu ngữ ở câu đời thường (id6 "ì ạch", id407 "ngủ nướng"), trang trọng ở câu công văn/news; idiom news khó cũng chuẩn (二兎…→"đuổi hai con thỏ", 転ばぬ先の杖→"cẩn tắc vô ưu", 井の中の蛙→"ếch ngồi đáy giếng").

## Caveat trung thực (không rubber-stamp)

1. **Pilot chạy trên câu CURATED (bench 1200), không phải corpus thô.** Kết quả rất mạnh nhưng nguồn KD thật (mine corpus / sinh) có thể nhiễu hơn — audit này chưa phủ câu thô.
2. **Bucket STT-fragment (§T2f) cần prompt thầy RIÊNG + mini-audit riêng.** Prompt hiện tại bảo "dịch tự nhiên" → với mảnh câu cụt, Sonnet sẽ có xu hướng **hoàn thành câu** thay vì để lửng. §T2f yêu cầu ngược lại (dịch lửng, cấm bịa) → phải sửa prompt thầy cho bucket đó và audit riêng, KHÔNG dùng chung kết quả pilot này.
3. **Vài ca acc2 nhưng sát mép:** id37 折り返す→"liên hệ lại sau" (đúng nghĩa, nhẹ sắc thái "gọi lại ngay"); id1041 二階から目薬→dịch nghĩa "hình thức xa rời thực tế" (mất hình ảnh). Không phải miss, chỉ ghi nhận.

## Kết luận cho PLAN_V8

- **§T3 audit thầy: PASS.** Sonnet 5 (qua subagent subscription) đủ chuẩn làm thầy KD — độ chuẩn + tự nhiên đều cao, sửa đúng 100% các bẫy đã định. **Không cần đường API trả tiền** cho pilot; scale cũng chạy subagent được (như job Haiku 1200).
- **Chốt teacher = Sonnet 5 (subscription).** Bung KD khi có nguồn câu (T0 corpus + sinh seed ①②).
- **Việc còn lại trước scale:** (a) T0 chuyển corpus về Máy A; (b) sửa prompt thầy cho bucket §T2f + audit riêng ~60 mảnh; (c) bạn duyệt 7 dòng `AUDIENCE` glossary.
