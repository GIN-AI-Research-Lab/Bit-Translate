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
- **Việc còn lại trước scale:** (a) T0 chuyển corpus về Máy A; (b) ~~sửa prompt thầy §T2f + audit~~ **✅ XONG (mục dưới)**; (c) bạn duyệt 7 dòng `AUDIENCE` glossary.

## Phụ lục — Mini-audit thầy Sonnet trên câu cụt (§T2f), 61 mảnh

> Prompt thầy RIÊNG cho fragment: "dịch lửng, cấm bịa vị ngữ/hoàn thành câu". Nguồn: 61 mảnh
> cắt câu sẵn có tại ranh giới trợ từ (30 sau-trợ-từ, 27 giữa-danh-ngữ, một số cắt-giữa-từ như
> STT thật) + ca THẬT từ transcript app 07-30. Tiêu chí PASS: dịch trung thực phần đang có +
> **để lửng, KHÔNG bịa thêm vị ngữ / KHÔNG hoàn thành câu** (không tính chuyện hoàn thành từ cụt).

**KẾT QUẢ: 61/61 PASS — 0 ca bịa vị ngữ, 0 ca tự hoàn thành câu.**

- **Ca THẬT nặng nhất (v7a từng bịa "Đội ngũ ĐÃ PHÁT TRIỂN Bitch đó"):** 「そのブリッチとしてそのベトノムの開発チームと日本の」 → Sonnet "Với tư cách là cầu nối đó, đội ngũ phát triển Việt Nam đó và của Nhật Bản…" — để lửng đúng, KHÔNG bịa vị ngữ, còn đoán ブリッチ=cầu nối (bridge) hợp lý. ✅
- Cắt-trước-vị-ngữ (chủ đích): 「財務部から予算超過の」→ "Từ phòng tài chính, về việc vượt ngân sách…" (KHÔNG thêm "bị nhắc nhở"); 「…ログインボーナスを」→ "…tiền thưởng đăng nhập…" (KHÔNG thêm "phải đi lấy"). ✅
- **Giữ đúng cực/thì ngay trong mảnh:** ログインできなくなった→"không thể đăng nhập" (#42), 終わらなくて→"không xong" (#53), 始まった→"đã bắt đầu" (#60); và bài học glossary vẫn giữ: 乗り換え→chuyển tàu, 十分→mười phút, バックアップ→sao lưu.
- Dùng "…" nhất quán đánh dấu chỗ đứt; render đúng thể liên kết/điều kiện (と/たら→khi/nếu, ので/から→vì…nên, て→rồi).

**Lưu ý trung thực:** vài bản dịch hơi lấn cấn tiếng Việt vì bị ép để lửng (vd "một cách đáng kể…", "còn của đám mây…"). Đây là **đặc tính mong muốn** của đích §T2f (thà lửng-trung-thực còn hơn mượt-mà-bịa), không phải lỗi.

**Kết luận §T2f:** prompt thầy fragment **ĐẠT** — Sonnet sinh được data câu-cụt chuẩn (lửng, không bịa). Sẵn sàng dùng để sinh KD §T2f khi có nguồn câu (cắt từ corpus sau T0). Nghiệm thu cuối vẫn là fragment-probe trên chính model v8 (§T7), không phải trên thầy.
