# Glossary canonical dựng từ 64 lỗi v7a (bench 1200, chấm mù Opus 4.8)

> Nguồn: `scratchpad/v7a_errors.jsonl` — 64 câu v7a acc≤1 trong bench 1200 câu 3 hệ.
> Mục đích: biến các lỗi **term-level** thành từ điển canonical để (1) inject `[term=訳語]`
> lúc tạo KD (`scripts/gen_glossary_inject.py`), (2) đo độ phủ (`scripts/measure_term_coverage.py`),
> (3) pin nghĩa đúng khi thầy dịch. **Teacher-agnostic** — dùng được dù thầy là Gemini (PLAN_V8 §T4)
> hay Sonnet 5.
>
> ⚠️ **Cần bạn duyệt:** cột **Register** ở Bảng A-KATA (giữ tiếng Anh vs dùng từ Việt chuẩn).

## Phát hiện quan trọng (làm rõ tiền đề câu hỏi glossary katakana)

Đào 64 lỗi thật cho thấy: **vấn đề katakana chủ yếu là lỗi NGHĨA / false-friend, KHÔNG phải
lỗi register (giữ-Anh-hay-không).** Các ca nguy hiểm nhất là chọn SAI nghĩa tiếng Việt:

- `ケース` → v7a dịch "trường hợp" (case=tình huống) trong khi ngữ cảnh là **ốp lưng** điện thoại
- `フォロー` → "đỡ đần" trong khi nghĩa MXH là **theo dõi/follow**
- `デフレ` → "lạm phát" trong khi đúng là **giảm phát** (ngược hẳn — acc0)
- `コースメニュー` → "khóa học" (course) trong khi là **thực đơn set**

Việc "giữ katakana ra tiếng Anh cho hợp lý" thực ra chỉ áp cho **3-4 từ ranh giới**
(smartwatch/app/backup — tiếng Việt IT vẫn hay giữ Anh). Phần lớn katakana trong 64 lỗi
phải render thành **từ Việt chuẩn** (đăng nhập, theo dõi, làm việc từ xa, sạc không dây) —
giữ Anh quá tay sẽ làm câu kém tự nhiên với người đọc phổ thông, kéo tụt chính guardrail nat.

## Cách phân loại 64 lỗi → đi đâu

| Nhóm | Số | Bản chất | Xử lý |
|---|---|---|---|
| **A. Term/nghĩa lexical** | ~27 | 1 từ JA → sai nghĩa VI | **→ Glossary (Bảng A)** |
| **B. Thành ngữ/tục ngữ** | ~9 | idiom dịch mặt chữ | **→ Glossary (Bảng B)** |
| **C. Keigo/cụm thương mại** | ~4 | cụm cố định | → V8 §T2a (keigo ~400 cụm) + 1 phần vào glossary |
| **D. Ngữ pháp / phủ định / thì** | ~13 | đảo cực, sai thì, câu hỏi tu từ | **KHÔNG vào glossary** → V8 §T2b (đa nghĩa/scope phủ định) |
| **E. Sinh rác (garbled)** | ~7 | acc0 nat0, đứt mạch | **KHÔNG sửa bằng data thuật ngữ** → cần chẩn đoán riêng |

Chỉ **A + B** thành glossary. D/E ghi rõ để không nhầm là "glossary sửa được".

---

## Bảng A — Term/nghĩa lexical (→ glossary, cột `ja,vi`)

| JA | VI canonical | v7a dịch SAI | Ghi chú | id |
|---|---|---|---|---|
| 折り返す | gọi lại (điện thoại) | "quay lại" | idiom điện thoại business | 37 |
| 査収 | xem xét và nhận (tài liệu kèm) | (bỏ, thay keigo chung) | keigo thương mại | 207 |
| 経費精算 | quyết toán chi phí | "thanh toán chi phí" | 経費精算を出す=nộp quyết toán | 226 |
| 指摘 | chỉ ra / nhắc nhở | "chỉ thị" | KHÁC 指示 (chỉ thị) | 241 |
| 筋がいい | có tố chất / có triển vọng | "chuyên nghiệp" | | 274 |
| 内定 | trúng tuyển / được offer | "nhận nội bộ" | 内定 tuyển dụng | 286 |
| 白紙になる | bị hủy bỏ / trở về vạch xuất phát | "để trống" | idiom, không dịch mặt chữ | 292 |
| 出張報告書 | báo cáo công tác | "báo cáo lưu động" | | 305 |
| 息が白くなる | hơi thở thành khói trắng (vì lạnh) | "thở không ra hơi" | | 98 |
| 一夜漬け | học nhồi/học tủ cấp tốc một đêm | "ngâm mình một đêm" | idiom | 123 |
| 虹 | cầu vồng | "tia hy vọng" | garbled — pin cứng | 163 |
| 乗り換え | chuyển tàu / đổi tuyến | "chuyển đổi" (mơ hồ) | giao thông | 83 |
| 十分 (じゅっぷん) | mười phút | "vài phút" | đọc số, không phải 十分=đầy đủ | 88 |
| 懐かしい | nhớ nhung / hoài niệm | "nhớ mang máng" | sai nuance | 508 |
| 機嫌がいい | tâm trạng tốt | "ghét máy móc" | garbled | 546 |
| のどが渇く | khát nước | "nước sôi họng" | garbled | 581 |
| 立ち直る | gượng dậy / hồi phục | "đứng dậy" | | 634 |
| だしを取る | nấu / lấy nước dùng | "ăn nước dùng" | ẩm thực | 821 |
| おかず | món ăn kèm | (mất ý) | | 828 |
| 辛い (からい) | cay | "khổ" (đọc tsurai) | phân biệt karai/tsurai theo ngữ cảnh 食 | 843 |
| 替え玉 | thêm (một) vắt mì | "mì thay thế" | văn hoá ramen | 845 |
| 下げる (bàn ăn) | dọn đi (đĩa) | "hạ lên bàn" | nghĩa ngữ cảnh nhà hàng | 855 |
| 春一番 | cơn gió xuân đầu mùa | (dịch lệch) | thuật ngữ mùa | 917 |
| どちら (hỏi nơi) | phía/hướng nào | "cái nào" | 駅はどちら=ga ở hướng nào | 940 |
| 市場 (いちば) | chợ | "thị trường" | phân biệt ichiba/shijō | 983 |
| 過疎 | thưa dân / vắng dân | "đông dân" (Google) | econ, NGƯỢC cực | 1050 |
| 生活の足 | phương tiện đi lại | "sinh kế" / "chân" | idiom | 1050 |
| 流出 (nhân lực) | chảy máu chất xám / di cư | "sự cố tràn lan" | garbled | 1025 |

---

## Bảng B — Thành ngữ / tục ngữ (→ glossary, dịch NGHĨA không mặt chữ)

| JA | VI canonical (nghĩa) | v7a dịch SAI | id |
|---|---|---|---|
| 灯台下暗し | ngay dưới chân đèn lại tối (cái gần nhất dễ bị bỏ qua) | "thế gian nan" | 1002 |
| 玉石混交 | vàng thau lẫn lộn | "thịt thạch lẫn lộn" | 1077 |
| 焼け石に水 | như muối bỏ bể / nước đổ lá khoai (vô ích) | "nước chảy đá mòn" (NGƯỢC) | 1091 |
| 捨てる神あれば拾う神あり | cửa này đóng thì cửa khác mở | "có thần thì có thần" | 1198 |
| 木を見て森を見ず | thấy cây mà chẳng thấy rừng | "coi thường cây cối" | 1027 |
| 郷に従う (郷に入っては〜) | nhập gia tùy tục | "thuận theo hương tộc" | 1148 |
| 看板だけに終わる | hữu danh vô thực / chỉ có hình thức | "dừng ở biển hiệu" | 1104 |
| 別腹 | (no rồi) vẫn còn bụng cho tráng miệng | "bụng riêng" (mặt chữ) | 852 |
| 骨が折れる* | vất vả cực nhọc | — (đã có trong pool cũ) | — |

\* dòng cuối lấy từ pool inject cũ để giữ format nhất quán.

---

## Bảng A-KATA — Katakana ⚠️ CẦN BẠN DUYỆT REGISTER

Cột **Đề xuất** = khuyến nghị của tôi; cột **Giữ Anh?** = có nên giữ nguyên tiếng Anh không.
Nguyên tắc: audience bench là **phổ thông** → mặc định từ Việt chuẩn; chỉ giữ Anh cho từ
tiếng Việt IT thực sự hay giữ, hoặc loanword đã Việt hoá hẳn.

| Katakana | Đề xuất (canonical) | Giữ Anh? | Ghi chú | id |
|---|---|---|---|---|
| デフレ | **giảm phát** | ✗ | thuật ngữ kinh tế, KHÔNG được để "deflation" mơ hồ; sai = acc0 | 1085 |
| ケース (điện thoại) | **ốp lưng** | ✗ | false-friend "trường hợp"; nghĩa case=ốp | 694 |
| フォロー (MXH) | **theo dõi** | △ | "follow" chấp nhận với audience trẻ/MXH | 676 |
| コースメニュー | **thực đơn set** / set menu | △ | KHÔNG "khóa học" | 853 |
| ガジェット | **thiết bị (công nghệ)** | △ | "gadget" ok với audience IT | 695 |
| ログインボーナス | **thưởng đăng nhập** | △ | game; "login bonus" ok với gamer | 669 |
| バックアップ | **sao lưu** | △ | "backup" phổ biến trong IT — chấp nhận cả hai | 634 |
| スマートウォッチ | **đồng hồ thông minh** | △ | "smartwatch" ok với IT | 693 |
| テレワーク | **làm việc từ xa** | ✗ | | 733 |
| ワイヤレス充電 | **sạc không dây** | ✗ | | 694 |
| マイク | **micro** | ✓ | loanword Việt hoá | 731 |
| アプリ | **ứng dụng** | △ | "app" rất phổ biến | 726 |
| カラオケ | **karaoke** | ✓ | Việt hoá hẳn | 553 |
| ラーメン | **ramen** | ✓ | tên món, giữ | 845 |

**Quy tắc rút ra để mã hoá vào glossary:** `✗` = luôn dùng VI (pin cứng), `✓` = luôn giữ
nguyên, `△` = tuỳ audience — nếu KD có tag miền IT thì cho phép biến thể Anh, còn lại dùng VI.

---

## Nhóm D — Ngữ pháp/phủ định/thì (KHÔNG vào glossary → V8 §T2b)

Đây là lỗi **cấu trúc**, không phải từ vựng; glossary vô dụng ở đây. Thuộc nhánh
"đa nghĩa/scope phủ định" và "aspect" của PLAN_V8 §T2b:

- 思わない? / じゃないか — câu hỏi tu từ bị đọc thành phủ định phẳng (id124, 339) → **đảo cực, acc0**
- 非婚化 — rơi tiền tố 非, dịch ngược "ngày càng kết hôn" (id1087) → **acc0**
- 入れないで — mệnh lệnh phủ định "đừng cho vào" → dịch "cho vào" (id846) → **acc0**
- 聞こえなくなった — mất khả năng nghe → dịch "nghe thấy" (id731) → **acc0**
- 雨も降ってきた — mưa BẮT ĐẦU rơi → "mưa tạnh" (id989) → **acc0**
- 賞味期限が切れている — ĐÃ hết hạn → "sắp hết hạn" (id837); 始まった đã bắt đầu → "sắp" (id669) — **sai thì**
- 充電の減りが早い — pin TỤT nhanh → "sạc nhanh" (id693) — **đảo cực nghĩa**
- 干してくれない? — nhờ "phơi giúp" → "sắp phơi chưa" (id411) — **sai form nhờ vả**

→ Cần **minimal-pair phủ định/aspect** (đúng nhánh (b) của V8), không phải từ điển.

## Nhóm E — Sinh rác / đứt mạch (garbled, acc0 nat0 → CHẨN ĐOÁN riêng)

Không sửa được bằng glossary lẫn data thuật ngữ đơn thuần — coherence breakdown:

- id546 "trông ghét máy móc" · id581 "nước sôi họng" · id767 "dùng điều hòa để bật điều hòa"
- id1077 "thịt thạch lẫn lộn" · id553 "chẳng ai thèm lịch" · id855 "hạ lên bàn"

→ Trùng nhánh (f) STT-fragment của V8 về hiện tượng (model bịa khi mất mạch), nhưng các câu
này là câu HOÀN CHỈNH → nghi vấn khác: decoding/rep-penalty hoặc lỗ hổng capacity. **Cần đo
riêng** (ví dụ chạy lại chính các câu này ở nhiệt độ/beam khác) trước khi kết luận là "thiếu data".

---

## Đối chiếu với PLAN_V8 (2 điểm cần bạn chốt)

1. **Thầy dịch:** V8 §T4 chốt **Gemini Live (6 key, gần miễn phí)** và §T3 sẽ audit thầy đó.
   Ở turn trước tôi khuyến nghị **Sonnet 5** (có phí ~$25-45 cho 30k) vì tiếng Việt tự nhiên hơn
   — đúng để giữ guardrail nat. Đây là **fork thật cần bạn quyết**. Glossary này teacher-agnostic
   nên KHÔNG chặn; nhưng trước khi bung KD phải chốt để chạy audit-300 đúng thầy.
2. **Tiền đề T0 (chuyển data laptop→Máy A):** corpus 11.88M + `glossary_merged.csv` CHƯA có trên
   Máy A → mine bucket ③ và `measure_term_coverage.py` chưa chạy được ở đây. Cần làm T0 trước.

## Merge vào pipeline (khi bạn duyệt xong)

1. File kèm: `eval/glossary_v8_candidates.csv` (cột `ja,vi,cat,register,wrong,id`).
2. Duyệt cột `register` ở Bảng A-KATA (đổi △ thành ✗/✓ theo audience đích).
3. Rút gọn còn `ja,vi` → nối vào `data/glossary/glossary_merged.csv` (sau khi T0 mang file về).
4. Chạy `measure_term_coverage.py` → biết term nào còn thiếu trong corpus → sinh KD đúng chỗ.
5. `gen_glossary_inject.py` sinh mẫu `[term=訳語]` cho các term mới.
