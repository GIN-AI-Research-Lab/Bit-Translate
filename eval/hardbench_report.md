# HARDBENCH 200 — 110M step23000 vs Google Translate vs Claude (2026-07-16 tối)

Benchmark "câu khó, ngữ cảnh đa dạng" theo yêu cầu user, chạy trong lúc chờ 292M train.

## 1. Phương pháp

- **Bộ đề:** `eval/hardbench200.jsonl` — 200 câu TỰ SOẠN (không trùng corpus/FLORES), 10 domain × 2 chiều × 10 câu: `keigo` (kính ngữ business), `hoithoai` (xưng hô quan hệ công sở), `caudai` (30-45 từ, mệnh đề lồng), `nguphap` (使役受身, bị động thiệt hại, とは限らない, ~ところだった...), `thanhngu`, `slang`, `zeropronoun`, `solieu` (copy-through + 万/億), `it_deep`, `hop`. Câu nguồn + ref do Claude (Fable 5) soạn ở vai dịch giả.
- **3 hệ:** ① 110M step23000 i2_s local (llama-server, temp 0, 21s/200 câu); ② Google Translate (endpoint web gtx — **metadata cho thấy Google dịch vi↔ja bắc cầu qua tiếng Anh**: `vi_en_2023q1` + `en_ja_2023q1`); ③ Claude (subagent dịch, **không được xem ref**).
- **Chấm:** (a) chrF từng câu so ref; (b) **trọng tài Claude chấm mù**: 3 output/câu xáo thứ tự ẩn danh A/B/C từng câu, 8 giám khảo độc lập chấm 0-5 (0=vô nghĩa, 2=lỗi nghĩa lớn, 3=đúng ý chính nhưng lỗi đáng kể, 4=lỗi nhỏ, 5=chuẩn dịch giả). Mapping đã xác minh 3 mẫu khớp 100%.
- **Caveat phải nhớ khi đọc:**
  - Ref do Claude soạn → **chrF của Claude bị thổi phồng** (cùng giọng văn). So chrF chỉ nên dùng cho cặp 110M vs Google.
  - Trọng tài cùng dòng model với thí sinh Claude → điểm 5.00 tuyệt đối của Claude nên đọc là "trần trên / không tìm thấy lỗi", có thể lệch +. Nhưng khoảng cách với 2 hệ còn lại lớn tới mức bias không đổi kết luận.
  - chrF trên câu khó **đánh giá thấp khoảng cách thật**: ja→vi 110M chỉ kém Google 3.5 chrF (32.6 vs 36.1) nhưng trọng tài chấm 1.95 vs 3.60 — 110M ra đúng "từ vựng domain" nhưng sai nghĩa/ngôi, chrF không bắt được.

## 2. Kết quả tổng (n=200)

| Hệ | Judge vi→ja | Judge ja→vi | % dùng được (≥4) | chrF vi→ja | chrF ja→vi |
|---|---|---|---|---|---|
| **110M step23000** | **1.43** | **1.95** | **5.5%** (11/200) | 15.0 | 32.6 |
| Google Translate | 3.64 | 3.60 | 57.5% (115/200) | 28.3 | 36.1 |
| Claude (Fable 5) | 5.00 | 5.00 | 100% (200/200) | 60.0* | 69.9* |

\* chrF Claude thổi phồng do ref cùng nguồn — xem caveat.

- **Đối đầu 110M vs Google (điểm trọng tài):** thắng **3** / hòa 20 / thua 177.
- **Phân bố điểm 110M:** 0:5, 1:96, 2:66, 3:22, 4:11, 5:0 — nửa bộ đề bị điểm 1 (hiểu sai ý chính).
- **Phân bố điểm Google:** 1:1, 2:15, 3:69, 4:89, 5:26 — chủ yếu 3-4, "hiểu đúng nhưng thô".

## 3. Điểm trọng tài theo domain × chiều

| domain | dir | 110M | Google | Claude |
|---|---|---|---|---|
| caudai | ja2vi | 1.80 | 3.70 | 5.00 |
| caudai | vi2ja | **1.00** | 4.50 | 5.00 |
| hoithoai | ja2vi | 1.90 | 3.20 | 5.00 |
| hoithoai | vi2ja | 1.40 | 3.40 | 5.00 |
| hop | ja2vi | 2.10 | 3.60 | 5.00 |
| hop | vi2ja | 1.20 | 3.90 | 5.00 |
| it_deep | ja2vi | 1.80 | 3.50 | 5.00 |
| it_deep | vi2ja | 1.10 | 3.60 | 5.00 |
| keigo | ja2vi | 1.90 | 3.30 | 5.00 |
| keigo | vi2ja | 1.70 | 3.80 | 5.00 |
| nguphap | ja2vi | 1.60 | 3.80 | 5.00 |
| nguphap | vi2ja | 1.40 | 4.10 | 5.00 |
| slang | ja2vi | 1.90 | 3.50 | 5.00 |
| slang | vi2ja | 1.10 | **3.10** | 5.00 |
| solieu | ja2vi | **2.40** | **4.50** | 5.00 |
| solieu | vi2ja | 1.90 | 4.00 | 5.00 |
| thanhngu | ja2vi | 1.90 | 3.30 | 5.00 |
| thanhngu | vi2ja | 1.20 | **2.70** | 5.00 |
| zeropronoun | ja2vi | 2.20 | 3.60 | 5.00 |
| zeropronoun | vi2ja | **2.30** | 3.30 | 5.00 |

**Đảo mạnh nhất của 110M** (điểm cao nhất): solieu ja2vi 2.40, zeropronoun (2.20/2.30), hop ja2vi 2.10 — trùng khớp data vòng 1-2 đã nhắm (copy-through, IT/họp, câu văn phòng ngắn). **Vực sâu nhất:** caudai vi2ja 1.00, slang/it_deep vi2ja 1.10 — chiều vi→ja + câu dài là điểm chết.

**Chỗ Google yếu thật sự** (mở cửa cho model chuyên domain): thanhngu vi2ja 2.70, slang vi2ja 3.10, và các lỗi hệ thống do **bắc cầu tiếng Anh**: 議事録→"minutes"→"những phút giây"; 元も子もない mất nghĩa vì "child"; 鈴木→"Chuông"; "gấu" (người yêu)→クマ; sai ngôi zero-pronoun hàng loạt (持ち帰って検討→"Hãy mang về nhà" — ra lệnh cho người nghe).

## 4. 110M thắng/hòa Google ở đâu (23/200)

3 câu thắng đều là ja→vi giọng công sở — đúng vùng data đã train:
- id44 `hop`: 持ち帰って検討させてください → 110M: "Cái này mình sẽ về và xem xét kỹ hơn..." (đúng ngôi) vs Google: "Hãy mang vấn đề này về nhà và xem xét nó" (sai ngôi, ra lệnh ngược).
- id66 `it_deep`: ログを仕込んで → 110M: "hãy log lên rồi xem sao... Sau đó là sửa chữa" vs Google: "tải nhật ký lên... Việc cải tạo" (改修→"cải tạo" kiểu xây dựng).
- id181 `zeropronoun`: お渡ししたはずですが → 110M chọn đúng ngôi "Tôi đã đưa nó cho anh" vs Google bị động lơ lửng.

20 câu hòa: đa số ticket/version copy-through (PROJ-5678, v2.3.1 giữ nguyên chuẩn), câu văn phòng ngắn kiểu 課長、資料の確認が終わりました. **Đây chính là niche "vượt Google trong công sở IT" mà PLAN §9 đặt cược — có bằng chứng thật, nhưng mới chỉ ở câu ngắn ja→vi.**

## 5. Taxonomy lỗi 110M trên bộ khó (từ note 8 giám khảo)

1. **Vỡ ngữ pháp/garble khi câu dài** (nhất là vi→ja): output thành chuỗi từ rời rạc — caudai vi2ja 1.00 gần như sập toàn bộ.
2. **Sai ngôi/đảo chủ thể** ở zero-pronoun và kính ngữ (dù đây vẫn là domain "khá" nhất của nó).
3. **Thành ngữ/slang dịch mặt chữ hoặc bịa**: "gấu"→熊, "bó tay"→手をつないでた, "xù"→別れました.
4. **Sai hàng số** khi quy đổi 万/億, % (2割→2%, 150.000円 thay 15.000円) — copy-through mã/version thì tốt nhưng quy đổi số thì hỏng.
5. **Bỏ vế sau** của câu ghép; **để nguyên từ Việt chưa dịch** lẫn trong output Nhật.
6. **Kính ngữ dịch mặt chữ**: 恐れ入ります→"Em rất sợ".

## 6. Kết luận & hàm ý

1. **Trên câu khó đa ngữ cảnh, 110M chưa dùng được** (5.5% đạt ≥4; Google 57.5%; Claude ~100%). Khoảng cách với Google là ~1.9 điểm judge — lớn hơn nhiều so với cảm giác từ chrF FLORES (vốn chỉ kém 12 điểm ja→vi). Củng cố kết luận trần sức chứa (4/4 tín hiệu) → quyết định scale 292M là đúng hướng.
2. **chrF một mình không đủ để đánh giá trên câu khó** — nên giữ hardbench200 + judge làm thước bổ sung: chạy lại y hệt cho 292M (`run_hardbench.py` + trọng tài) để so trực tiếp với bảng này. Gate 292M nên nhìn cả judge trên hardbench, không chỉ chrF FLORES.
3. **Niche khả thi đã có bằng chứng**: câu công sở IT ja→vi ngắn (thắng/hòa Google 23 câu, toàn vùng này). Chiều vi→ja là điểm chết toàn diện → đúng như kế hoạch, **back-translation dồn cho chiều vi→ja / data JA** (kế hoạch 500k-1M câu, mới dùng 86k) là đòn tiếp theo giá trị nhất về data.
4. Data vòng 3 nên nhắm: câu dài vi→ja (sập nặng nhất), quy đổi số 万/億/%, thành ngữ + khẩu ngữ (cả Google cũng yếu — cơ hội khác biệt hóa), kính ngữ mặt chữ.

## 7. Chạy lại / tái lập

```bash
# model local (đổi gguf khi có 292M):
LLAMA_SERVER=$HOME/BitNet/build/bin/llama-server \
  ~/BitNet/.venv_bitnet/bin/python eval/run_hardbench.py dist/vija_23000_i2s.gguf step23000

# google (gtx, có resume):
~/BitNet/.venv_bitnet/bin/python scripts/google_translate_bench.py

# file kết quả từng câu (đọc bằng mắt): eval/hardbench_{step23000,google,claude}.jsonl
# điểm trọng tài + note: eval/hardbench_scores.json
```

Lưu ý kỹ thuật: trong `run_hardbench.py` phải dùng `pkill -x llama-server` (KHÔNG `-f` — tự khớp cmdline cha chứa `LLAMA_SERVER=.../llama-server` và tự giết mình, exit 144).

## 8. Bổ sung: câu NGẮN thì sao? (phân tích theo yêu cầu user)

**(a) Hardbench chia tercile độ dài nguồn** (điểm trọng tài 110M | Google):

| chiều | NGẮN | VỪA | DÀI |
|---|---|---|---|
| vi→ja (9-18 / 18-23 / 23-45 từ) | 1.67 \| 3.45 | 1.45 \| 3.39 | **1.18 \| 4.06** |
| ja→vi (22-33 / 33-43 / 43-85 ký tự) | 2.06 \| 3.82 | 1.94 \| 3.39 | 1.85 \| 3.59 |

- vi→ja của 110M **tụt dần theo độ dài** (1.67→1.18) trong khi Google **càng dài càng tốt** (4.06 — câu dài formal là sở trường NMT lớn). Toàn bộ 3 câu vi→ja đạt ≥4 của 110M đều nằm ở tercile ngắn; 23 câu thắng/hòa Google có độ dài vi→ja trung bình 13.4 từ (vs 21.8 toàn bộ).
- ja→vi ít nhạy độ dài (2.06→1.85): cái sập ở ja→vi là nội dung khó (thành ngữ/kính ngữ), không phải độ dài.

**(b) Probe64 — 64 câu ngắn công sở "độ khó thường"** (chrF, Google + Claude chạy 2026-07-16 tối; ref probe64 có sẵn từ trước):

| hệ | vi→ja | ja→vi |
|---|---|---|
| 110M step23000 | 31.7 | **48.7** |
| Google | 50.4 | **52.2** |
| Claude | 76.1 | 73.0 |

- **Câu ngắn thường, chiều ja→vi: 110M bám sát Google (kém 3.5 chrF), thắng 19/64 câu đối đầu, và THẮNG hẳn domain hội thoại (56.5 vs 52.3)** — niche "công sở IT ja→vi" là có thật.
- Chiều vi→ja vẫn thua xa cả trên câu ngắn (31.7 vs 50.4) — điểm yếu chiều, không phải độ dài.
- File: `eval/hardbench_probe64_google.jsonl`, `eval/probe64_claude.jsonl`.

**Kết luận câu ngắn:** ngắn + domain quen (công sở/IT, ja→vi) → cạnh tranh được với Google. Ngắn nhưng nội dung khó (slang/thành ngữ/kính ngữ) → vẫn thua rõ, vì đó là lỗi THIẾU TRI THỨC trong data, không phải lỗi độ dài.

## 9. Vì sao thua Google và Claude (chẩn đoán)

**Thua Google — 4 nguyên nhân xếp theo trọng số:**
1. **Data kém ~3 bậc độ lớn**: 5.4M cặp (nặng OpenSubtitles nhiễu + synthetic hẹp) vs hàng tỷ cặp đa domain; Google còn dùng EN pivot với kho vi-en/en-ja khổng lồ.
2. **Trần sức chứa 110M** (4/4 tín hiệu đã bật): thành ngữ/slang/kính ngữ là tri thức phải "nhớ" từng mapping — model nhỏ không có chỗ chứa. Lỗi "gấu"→熊, 恐れ入ります→"Em rất sợ" là lỗi thiếu tri thức, không phải thiếu khả năng dịch.
3. **Phân bố data lệch về casual/IT/họp** → làm tốt đúng vùng đó (thắng Google hội thoại ja→vi probe64) và sập ở vùng trống (kính ngữ đa tầng, thành ngữ, câu dài formal, quy đổi 万/億).
4. **Chiều vi→ja yếu cấu trúc**: sinh tiếng Nhật khó hơn (chọn kanji, trợ từ, tầng kính ngữ) + JA-side data ít đa dạng; asymmetry nhất quán từ FLORES (21 vs 42) tới hardbench (1.43 vs 1.95). Câu >23 từ vượt vùng phân bố train (p99 147 token nhưng câu dài phức hiếm) → garble.

**Thua Claude — khác hạng cân:** frontier LLM có tri thức thế giới + ngữ dụng (biết "gấu" là slang người yêu, 議事録 là biên bản, suy ngôi zero-pronoun bằng lẽ thường, chọn xưng hô theo quan hệ). Lợi thế này NMT thuần cũng không có — Claude thắng cả Google 187/200. (Caveat: trọng tài cùng dòng model, 5.00 tuyệt đối có thể thổi nhẹ, không đổi kết luận.)

**KHÔNG phải thủ phạm chính: 1.58-bit.** Ternary có trả giá nhỏ so FP16 cùng size, nhưng vực với Google là vực data+scale — bằng chứng là vùng có data thì model bám/thắng Google bất chấp 1.58-bit. Hàm ý giữ nguyên chiến lược: scale 292M (đang train) + BT dồn JA + data vòng 3 nhắm lỗ hổng tri thức (thành ngữ, 万/億, keigo, câu dài vi→ja).

## 10. FLORES n=100 — 3 hệ trên thước TRUNG LẬP (2026-07-16 tối, bổ sung)

FLORES là thước công bằng nhất trong mọi eval ở đây: ref do dịch giả chuyên nghiệp làm sẵn từ trước (KHÔNG phải Claude soạn như hardbench/probe64) — nên đây là con số nên tin nhất khi so 3 hệ bằng chrF.

| Hệ | vi→ja | ja→vi |
|---|---|---|
| **110M step23000** | 21.5 | 42.3 |
| Google | 42.6 | 53.5 |
| Claude | 40.2 | 54.1 |

Quan sát: (1) khoảng cách 110M↔Google gần như y hệt cả 2 chiều tính theo TỈ LỆ (~2x cả hai chiều dù chrF tuyệt đối khác nhau — literary/formal style của FLORES bất lợi cho model nhỏ hơn câu công sở ngắn của probe64); (2) trên văn phong tin tức/wiki trang trọng này, **Google nhỉnh hơn Claude ở vi→ja** (42.6 vs 40.2) — khác hẳn hardbench nơi Claude áp đảo tuyệt đối, vì FLORES chấm sát ref chữ-đối-chữ hơn còn hardbench có nhiều câu đòi hỏi tri thức ngữ dụng (nơi Claude ăn đứt). File: `eval/flores_claude.jsonl`, `eval/hardbench_flores_google.jsonl`.
