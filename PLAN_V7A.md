# PLAN V7A — nâng MỌI hạng mục chất lượng; gate trần scale trước, đủ điều kiện thì grow 24L

> Viết 2026-07-29, sau khi: fix tokenizer GGUF (deploy = PyTorch, p≈1), 3 bench mới
> (rand/opus100/chấm kép), cô lập 2 lớp lỗi còn lại. Kế thừa ISSUES.md #1, bác bỏ
> hướng ghép-ngẫu-nhiên và tách-câu (đã đo, chết).

## 0. Mốc xuất phát và đích (đo trên BỘ BA bench, chấm mù cùng phiên)

| thước | hiện tại | đích V7A |
|---|---|---|
| OOD (opus100) TỔNG | 84% (Claude-judge) | ≥ 90% |
| OOD câu ngắn | 96% | giữ ≥ 96% (sát trần thầy) |
| **OOD câu dài** | **72%** (Google 78%) | **≥ 80%** |
| rand in-domain | 74-76% | ≥ 80% |
| TED-style (bench cũ) | 74% (Google 86%) | ≥ 80% |
| Hồi quy cấm | keigo/number/negation không tụt >3đ | |

Hai lớp lỗi đã cô lập (chấm mù 400 bản dịch):
- **Lớp A — kiến thức**: từ ghép văn hoá hiếm (弾丸登山, あめ色, 過怠税, 会席料理,
  支払いサイト). Chữa bằng data nhắm trúng, KHÔNG cần scale.
- **Lớp B — năng lực**: lật nghĩa/chủ thể ở câu đa mệnh đề (押されていた→"đã đẩy được").
  3 bằng chứng trần: 6 vòng data không nhích clause; oracle rerank +5,2 chrF;
  81% câu hỏng ≥2 nguyên nhân. **Đây là ứng viên trần scale.**

## 1. BƯỚC MỘT — gate "trần scale chưa" (làm TRƯỚC mọi thứ, ~$4, 2 ngày)

1. **Lọc 558 câu dev rò** (12,56% trùng train — nợ từ 2026-07-28), dựng lại dev sạch.
2. Chuẩn bị data thử nhỏ: 150–300k cặp CÂU DÀI THẬT (mục §3a) + 100k lexicon (§2).
3. **Train thử 4.000 step trên 18L hiện tại** (LR-restart 5e-5, `--dev-every 500` dev sạch).
4. Chấm long-OOD + clause cùng phiên với baseline.

**Luật quyết định:**
```
long-OOD ≥ 78% (đạt mức Google)   → CHƯA trần → chạy trọn V7A trên 18L, hoãn scale
long-OOD < 75% VÀ dev-loss vẫn giảm → data chưa đủ liều → thêm liều, thử lại 1 lần
long-OOD < 75% VÀ dev-loss phẳng   → TRẦN XÁC NHẬN → grow 24L (§4) rồi chạy trọn V7A
```

## 2. Nhánh LEXICON — nâng cả ngắn lẫn dài, không cần scale (~$0 quota, $3 GPU)

Công thức đã thắng ở chiến dịch katakana 9.000 từ, áp cho các lớp từ ĐÓNG:
- Quán ngữ + thành ngữ 4 chữ (四字熟語) + từ ghép văn hoá (danh sách công khai, ~5-8k mục)
- Thuật ngữ ghép kanji chuyên ngành đuôi thấp (税/料/法/式 compounds)
- Mỗi mục: mine câu chứa từ trong CC-100/kokkai (neo đuôi biến cách — tránh bẫy regex
  2 ký tự cũ) → KD Live API. ~200-400k câu.
- **Nâng thầy cho tập này**: audit 200 câu thầy hiện tại (flash-live/audio — CHƯA từng
  audit); nếu <4,5/5 thì tập lexicon + câu dài dùng thầy mạnh hơn (pro/flash REST xoay
  ngày, chỉ vài trăm k câu nên quota chịu được).

## 3. Nhánh CÂU DÀI — data đúng loại + thủ thuật không-cần-data-cực-dài

### 3a. Data (nguồn có sẵn, chưa từng khai thác đúng cách)
- **Đào theo ĐỘ DÀI** 120–250 ký tự từ kokkai (7,9% ≥150 ký tự = 158k+/2M dòng) +
  CC-100. Dedup train. (ISSUES #1 hướng 2 — "cách duy nhất dạy cấu trúc lồng thật".)
- **Ghép câu LIỀN KỀ cùng văn bản** (không phải ghép ngẫu nhiên — đã bác): kokkai là
  diễn văn liên tục, lấy 2-3 câu liền nhau làm một mẫu → ngữ cảnh thật, đại từ thật,
  100-180 token — không cần "cực dài". Thầy dịch cả đoạn.
- Tái dùng format `ctx|||src` đã có từ vòng 3a (50k mẫu): câu trước làm ngữ cảnh,
  chỉ chấm loss trên bản dịch câu sau — dạy zero-pronoun/mạch văn với chi phí câu NGẮN.

### 3b. Thủ thuật $0 data (đổi cách TRAIN, không đổi corpus)
- **Oversample đuôi dài có sẵn**: corpus đã có 4,8% chuỗi ≥160 token — weighted
  sampling nâng vùng ≥128 lên ~15-20% token-budget mỗi batch. Một tham số trong
  make_batches. Đây là đòn rẻ nhất đúng nghĩa "không cần truyền data cực dài".
- **Loss weighting theo vị trí** (thí nghiệm, 1 dòng): nhân trọng số CE ~1,3× cho
  token vị trí >96 — ép model học kỹ vùng đuôi. Gate bằng dev sạch, bỏ nếu không ăn.
- **⚠️ KHÔNG làm position-offset augmentation**: RoPE là tương đối (score chỉ phụ
  thuộc hiệu m−n) → offset câu ngắn sang vị trí cao KHÔNG dạy được attention span dài.
  Ghi để không ai đề xuất lại.
- **KHÔNG lặp lại**: tách câu inference (đo rồi, +0,1 chrF), ghép cặp ngẫu nhiên
  (32% tràn 256, dạy sai prior), beam/rerank (oracle trần +5,2).

## 4. GROW 24L → ~200M (chỉ khi gate §1 xác nhận trần)

- **Cách**: chèn 6 block identity xen kẽ vào 18L (thủ thuật đã thắng ở 12→18: rẻ hơn
  train lại 3×, chất lượng giữ nguyên tại thời điểm chèn). 24L × d768 ≈ 195-205M.
  i2_s ≈ 100-105MB — vẫn CPU thoải mái (~280-300 tok/s ước từ tỉ lệ băng thông).
- **Train**: 20-25k step trên corpus V7A ≈ $25-35 (đơn giá đo $1/1000 step, nặng hơn
  ~30%/step). Milestone 250 step + KEEP=12 + curve_anchor như v6.
- **Context 256 → 384 khi grow — CÓ, với điều kiện**: bảng RoPE là precompute, nới
  KHÔNG thêm tham số; chi phí thật là attention O(n²) lúc train (chấp nhận được ở 384)
  và **phải có data lấp vùng 256-384** — chính là mẫu ghép-liền-kề §3a (100-180 token/câu
  ghép 2-3 câu = vừa vùng đó). KHÔNG nhảy thẳng 512: vùng 160-256 hiện còn 4,8% phủ,
  nới quá xa chỉ mở rộng vùng chết (bài học ISSUES #1 hướng 4). 384 đủ dịch đoạn
  ~300 ký tự Nhật — gấp rưỡi trần hiện tại.
- Binarize: nâng ngưỡng bỏ câu >256 thành >384 tương ứng.

## 5. Trình tự + ngân sách

| bước | việc | tiền | ngày |
|---|---|---|---|
| 1 | dev sạch + mine lexicon/câu-dài + audit thầy | $0 | 2 |
| 2 | KD Live (~400-700k câu) | $0 (quota) | 1 |
| 3 | **GATE trần**: 4k step 18L + chấm | ~$4 | 1 |
| 4a | chưa trần: +12k step 18L đủ liều | ~$12 | 1-2 |
| 4b | trần: grow 24L (+384 ctx) + 20-25k step | ~$25-35 | 2-3 |
| 5 | avg7 + bộ ba bench chấm kép + GGUF i2_s qua đường ids | $0 | 1 |

Tổng: **$16 (nhánh 18L) hoặc $30-40 (nhánh 24L)** — trong tầm credit tritue12 + nạp nhỏ.
Deploy cuối: convert bằng pipeline đã chứng minh đúng; bench GGUF bắt buộc qua
llama-server + token ids (ISSUES #3/#4); `-t 8`; so PyTorch phải p>0,1 mới phát hành.

## 6. Trả lời gọn các câu chiến lược (để phiên sau khỏi hỏi lại)

- **"Đã trần chưa?"** — Lớp A chưa trần (chữa bằng lexicon). Lớp B nhiều khả năng trần
  nhưng CHƯA CHỨNG MINH SẠCH (6 vòng trước đo bằng thước nhiễu). Gate §1 là phép chứng
  minh rẻ nhất: $4 và 2 ngày, đo bằng thước đã hiệu chuẩn.
- **"200M có tăng context không?"** — Có, 256→384, vì đi kèm nguồn data vừa vặn
  (ghép liền kề); không nhảy 512 khi vùng cũ còn trống.
- **"Thủ thuật câu dài không cần data cực dài?"** — oversample đuôi có sẵn + ghép
  liền kề 2-3 câu + ctx||| + loss-weight vị trí. Offset-RoPE vô ích (tương đối),
  tách câu inference đã chết, đừng lặp.
