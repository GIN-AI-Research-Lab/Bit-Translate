# PLAN V8 — Trọng tâm: Data + Train 18L ($0-2, 7-10 ngày); kèm vệ sinh deploy ($0, 0,5-1 ngày)

> Tổng hợp 2026-07-30 từ 3 kế hoạch độc lập (data-first / scale-first / product-first),
> mỗi cái đã qua 1 agent phản biện kiểm từng khẳng định với file repo. Các con số dưới đây
> đều đã kiểm nguồn; chỗ nào phản biện bác thì đã sửa (ghi chú ⚠).
> Bối cảnh: gate V7A kết luận **18L CHƯA TRẦN** (long-OOD 80% ≥ 78%, dev −0,011/1000 cuối gate)
> → KHÔNG grow 24L vòng này. Modal $0 → train local 3060 Ti (máy A). Gói deploy đã đóng
> (`D:/Bit-Translate-data/dist/v7a_deploy_pkg/`), app teams-caption-translator đã tích hợp.
>
> **Bổ sung 30/7 chiều — từ transcript app THẬT** (song-ngu-2026-07-30.txt, 24 dòng họp thật):
> 3 lớp lỗi mới có bằng chứng, **xử lý thuần bằng DATA/TRAIN (Track T), không vá ở tầng app**
> (quyết định user 30/7: đây là vấn đề năng lực model): (1) **câu cụt kiểu STT → model bịa vị
> ngữ/đảo vai** (nặng nhất) → T2f là tuyến phòng thủ DUY NHẤT — model phải tự dịch lửng đúng,
> không dựa app gộp câu; (2) đa nghĩa やる + scope phủ định だけじゃなく → T2b; (3) tên riêng/từ
> nghe nhầm là lỗi STT thuần (ずん, 金質↔品質…) — ngoài phạm vi model, ghi nhận nhưng không xử lý
> trong plan này (T2g là lựa chọn thí nghiệm duy nhất chạm tới nó). Tham khảo: ~14/24 dòng dịch
> ổn với input nhận được, 5 lỗi thuộc model dịch.

## Track D — Vệ sinh tầng deploy của repo ($0, 0,5-1 ngày) — KHÔNG phải cải thiện model

> Phạm vi thu hẹp 30/7 theo quyết định user: chất lượng phải đến từ MODEL (Track T), không
> vá ở tầng tool. Track D chỉ giữ 2 việc thuộc chính repo này — bảo đảm model được ship
> ĐÚNG chất lượng đã đo, không thêm lớp che lỗi nào.

| # | Việc | Chi tiết + điều kiện nghiệm thu |
|---|------|--------------------------------|
| D1 | **Sửa/deprecate `demo/server.py`** | Đang dính 4 lỗi thật (phản biện kiểm từng dòng) làm ship DƯỚI chất lượng model: gọi `/tokenize` của llama.cpp (đúng bug ISSUES #4, −16 điểm use đo thật), `repeat_penalty 1.25` (bench chuẩn 1.0), `n_predict 96` cứng (cắt câu dài), gửi token `>>fix<<`=32000 mà lineage v7a **chưa từng train** (HANDOFF:387). Chuẩn duy nhất: `v7a_deploy_pkg/src/translate.py`. Nghiệm thu: 200 câu **`eval/bench_opus200b.jsonl`** (⚠ GHI CỨNG — translate_bench mặc định trỏ nhầm bench_new.jsonl) qua đường demo-đã-sửa vs đường bench, McNemar **p>0,1** cùng phiên. |
| D2 | **Cấu hình tốc độ đã đo** | `-t 8` (353 vs 301 tok/s @4 — TONGKET_V6 §4; các client đang -t 4 → đổi), hàng đợi TUẦN TỰ (2 job song song chậm 5,6× — đo thật), ghi chú cắm sạc. Chỉ là áp số đã đo, không đổi hành vi dịch. |

**Đã LOẠI khỏi plan model** (đồ tầng tool, ai làm tool thì tự cân nhắc, không tính vào chất
lượng model): bảng kính ngữ tra cứu (trùng vai T2a — kính ngữ phải do model HỌC), cờ
confidence/fallback, stream UI, glossary/name-glossary phía app.

## Track T — Data + train tiếp 18L ($0-2, 7-10 ngày)

**T0 (0,5 ngày — ⚠ thiếu trong plan gốc)**: chuyển bin + checkpoint (~2,5-4GB) từ laptop
sang máy A, kiểm env WSL2/torch/compile ở đó.

**T1 — Dựng bench nghiệm thu TRƯỚC khi có data** (bài học "bench đo sai miền"):
keigo-letter ~100 câu mine từ thư thương mại thật held-out (ĐỘC LẬP pipeline sinh — né bẫy
vòng 3); polysemy-probe 150 câu nghĩa phụ từ CC-100 thật; in-domain **n≥100/miền**;
**fragment-probe 60-100 mảnh câu cụt** (cắt câu thật tại ranh giới trợ từ, held-out — tiêu chí
đếm VỊ NGỮ BỊA THÊM, không phải chrF). Đo baseline v7a + Google cùng phiên, lưu làm mốc.

**T2 — Data 4 nhánh (+1 tuỳ chọn)**:
- (a) **Kính ngữ thư tín ~400 cụm đóng** (頭語/結語, 時候の挨拶, ご清栄/ご査収/お納め…, sonkeigo/kenjougo
  N2-N1): Gemini enumerate 500 → **đếm CC-100 phán quyết** giữ ~400; mine exact-string ≤60 câu/cụm
  + sinh bù cụm hiếm ×3 oversample ≈ 30k câu. ⚠ Mục tiêu là PHỦ CỤM ≥60 lần/cụm, không phải kéo
  mật độ keigo tổng (đã 4% từ v5 mà vẫn fail cụm thư tín — "4% = ngang Google" chỉ là giả thuyết).
- (b) **Đa nghĩa chia THEO NGHĨA** (khác vòng 6: 512 từ×200 câu KHÔNG chia nghĩa → 当たる vẫn fail):
  ~200-250 động từ JMdict ≥3 nghĩa × 70 câu/NGHĨA-PHỤ ≈ 50-60k; kèm aspect-suffix lật nghĩa
  (〜かける suýt/dở, 〜そびれる, 〜きれない…). **Stop-rule**: probe không nhích ≥+8 điểm → dừng
  loại liều này vĩnh viễn (chồng lớp B, có rủi ro không ăn — 3 bằng chứng trần đã có).
  ⚠ Bổ sung từ transcript 07-30: **やる** vào top danh sách (ブリッチをやっております → "chơi britch"
  — cùng dòng ghi âm やっております chỗ khác lại dịch đúng "làm việc" ⇒ lỗi theo ngữ cảnh, đúng dạng
  chia-theo-nghĩa); thêm nhóm **mẫu scope phủ định**: XだけじゃなくY / 〜だけでなく = "không CHỈ X
  mà còn Y" (transcript dịch thành "không phải X" → lật nghĩa), kèm họ hàng 〜ばかりか/〜のみならず.
- (c) **Câu dài**: mine kokkai/CC-100 120-250 ký tự + ghép LIỀN KỀ ≤256 token (PLAN_V7A §3a cho
  phép tường minh). Đồng thời xuất **tập 256-384 (~250-300k mẫu) ĐỂ DÀNH cho grow** — KD luôn
  trong đợt này (phòng quota đổi), KHÔNG trộn vào bin_v8.
- (d) **Miền đỏ: SINH thay mine** (⚠ phản biện bác tiền đề "đào CC-100 lần 2": kho đã quét TOÀN BỘ
  2 lượt ngày 07-28 cho danh sách 54 miền, 2,2M câu đã tiêu thụ — mine lại ra ~0; 4.073 từ còn thiếu
  "chỉ có thể SINH" theo DOMAIN_MAP §4). Giữ NHỎ: thuật ngữ chỉ ~3% lỗi taxonomy. Mine mới thật sự
  chỉ còn: 6-7 miền văn hoá mới + cụm keigo + anchor đa nghĩa.
- (e) **ĐÃ CHỐT (user duyệt 30/7) — hồi sinh chiều vi→ja, trộn 15% token-budget**: đảo chính
  cặp câu đã có + tag >>jpn<< — **không tốn KD mới, không tốn tiền**, chỉ thêm ~15 phút binarize.
  Bối cảnh: vi2ja đã chết vì KD v5-v7 một chiều (tag swap + v4_avg5 đều đã đo — xem ⚠ dưới).
  Cấu hình: chọn ~2,4M cặp LaBSE cao nhất trong 15,92M, đảo thành **vi(thầy, sạch) → ja(gốc thật
  CC-100/kokkai)** — đúng công thức back-translation: target là tiếng Nhật người thật viết nên
  output tự nhiên. Model thấy ~150-240M token chiều ngược trong 8-12k step. Kỳ vọng TRUNG THỰC:
  vi→ja "dùng được" câu ngắn/trung bình, vòng đầu THUA Google chiều đó (chiều xuôi đã ăn ~1B+
  token, chiều ngược mới ~0,2B; sinh tiếng Nhật khó hơn: chọn kanji + mức kính ngữ) — muốn ngang
  phải dồn liều qua các vòng sau. Nghiệm thu: bench RIÊNG chiều ngược 100 câu vi→ja chấm mù cùng
  phiên (mốc: "dùng được" ≥70%) + regression ja→vi trên 200b KHÔNG tụt (McNemar vs v7a) — chiều
  xuôi vẫn ăn 85% gradient + tiền lệ v4 gốc hai chiều cùng cỡ chạy tốt. Sau khi V8 đạt: mở khoá
  vi2ja trong sidecar app + gói deploy; autodetect chiều nằm ở tầng client (kana/kanji → >>vie<<,
  dấu tiếng Việt → >>jpn<< — hai ngôn ngữ không chung bảng chữ, detect theo dải ký tự ~100%).
  ⚠ Đường zero-train ĐÃ THỬ VÀ BÁC (30/7): (i) tag >>jpn<< trên v7a → rác tiếng Việt (đo);
  (ii) v4_avg5_i2s (tưởng là bản 2 chiều gốc) → cũng ra rỗng/tiếng Việt — nó là bản SAU KD pilot,
  checkpoint 2 chiều thật (v4 step 15500 pre-KD) đã bị xoá. Không prompt trick nào tạo được năng
  lực không có trong trọng số. Zero-train duy nhất còn giá trị: **reverse-reranking** — engine
  ngoài sinh N ứng viên ja, v7a CHẤM P(vi_nguồn|ja_ứng viên) chọn bản tốt nhất (model làm giám
  khảo, không làm máy dịch; trần chất lượng = trần của engine sinh).
- (f) **MỚI — Câu cụt kiểu STT (fragment robustness), ~30-50k mẫu — TUYẾN PHÒNG THỦ DUY NHẤT**
  (quyết định 30/7: không vá ở tầng app, model phải TỰ dịch lửng đúng): v7a chỉ luyện trên câu HOÀN
  CHỈNH, còn caption STT stream đầy mảnh cắt giữa chừng → model bịa vị ngữ + đảo vai (transcript
  07-30, ca nặng nhất: 「そのブリッチとしてそのベトノムの開発チームと日本の」→ "Đội ngũ… ĐÃ PHÁT
  TRIỂN Bitch đó" — bịa hoàn toàn). Cách làm: lấy câu thật từ corpus, CẮT tại ranh giới trợ từ/mệnh
  đề (sau の/と/が/で/、) mô phỏng chỗ STT hay đứt, thầy dịch thành **bản dịch lửng tương ứng, cấm
  hoàn thành ý** (prompt thầy ghi rõ); mỗi câu gốc giữ cả bản đầy đủ lẫn 1-2 bản cụt để model học
  phân biệt câu trọn vẹn vs mảnh. Phủ đủ các kiểu đứt hay gặp trong caption thật: đứt sau trợ từ
  sở hữu/liệt kê (の/と), đứt giữa danh ngữ, đứt trước vị ngữ. Nghiệm thu: fragment-probe T1 —
  tiêu chí là KHÔNG bịa (đếm vị ngữ thêm vào so nguồn), so baseline v7a cùng phiên. Stop-rule như
  (b): không giảm ≥50% số ca bịa → dừng loại liều này, đưa vào hồ sơ gate grow.
- (g) **TUỲ CHỌN thí nghiệm nhỏ — chịu nhiễu ASR, ≤15k mẫu**: transcript cho thấy STT nghe nhầm
  từ đồng âm/gần âm (金質↔品質, ステックフォーラ↔ステークホルダー) và model dịch trung thành rác.
  Sinh cặp (câu nhiễu kiểu ASR → bản dịch của câu SẠCH) bằng cách tự làm nhiễu kana/katakana câu
  corpus. Rủi ro: dạy model "đoán ý" có thể tăng bịa ở chỗ khác — chạy liều nhỏ, gate riêng bằng
  probe (f) + regression 200b, không ăn thì bỏ. Lưu ý phạm vi: nút thắt chính vẫn là STT của app
  (việc của app, không phải của model dịch).

**T3 — Audit thầy TRƯỚC khi bung**: 200 câu flash-live (CHƯA từng audit — nợ PLAN_V7A §2).
⚠ Nếu fail: phương án B "chuyển REST thầy mạnh" CHƯA có cơ sở quota (REST free 20 req/ngày) —
phải thử đo trước khi cam kết, đừng coi là gạt sẵn.

**T4 — KD ~300-450k câu** Live API 6 key (623k ≈ 2,5-3h theo log ⚠ không phải 2h) → rule filter
+ LaBSE ≥0,55 → **dedup tuyệt đối** với train + toàn bộ bench/dev (nợ 558 câu rò là bài học).

**T5 — Calibrate 3060 Ti 300 step = GATE CỨNG** trước khi cam kết: BITNET_OPT mặc định bật
(tắt là eager OOM vì STE fp32), `--max-tokens 2048 --grad-accum 64` (giữ 131k tok/step),
`--grad-ckpt` dự phòng. ⚠ **KHÔNG có gạt 8-bit Adam** — bnb Adam8bit crash "device not ready"
trên chính WSL2 này (train.py:311). Số nguồn đúng: L40S đo 1,27-1,30 s/step @ ~43k tok/s
(⚠ không phải 90-100k). Luật cắt: >12-15 s/step → thu 8k step, hoặc hỏi user nạp Modal $12.

**T6 — Train +8-12k step 18L từ v7a_avg**: LR-restart 5e-5→5e-6 (2 lần đã mở lại slope
−0,011~−0,013/1000: gate + p2; ⚠ dev cuối v7a −0,002/1000 là ĐÁY COSINE ở lr 5e-6, không phải
trần — đòn ăn tiền là DATA MỚI + restart, không phải step thuần), warmup 200, dev-every 500,
milestone 250, KEEP=12, backup về D:.

**T7 — Nghiệm thu MỘT phiên chấm mù**: (i) bench T1 so baseline đã đo (kỳ vọng neo tiền lệ
vòng 5 in-domain 14%→95% — ⚠ KHÔNG neo "katakana đã thắng": số thật 70%→70% phẳng);
(ii) regression 200b đủ 4 hệ — tiêu chí là **McNemar v8-vs-v7a cùng phiên**, không neo 78,5%
liên phiên; (iii) held-out kokkai 130-218 ký tự (⚠ bench 200b max 76 ký tự — không đo được câu
dài thật, ISSUES #1); cấm tụt long acc. Convert i2s → bench qua llama-server + token ids →
McNemar vs PyTorch p>0,1 mới phát hành `v8_avg_i2s.gguf` + cập nhật gói deploy/app.

## Luật grow 24L — SỬA LUẬT so PLAN_V7A §1, cần user duyệt tường minh

Grow 24L + ctx384 khi **một trong hai**:
1. long-OOD < 78% **VÀ** dev slope ≥ −0,003/1000 **đo ở LR ≥ 2e-5 trên data MỚI**
   (chặn nhầm đáy-cosine với trần thật — bằng chứng p3);
2. **Hard cap chống trì hoãn vô hạn**: 2 vòng 18L liên tiếp acc==2 tăng <2 điểm.

Khi grow: `grow_depth.py 18→24 --verify` (đo thật lần 12→18: sai lệch logits 0.000e+00 —
hoãn grow có regret ≈ 0), rải đều 6 block, max_seq 384 (bảng RoPE, 0 tham số mới), trộn tập
256-384 đã để dành, train 20-25k step. 24L ≈ 100MB i2s, ~275 tok/s (ước theo băng thông).

## Chi phí & lịch

| Track | Tiền | Thời gian | Ghi chú |
|---|---|---|---|
| D (vệ sinh deploy) | $0 | 0,5-1 ngày | chạy trước / song song đầu Track T |
| T (data+train) | $0 + điện ~$1-2 | 7-10 ngày (GPU máy A chiếm 17-40h liên tục) | dự phòng Modal $12 CHỈ khi user duyệt |

## Không làm vòng này (kèm lý do đã kiểm)

- **Self-edit >>fix<< 2 lượt**: token 32000 không có trong data train lineage v7a — muốn có phải
  train thêm data fix (ứng viên vòng sau, train local được).
- **Back-translation check**: vòng NÀY chưa làm được (vi2ja chỉ sống lại sau khi V8 train xong
  với T2e) — thành ứng viên cho vòng sau: dịch ngược output để tự kiểm.
- **Beam/rerank, tách-ghép ngẫu nhiên, position-offset RoPE, cân độ dài, data-free ternary**: đã bác bằng số — không lặp.
- **Grow 24L ngay**: chưa thỏa luật; +33% FLOPs/step vĩnh viễn + 20-25k step trong khi 18L còn ăn.
