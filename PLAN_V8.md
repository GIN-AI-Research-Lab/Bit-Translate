# PLAN V8 — Hai track: Sản phẩm ($0, 2-3 ngày) + Data-Train 18L ($0-2, 7-10 ngày)

> Tổng hợp 2026-07-30 từ 3 kế hoạch độc lập (data-first / scale-first / product-first),
> mỗi cái đã qua 1 agent phản biện kiểm từng khẳng định với file repo. Các con số dưới đây
> đều đã kiểm nguồn; chỗ nào phản biện bác thì đã sửa (ghi chú ⚠).
> Bối cảnh: gate V7A kết luận **18L CHƯA TRẦN** (long-OOD 80% ≥ 78%, dev −0,011/1000 cuối gate)
> → KHÔNG grow 24L vòng này. Modal $0 → train local 3060 Ti (máy A). Gói deploy đã đóng
> (`D:/Bit-Translate-data/dist/v7a_deploy_pkg/`), app teams-caption-translator đã tích hợp.

## Track P — Sản phẩm, không train ($0, 2-3 ngày) — LÀM TRƯỚC

| # | Việc | Chi tiết + điều kiện nghiệm thu |
|---|------|--------------------------------|
| P1 | **Chuẩn hoá client inference** | `v7a_deploy_pkg/src/translate.py` là đường chuẩn duy nhất. Sửa hoặc deprecate `demo/server.py` — nó đang dính 4 lỗi thật (phản biện kiểm từng dòng): gọi `/tokenize` của llama.cpp (đúng bug ISSUES #4, −16 điểm use đo thật), `repeat_penalty 1.25` (bench chuẩn 1.0), `n_predict 96` cứng (cắt câu dài), gửi token `>>fix<<`=32000 mà lineage v7a **chưa từng train** (HANDOFF:387). App Teams đã đi đường đúng (sidecar). |
| P2 | **Bảng kính ngữ thư tín tầng tool** | 150-300 **TEMPLATE NGUYÊN CÂU** (⚠ phản biện: match cụm-cuối-segment sẽ tự vô hiệu vì ご清栄/ご査収 nằm GIỮA câu định hình). Gemini nháp → user duyệt mắt 1 lượt (bảng đóng, duyệt 1 lần). Nhắm id182/187/189 (⚠ id184 v7a ĐÃ đúng — bỏ khỏi danh sách). |
| P3 | **Cờ confidence + fallback** | Tín hiệu: logprob (build BitNet-test CÓ n_probs — đã kiểm source server.cpp:882), tỉ lệ độ dài VI/JA ngoài [0,4;2,5], thiếu dấu câu cuối (nghi cắt cụt), loop n-gram (retry 1 lần rp=1.25 chỉ-khi-loop), rỗng. ⚠ Hiệu chỉnh trên lớp **acc≤1 (43 câu)** của judge 200b, không phải 13 câu acc==0 (quá mỏng). Precision-first; GO/NO-GO logprob ngay ngày 1. |
| P4 | **Tốc độ + UX server** | `-t 8` (353 vs 301 tok/s @4 — TONGKET_V6 §4; sidecar app đang -t 4 → đổi), stream token, hàng đợi TUẦN TỰ (2 job song song chậm 5,6×), ghi chú cắm sạc. |

**Nghiệm thu Track P**: 200 câu **`eval/bench_opus200b.jsonl`** (⚠ GHI CỨNG — không phải
bench_new.jsonl, translate_bench mặc định trỏ nhầm bộ) qua tool-path vs raw-path, chấm mù
CÙNG PHIÊN, McNemar **p>0,1** (bỏ ngưỡng % tuyệt đối — thang judge trôi ±8 giữa phiên).
Mini-bench thư tín: **n≥100** câu mine từ thư thật held-out (⚠ n=40 chênh 4 câu = nhiễu).

## Track T — Data + train tiếp 18L ($0-2, 7-10 ngày)

**T0 (0,5 ngày — ⚠ thiếu trong plan gốc)**: chuyển bin + checkpoint (~2,5-4GB) từ laptop
sang máy A, kiểm env WSL2/torch/compile ở đó.

**T1 — Dựng bench nghiệm thu TRƯỚC khi có data** (bài học "bench đo sai miền"):
keigo-letter ~100 câu mine từ thư thương mại thật held-out (ĐỘC LẬP pipeline sinh — né bẫy
vòng 3); polysemy-probe 150 câu nghĩa phụ từ CC-100 thật; in-domain **n≥100/miền**.
Đo baseline v7a + Google cùng phiên, lưu làm mốc.

**T2 — Data 4 nhánh (+1 tuỳ chọn)**:
- (a) **Kính ngữ thư tín ~400 cụm đóng** (頭語/結語, 時候の挨拶, ご清栄/ご査収/お納め…, sonkeigo/kenjougo
  N2-N1): Gemini enumerate 500 → **đếm CC-100 phán quyết** giữ ~400; mine exact-string ≤60 câu/cụm
  + sinh bù cụm hiếm ×3 oversample ≈ 30k câu. ⚠ Mục tiêu là PHỦ CỤM ≥60 lần/cụm, không phải kéo
  mật độ keigo tổng (đã 4% từ v5 mà vẫn fail cụm thư tín — "4% = ngang Google" chỉ là giả thuyết).
- (b) **Đa nghĩa chia THEO NGHĨA** (khác vòng 6: 512 từ×200 câu KHÔNG chia nghĩa → 当たる vẫn fail):
  ~200-250 động từ JMdict ≥3 nghĩa × 70 câu/NGHĨA-PHỤ ≈ 50-60k; kèm aspect-suffix lật nghĩa
  (〜かける suýt/dở, 〜そびれる, 〜きれない…). **Stop-rule**: probe không nhích ≥+8 điểm → dừng
  loại liều này vĩnh viễn (chồng lớp B, có rủi ro không ăn — 3 bằng chứng trần đã có).
- (c) **Câu dài**: mine kokkai/CC-100 120-250 ký tự + ghép LIỀN KỀ ≤256 token (PLAN_V7A §3a cho
  phép tường minh). Đồng thời xuất **tập 256-384 (~250-300k mẫu) ĐỂ DÀNH cho grow** — KD luôn
  trong đợt này (phòng quota đổi), KHÔNG trộn vào bin_v8.
- (d) **Miền đỏ: SINH thay mine** (⚠ phản biện bác tiền đề "đào CC-100 lần 2": kho đã quét TOÀN BỘ
  2 lượt ngày 07-28 cho danh sách 54 miền, 2,2M câu đã tiêu thụ — mine lại ra ~0; 4.073 từ còn thiếu
  "chỉ có thể SINH" theo DOMAIN_MAP §4). Giữ NHỎ: thuật ngữ chỉ ~3% lỗi taxonomy. Mine mới thật sự
  chỉ còn: 6-7 miền văn hoá mới + cụm keigo + anchor đa nghĩa.
- (e) **TUỲ CHỌN — user quyết**: trộn 10-20% chiều **vi→ja** (tag >>jpn<<, đảo chính cặp câu đã có).
  Phát hiện 2026-07-30 khi tích hợp app: **vi2ja đã chết** (input vi ra output vi) vì KD v5-v7 một
  chiều. App Teams hỗ trợ 2 chiều — nếu cần vi→ja trong tool thì đây là đường rẻ nhất.

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
| P (sản phẩm) | $0 | 2-3 ngày | chạy trước / song song đầu Track T |
| T (data+train) | $0 + điện ~$1-2 | 7-10 ngày (GPU máy A chiếm 17-40h liên tục) | dự phòng Modal $12 CHỈ khi user duyệt |

## Không làm vòng này (kèm lý do đã kiểm)

- **Self-edit >>fix<< 2 lượt**: token 32000 không có trong data train lineage v7a — muốn có phải
  train thêm data fix (ứng viên vòng sau, train local được).
- **Back-translation check**: vi2ja chết (trừ khi làm T2e).
- **Beam/rerank, tách-ghép ngẫu nhiên, position-offset RoPE, cân độ dài, data-free ternary**: đã bác bằng số — không lặp.
- **Grow 24L ngay**: chưa thỏa luật; +33% FLOPs/step vĩnh viễn + 20-25k step trong khi 18L còn ăn.
