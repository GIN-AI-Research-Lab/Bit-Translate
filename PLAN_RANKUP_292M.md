# PLAN RANK-UP — Leo bảng xếp hạng với model 292M (lập 2026-07-17)

> **⚠️ CẬP NHẬT 2026-07-20:** Phase 1 (§2, vòng 3a) đã chạy xong nhưng gate G1 TRƯỢT
> (ja→vi 2.37 vs Google 3.67, thua 5/5 domain — chi tiết STATUS TL;DR). User chốt
> ja→vi là sản phẩm chính. **Chiến lược ja→vi chuyển sang sequence-level KD — xem
> `PLAN_KD_JA2VI.md`** (thay thế §2-§3 của file này; thầy ĐỔI từ Haiku trả phí sang
> gemini-flash-lite+qwen-plus FREE sau benchmark 2026-07-20 — điểm cao hơn cả Haiku).
> Đợt 0 (train tiếp step30000→32500) đã xong, kết luận OVERFIT — dùng step30000 làm
> nền, không dùng wave0. §8.4 (app-side) + §4-§5 (Haiku glossary-bench, Fable vận
> hành) GIỮ NGUYÊN giá trị.

> Bảng xếp hạng hiện tại (hardbench200 4 hệ, `eval/hardbench_4way_haiku_report.md`):
> **Fable 5.00 > Haiku 4.57 > Google 3.80 >> 110M 1.56** (acc TB 2 chiều).
> Kế hoạch này trả lời: nâng lên 292M thì "đánh bại từng bậc" bằng cách nào, đo bằng gì,
> và bậc nào KHÔNG thể đánh bại theo nghĩa đen — nói rõ từ đầu, không hứa quá (CLAUDE.md §3).

## 0. Nguyên tắc trung thực — "đánh bại" nghĩa là gì ở từng bậc

| Bậc | Có thắng được không? | Nghĩa của "thắng" | Vì sao |
|---|---|---|---|
| **Google** | **CÓ — thắng thật, theo lộ trình domain** | Judge acc cao hơn trên hardbench từng domain, rồi cả chiều ja→vi | Google yếu hệ thống do bắc cầu EN (thành ngữ 2.5-3.0, slang 3.3, hội thoại 3.2-3.5, zeropronoun vi→ja 3.5) — lỗi TRI THỨC ĐÓNG GÓI ĐƯỢC vào data hẹp |
| **Haiku** | **CHỈ trên trận địa tự chọn** | Thắng về **term-accuracy + tính nhất quán** trên bộ đề nội bộ có glossary công ty; KHÔNG thắng đề mở | Haiku là frontier LLM ~1000× tri thức thế giới; nhưng nó KHÔNG biết thuật ngữ/quy ước nội bộ của công ty mình — mình nhét được vào data + glossary-injection |
| **Fable** | **KHÔNG (chất lượng), CÓ (vận hành)** | Thắng tuyệt đối: tốc độ/câu, chi phí 0đ, offline, riêng tư — đo thành số để "thắng có bằng chứng" | Đã chốt PLAN §4.3: không pipeline 300M nào bù nổi tri thức thế giới |

Trục leo hạng THẬT của dự án = trục Google. Haiku/Fable là mốc tham chiếu + công cụ (thầy sinh data).

## 1. Phase 0 — Gate 292M (điều kiện tiên quyết, ~2 ngày)

Đang dở: resume train 292M từ autosave step 1375/25000 (~31.5h GPU, thủ tục ở STATUS TL;DR).

- **Gate vào (đã chốt §3.6):** 292M thắng 110M step23000: FLORES vi→ja >21.49 / ja→vi >42.27; probe64 TB >31.7/48.7; glossary >28%.
- **Gate mới thêm (từ bài học 16/07):** chạy lại **hardbench200 + trọng tài 4-way y hệt** (`eval/build_judge_panels_4way.py`, thay 110M bằng 292M) — chrF một mình đánh giá thấp khoảng cách thật.
- Thua gate ⇒ nút thắt là DATA không phải size ⇒ vẫn chạy Phase 1 nhưng trên 110M, hoãn mọi bàn scale tiếp.
- Kỳ vọng thực tế: 292M from-scratch trên CÙNG data vòng 2 sẽ ăn chủ yếu ở chỗ 4 tín hiệu trần đã bật (T1-T4): câu dài, giữ đồng thời nhiều domain, glossary. Judge dự kiến 1.56 → ~2.0-2.4. **Chưa chạm Google — đó là việc của data Phase 1-2, không phải của params.**

## 2. Phase 1 — VƯỢT GOOGLE bậc 1: chiều ja→vi (vòng 3 data, ~1 tuần)

Chiến lược: đánh chỗ Google ĐANG YẾU + mình ĐANG MẠNH SẴN (ja→vi công sở), chưa đánh vi→ja.

**Data (thứ tự tác động, chạy trên 292M sau gate):**
1. **Thành ngữ + slang + câu công thức văn hoá** (Google 2.5-3.6): từ điển thành ngữ/khẩu ngữ 2 chiều ~2k mục × 5-8 câu/mục — sinh bằng Haiku ($1/$5, hôm nay đo acc 4.45-4.69 trên chính đề khó của ta ⇒ đủ chuẩn làm THẦY, rẻ hơn Fable ~10×), lọc LaBSE ≥0.8 + eyeball 50 mẫu.
2. **Hội thoại xưng hô + zero-pronoun có ngữ cảnh** (Google 3.2-3.5): format `câu ngữ cảnh ||| câu cần dịch` sinh bằng CODE từ cặp liền kề OpenSubtitles/TED (§9#1, ~0 token) + persona quan hệ rõ (§9#2).
3. **Số liệu 万/億/loại từ/niên hiệu + phủ định N2-N1 + câu ghép** — sinh bằng code theo §4.1 (đã viết sẵn công thức).
4. **Glossary-injection (§4.3.1)**: train mẫu có gợi ý `[term=訳語]` — bật lợi thế mà Google/Haiku không có: TÙY BIẾN theo khách.
5. **Thầy–trò error-targeted (§4.3.2), pilot bằng Haiku**: 292M dịch 30-50k câu monolingual → Haiku chấm nhanh + viết bản sửa → train trên bản sửa. Ước phí: 50k câu × ~150 tok out ≈ 7.5M tok ≈ **~$40**.

**Gate G1 (đo hardbench 4-way + probe64):**
- Thắng Google judge acc ở ≥3 domain ja→vi trong {hop, it_deep, hoithoai, thanhngu, slang} (hiện Google 3.0-3.8 ở nhóm này).
- probe64 ja→vi chrF ≥ Google (52.2) — hiện 110M đã 48.7, còn 3.5 điểm.
- Không domain cũ nào tụt >2 chrF (replay ≥65%).

## 3. Phase 2 — VƯỢT GOOGLE bậc 2: kéo chiều vi→ja + tổng ja→vi (vòng 4-5, ~2 tuần)

vi→ja là điểm chết cấu trúc (1.22 vs Google 3.88) — đòn duy nhất đủ lực là **back-translation quy mô lớn** (kế hoạch gốc 500k-1M câu JA, mới dùng 86k):

1. **BT 500k-1M câu JA đơn ngữ** (harvest_ja_indomain.py sẵn) dịch ngược bằng model 292M chiều ja→vi (chiều mạnh) → cặp (VI_synthetic, JA_gold) train chiều vi→ja. Lọc round-trip + LaBSE.
2. **Self-edit `>>fix<<` (§4.3.3)** train cùng lúc — vá garble câu dài vi→ja (hiện 0.67!), latency 2× chấp nhận được ở 140-460 tok/s.
3. Vòng 5: lặp thầy–trò lần 2 nhắm đúng residual từ eyeball vòng 4.

**Gate G2:**
- hardbench **ja→vi tổng: judge acc ≥ Google (3.73)** — tuyên bố "thắng Google chiều ja→vi trên câu khó".
- vi→ja: judge ≥2.8 + caudai vi→ja hết garble (≥2.5). *(Thắng Google 3.88 chiều vi→ja trong đời 292M là KHÔNG cam kết — asymmetry sinh tiếng Nhật cần tri thức chọn kanji/keigo; nếu G2 đạt mà vi→ja kịch trần → đó là lúc bàn scale tiếp, không phải bây giờ.)*
- FLORES formal news: **chấp nhận không thắng Google** — ngoài phạm vi sản phẩm (công sở IT), chỉ cần không tụt.

## 4. Phase 3 — "VƯỢT HAIKU" trên trận địa tự chọn (~1 tuần, sau G1)

Định nghĩa trận địa trước, đo cả 2 bên trên CÙNG đề:
1. Dựng **glossary-bench 100 câu** từ tài liệu/term nội bộ user cấp (§6 việc user): câu chứa ≥1 term bắt buộc + số liệu + mã ticket.
2. Metric: **term-accuracy** (đúng term chuẩn công ty, kiểm bằng code) + judge acc/nat 4-way.
3. So 4 cấu hình: 292M+glossary-injection vs Haiku vanilla vs Haiku được nhét glossary vào prompt vs Google.
- **Thắng tuyên bố được:** term-accuracy > Haiku vanilla (khả thi cao — Haiku không biết term nội bộ) và ≥ Haiku+glossary ở tính nhất quán/latency.
- **Không tuyên bố:** thắng Haiku về judge acc trên đề mở — số hôm nay (4.57 vs trần khả dĩ ~3.x của model 292M) nói rõ điều đó.

## 5. Fable — thắng trục vận hành, đo thành số (0.5 ngày, làm cùng Phase 1)

Bảng "tổng chi phí sở hữu" đưa vào README sản phẩm: p50 latency/câu 30 token (~0.2-0.25s local vs round-trip API), chi phí 1M câu/tháng (0đ vs ~$300-400 Haiku vs ~$3-4k Fable), offline/không rò dữ liệu ra ngoài (yêu cầu nhiều khách doanh nghiệp Nhật). Đây là lý do tồn tại của model 69MB — nói bằng số, không nói bằng cảm tính.

## 6. Nhịp đo & kỷ luật (giữ nguyên harness)

- Mỗi vòng: probe64 + FLORES n=100 + **hardbench200 4-way judge** (script sẵn, ~1h) + user eyeball 50 câu.
- Điền `eval/capacity_log.md` sau mỗi vòng — nếu 292M bật lại ≥2 tín hiệu trần TRƯỚC khi đạt G2 thì dừng đổ data, họp lại về size (500M là ranh giới phải cân nhắc vì mục tiêu CLAUDE.md là 25-50MB runtime; 292M i2_s đã ~75-90MB).
- Ngân sách sinh data ước tính cả 3 phase: **~$150-250 nếu dùng Haiku làm thầy chính** (so ~$1.5-2.5k nếu dùng Fable) — cần user chốt (§6 việc user, mục 3).

## 7. Timeline tổng

| Phase | Việc | Thời gian | Kết quả tuyên bố được |
|---|---|---|---|
| 0 | Train nốt 292M + gate 4-way | ~2 ngày | 292M > 110M (hoặc kết luận data là nút thắt) |
| 1 | Vòng 3 data ja→vi + glossary-inject + thầy-trò pilot | ~1 tuần | Thắng Google ≥3 domain ja→vi khó + probe64 ja→vi |
| 2 | BT 500k-1M + self-edit + vòng 5 | ~2 tuần | **Thắng Google toàn chiều ja→vi trên câu khó**; vi→ja ≥2.8 |
| 3 | Glossary-bench vs Haiku | ~1 tuần (song song) | Thắng Haiku về term-accuracy nội bộ |
| — | Bảng vận hành vs Fable/Haiku | 0.5 ngày | Thắng tuyệt đối trục tốc độ/chi phí/riêng tư |

Tổng ~4-5 tuần lịch (phần lớn là train nền + chờ eyeball). Rủi ro lớn nhất: BT chất lượng thấp kéo tụt (đối sách: lọc round-trip + LaBSE + trộn ≤35% synthetic mới/vòng); node cloud chết giữa chừng (autosave 30' đã có).

## 8. Chuyên đề: câu DÀI vs câu KHÓ — hai bệnh, hai thuốc (bổ sung 2026-07-17)

**Chẩn đoán từ số 4-way (acc theo tercile độ dài nguồn):**

| chiều | NGẮN | VỪA | DÀI | kết luận |
|---|---|---|---|---|
| vi→ja 110M | 1.44 | 1.23 | **0.99** | tụt theo độ dài — bệnh PHÂN BỐ DATA + sinh JA |
| vi→ja Google/Haiku | 3.60/4.57 | 3.67/4.67 | **4.35/4.84** | hệ lớn càng dài càng tốt → không phải "câu dài vốn khó" |
| ja→vi 110M | 2.00 | 1.85 | 1.85 | PHẲNG — cái sập là TRI THỨC (thành ngữ 1.77, slang 1.80, keigo 1.90), không phải độ dài |

⇒ "Train câu dài" chỉ đáng tiền cho **chiều vi→ja**. "Train idiom" là thuốc của **ja→vi + slang/thanhngu vi→ja**. Trộn hai bệnh vào một đơn thuốc là lãng phí ngân sách data.

### 8.1 Thuốc TRAIN cho câu dài (vi→ja)

1. **Đo phân bố trước khi đổ data**: histogram độ dài data train hiện tại vs hardbench (p50/p95). Giả thuyết cần xác nhận: OpenSubtitles ngắn áp đảo → câu 25-60 token nằm ngoài vùng model từng thấy.
2. **Nối câu bằng code (miễn phí, làm đầu tiên)**: ghép 2-3 cặp ngắn liền kề thành câu ghép có liên từ (rồi/sau đó/tuy…nhưng ↔ てから/が/ので) — §4.1 đã có công thức; mục tiêu ~100-150k cặp 25-60 token, oversample bucket dài khi mix.
3. **BT chọn lọc theo độ dài**: trong 500k-1M câu JA của Phase 2, ép ≥40% là câu 30-60 token (harvest_ja_indomain lọc theo len) — BT vốn là đòn cứu vi→ja, thêm điều kiện dài là trúng hai đích một mũi tên.
4. **Curriculum nhẹ**: 2k step cuối mỗi vòng oversample bucket dài ×2 (tránh làm sớm gây quên câu ngắn — niche đang thắng Google là câu ngắn, không được hy sinh).
5. Gate riêng: caudai vi→ja judge 0.67 → **≥2.5 (hết garble)** sau Phase 2; tercile DÀI vi→ja không còn thấp hơn tercile NGẮN.

### 8.2 Thuốc TRAIN cho idiom/tri thức (ja→vi + slang vi→ja)

- Đây là bài học THUỘC LÒNG (memorization), không phải kỹ năng: từ điển ~2k mục thành ngữ/slang/câu công thức × 5-8 ngữ cảnh/mục × 2 chiều (Haiku sinh, ~$30-50) + **cặp tương phản** nghĩa đen vs nghĩa bóng ("gấu"=恋人 trong ngữ cảnh yêu đương / =クマ trong ngữ cảnh sở thú) để model học ĐIỀU KIỆN chọn nghĩa, không học vẹt 1-1.
- Trần sức chứa là rủi ro thật (T1-T4 từng bật ở 110M): nếu 292M nhét 2k mục mà glossary-rate không tăng → đừng đổ thêm, chuyển tri thức sang app-side (8.4).

### 8.3 Ngữ cảnh — có ổn không? CÓ, nhưng đúng liều

- Ngữ cảnh (1-2 câu trước, format `ctx ||| src`) chữa được: **zero-pronoun, xưng hô, nhất quán register** — đúng nhóm đang 1.9-2.3 và là chỗ Google cũng yếu (3.2-3.5). Data sinh bằng CODE từ cặp liền kề OpenSubtitles/TED (~0 token). Đáng làm ở vòng 3 (đã chốt §3.6, không đổi arch).
- Ngữ cảnh KHÔNG chữa được: garble câu dài (bệnh phân bố), idiom (bệnh tri thức). Kỳ vọng đúng: +0.3-0.6 judge ở hoithoai/zeropronoun, ~0 ở caudai/thanhngu.
- Chi phí thật nằm ở APP: phải giữ buffer câu trước + latency prompt dài hơn chút — rẻ, đáng.

### 8.4 Xử lý qua APP (không cần train hoặc train 1 lần rồi tri thức sống ở app)

Xếp theo tác động/chi phí:

0. **Chuẩn hóa NFKC input — BUG THẬT ĐÃ XÁC NHẬN 17/07, fix bắt buộc, 0 đồng.**
   SPM lúc train áp NFKC (？→?, ！→!, ％→%) nhưng tokenizer llama.cpp lúc chạy KHÔNG chuẩn hóa
   → dấu fullwidth tiếng Nhật thật bị tách thành byte-fallback (？ → `[245,194,165]` thay vì `?`=274),
   model gần như chưa từng thấy. Hardbench không dính (đề soạn bằng halfwidth) nhưng ĐỜI THẬT dính 100%
   câu hỏi/cảm thán/% tiếng Nhật. Đã fix: `scripts/text_norm.py` (`normalize_for_model`) + vá
   `run_hardbench.py`/`run_probe64.py`; APP phải gọi cùng hàm này trước mọi lần tokenize,
   và `restore_ja_fullwidth` cho output vi→ja khi hiển thị.
1. **Tách câu trước khi dịch (segmentation)** — pilot 17/07 trên 32 câu ja→vi nhiều-câu cho kết quả CÓ ĐIỀU KIỆN:
   chrF TB không đổi (15 tăng/10 giảm) nhưng (a) **xóa lỗi bịa định dạng phụ đề** "- ... - ..."
   (model tưởng input 2 câu = 2 lượt thoại — di chứng OpenSubtitles), (b) **câu formal/IT nhiều-câu lợi rõ**
   (id63: 37→57 chrF, term IT dịch đúng), (c) khẩu ngữ KHÔNG cứu được (bệnh tri thức).
   ⇒ Luật app: tách tại 。！？!? (an toàn tuyệt đối, làm luôn — `split_sentences` trong text_norm.py);
   tách mệnh đề tại phẩy/liên từ CHỈ cho câu >25 token và đo thêm trước khi bật mặc định.
   Kèm `clean_output` lột "- " đầu output. Data-side đối xứng: (i) lọc/lột dash "- " của OpenSubtitles
   trong lần binarize tới, (ii) thêm mẫu train 2-3 câu ghép GIỮ nguyên dấu 。？ để model quen input nhiều câu.
2. **Glossary/idiom-injection lúc chạy** (§4.3.1): từ điển nằm trong APP (file tsv user sửa được, không cần retrain), app quét câu nguồn nhét gợi ý `[term=訳語]`. Train 1 lần cho model biết dùng gợi ý; sau đó thêm term mới = sửa file. Đây là câu trả lời app-side cho idiom + term khách hàng.
3. **Placeholder số/mã/tên riêng (pre/post-process thuần code)**: thay số tiền, ngày giờ, version, mã ticket bằng token giữ chỗ trước khi dịch, dịch xong thay lại + bảng quy đổi 万/億 bằng regex. Trị dứt điểm lỗi sai hàng số mà KHÔNG tốn params — deterministic 100%.
4. **Self-edit `>>fix<<` 2 lượt** (§4.3.3): chạy ở app, latency 2× (~0.3-0.5s/câu vẫn ổn), vá fluency/sót vế.
5. **Cầu dao chất lượng (confidence gate)**: app phát hiện output rủi ro bằng heuristic rẻ — tỉ lệ độ dài out/in bất thường, ký tự Việt sót trong output Nhật, lặp n-gram — rồi (a) gắn nhãn "bản dịch tham khảo" hoặc (b) **hybrid: nếu có mạng, đẩy đúng CÂU KHÓ đó sang API Haiku ($1/$5)**. Sản phẩm cuối = 292M offline xử 80-90% câu thường + thoát hiểm cloud cho đuôi khó; người dùng không bao giờ thấy garble.
6. Streaming: re-translation + local agreement (§10 PLAN_BUOC5, đã có) — không hiển thị bản chốt khi câu Nhật chưa hết vì phủ định nằm cuối.

**Thứ tự làm:** pilot (1) ngay hôm nay/mai trên 110M (1-2h, script sẵn) → (3) viết luôn vào app (nửa ngày, thuần regex) → (2)+ngữ cảnh vào data vòng 3 → (4)(5) sau gate 292M.

## 9. CHECKLIST SAU KHI 292M ĐẠT 25000 STEP (ghi 2026-07-18 — data vòng 3a ĐÃ SẴN)

Bối cảnh: toàn bộ data vòng 3a đã sinh + review + mix xong trong 18/07 (Opus review
`eval/DATA_REVIEW_VONG3_OPUS.md`, tổng kết `data/synthetic/vong3/SUMMARY_VONG3A.md`).
Premix **`train-assets-vong3a`** (23,45M seq, new 18,7%, cùng công thức oversample-short
với vòng 2) = base + vòng1 + vòng2 + [6.615 idiom/slang + 50k `ctx|||src` + 5.457 số liệu
+ 4.232 glossary-inject]. Node cloud round kế CHỈ cần đổi tag tải premix.

1. **Gate Phase 0 (~nửa ngày, harness sẵn):** convert GGUF i2_s → FLORES n=100
   (vi→ja >21.49 / ja→vi >42.27) + probe64 (>31.7/48.7) + glossary-test (>28%)
   + **hardbench200 4-way judge** (thay 110M bằng 292M; kỳ vọng 1.56→~2.0-2.4)
   → điền `eval/capacity_log.md` + USER eyeball 50 câu.
2. **Rẽ nhánh:** ĐẠT → bước 3. TRƯỢT → nút thắt là DATA (§1): vẫn train vòng 3a
   (cân nhắc trên 110M cho rẻ), hoãn mọi bàn scale.
3. **Train vòng 3a trên cloud:** đổi tag premix `train-assets-vong2` → `train-assets-vong3a`
   (dọn data/bin/train.* + marker premix cũ trước khi giải nén) → resume từ ckpt 25000,
   **+4.000-6.000 step** (PLAN_BUOC5 §4.2) với `--lr-anchor 25000` (LR restart từ đỉnh)
   → autosave + watcher như cũ.
4. **Đo G1 (§2) + đo trần:** thắng Google ≥3/5 domain ja→vi {hop, it_deep, hoithoai,
   thanhngu, slang}; probe64 ja→vi ≥52.2; không domain cũ tụt >2 chrF. Kèm **glossary
   term-rate**: >35-40% → được harvest từ điển 1.247→2k mục; đứng ~28% → DỪNG nhồi
   idiom, chuyển app-side (§8.4#2).
5. **Test glossary-injection runtime:** dịch câu có hint `[term=訳語]` (đã train 4.232 mẫu)
   — model dùng gợi ý ⇒ mở khóa app-side glossary + Phase 3 (glossary-bench vs Haiku,
   cần USER cấp term nội bộ).
6. **Song song:** `harvest_ja_indomain.py` gom 500k-1M câu JA mono (CPU local, làm được
   NGAY) chuẩn bị BT vòng 3b/G2; nối câu code-gen (§8.1) + self-edit `>>fix<<` để dành
   vòng 3b; bảng vận hành Fable (0.5 ngày).
