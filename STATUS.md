# STATUS — BitNet 1.58-bit VI↔JA (cập nhật 2026-07-30)

## 🟢 VÒNG V7A XONG (2026-07-28 → 30) — v7a_avg là bản deploy chính, thay v6

> Thẻ model + bảng số đầy đủ: **`TONGKET_V7A.md`**. Kế hoạch gốc: `PLAN_V7A.md`.
> Số liệu bench chính: `eval/judge_200b_claude/KETQUA.md`. Verdict gate: `logs/KETQUA_V7A_GATE.txt`.

1. **Data KD vòng này: 623.007 câu** (Gemini Live API, 6 key × 20 session = 120 luồng,
   dịch đạt 98% = 618.854 câu rồi dừng "đủ" — `logs/kd_v7a.log`). Sau QC còn **614.120 câu
   mới (3,9%)**, gộp với corpus cũ → **15.921.937 cặp** (`kd_v7a_merged.jsonl`).
   **Dev ĐÃ LỌC RÒ**: dev sạch **3.886 câu** (nợ 558 câu rò 12,56% từ vòng 6 đã trả).
   Binarize `bin_v7g`: 15.885.724 seq / 946M token, bỏ 40.091 câu >256, mean len 59,6
   (`logs/overnight_v7a.log`).
2. **GATE trần scale (PLAN_V7A §1): 4.000 step trên Modal tritue12** (lr-restart 5e-5,
   dev-every 500 trên dev sạch). Dev: 2,1283@500 → **2,0617@4000**, cuối gate vẫn giảm
   **−0,011/1000 step**. Chấm mù long-OOD: **80% ≥ ngưỡng 78%** ⇒ **VERDICT: 18L CHƯA
   ĐẾN TRẦN** — chạy tiếp nhánh 18L, hoãn grow 24L (`logs/KETQUA_V7A_GATE.txt`,
   `eval/judge_gate_opus/`).
3. **Train tiếp 4000 → 8750** (`train_v7a_p2.log`/`p3.log`): p2 chạy đến ~7970 thì
   **hết credit tritue12 → failover sang tài khoản `nguyentuanngai`**, resume từ 7500.
   Dev tiếp tục giảm: 2,0938@4500 (bump lr-restart) → 2,0579@8000 → **2,0569@8500**.
   **Step 9000 KHÔNG kịp save** — container chết ngay sau dòng log cuối; milestone cuối
   còn lại là 8750. **`v7a_avg` = trung bình 4 mốc step 8000–8750**; dev (thang ls=0):
   v7a_avg **0,8435** vs gate_avg 0,8535 (`logs/v7a_final.log`).
4. **Bench — hai lượt chấm mù độc lập:**
   - **Opus chấm mù 200 câu tự sinh** (10 domain × 20, 100 ngắn/100 dài, 68 câu khó;
     4 hệ v7a_i2s-deploy/Google/gate/v6 cùng phiên — `eval/judge_200b_claude/KETQUA.md`):
     use% hòa 4 hệ ~92% (bench phổ thông đã trần). **v7a i2s ĐỨNG ĐẦU acc==2 78,5%**
     (Google 69,0 / v6 75,5 / gate 74,5) và **nat==2 76,5%** (Google 60,0).
     **Câu dài acc 73%** vs v6 67 / gate 65 (+6..+8 — đúng chỗ nhắm); câu khó 75,0%
     vs v6 67,6. Bản i2_s deploy giữ nguyên chất lượng (xác nhận fix tokenizer #4).
   - **Judge Gemini** (`eval/judge_v7a_opus/` n=100 OOD, `eval/judge_v7a_rand/` n=200):
     opus100 acc==2: v7a 87% = v6-pt 87%, gate 81%, Google 89%; rand200: v7a 79%,
     v6-pt 83%, Google 75,5%. Chênh v7a vs v6-pt ở rand KHÔNG có ý nghĩa (McNemar
     8 hơn/16 kém, p≈0,15) — không mâu thuẫn với bench Opus 4-hệ-cùng-phiên ở trên.
5. **Deploy**: `D:/Bit-Translate-data/dist/v7a_avg_i2s.gguf` (**77,56MB**, ~350 tok/s
   @6-8 luồng CPU laptop, llama-server + token ids theo ISSUES #4). Gói deploy cho tool
   dịch: `D:/Bit-Translate-data/dist/v7a_deploy_pkg/` (đang đóng song song).
6. **Ngân sách Modal: HẾT.** tritue12 = $0; `nguyentuanngai` đã lố hạn mức $3 free sang
   tiền thật (user đã stop, KHÔNG train trả phí nữa). Đường train miễn phí còn lại duy
   nhất: **máy A desktop (3060 Ti 8GB + 48GB RAM, WSL2)** theo cấu hình CLAUDE.md §7,
   ước chậm hơn L40S 2-4×.
7. **Lỗi còn lại** (từ bench 200b): cụm kính ngữ thư tín cố định (ご査収/お納めください/
   ご清栄/取り急ぎ — id182/184/187/189) + nghĩa phụ động từ thường theo ngữ cảnh
   (当たる=trút giận id67, 漏れかける=suýt lộ id169 — cả 4 hệ cùng fail); domain
   `art_ent` 80% use (Google 95%), `formal_letter` 80%. → **ISSUES.md #5 (MỚI)**.
   Câu dài: tiến triển +6 điểm nhưng vẫn dưới câu ngắn (73 vs 84) → ISSUES #1 còn MỞ.

## ⏭️ VIỆC TIẾP THEO (đọc trước khi làm gì mới — máy khác đọc mục này là đủ bắt nhịp)

> ⚠️ Mục dưới đây là trạng thái 2026-07-21 (trước vòng 5/6/V7A) — giữ làm hồ sơ.
> Trạng thái mới nhất: đọc mục 🟢 V7A ở trên + `TONGKET_V7A.md` + `PLAN_V7A.md`.

1. **KẾT LUẬN CHUỖI THÍ NGHIỆM 100M ja→vi (v1→v3→v4→clean-finetune, 2026-07-21):
   MỌI đòn bẩy data-nhỏ-giọt và decoding ĐÃ THỬ VÀ ĐỀU KHÔNG PHÁ ĐƯỢC TRẦN.**
   Chuỗi bằng chứng (chi tiết từng bước ở các mục dưới + commit `dad890c`):
   - v1 (step10500, 778 KD): judge acc ja→vi **1,93**.
   - v3 (step13000, +19.584 KD + 2.759 fix): **2,08** — thua 292M (2,42 cùng
     panel), thua Google (4,05). FLORES câu ĐƠN GIẢN: v3 2,45 vs Google 4,83,
     đối đầu 1/6/93 — khoảng cách thật ở mọi độ khó (`eval/judge_4way_v3`,
     `eval/judge_flores_v3`).
   - v4 (step15500, +12.806 KD backlog = 32.390 KD tổng, vẫn ~0,84% mix): chrF
     đứng yên (41,4 FLORES; 34,1 hardbench).
   - **Clean-finetune thử giả thuyết "KD bị pha loãng"** (+300 step từ v4, mix KD
     ĐẬM 35%: 120k base replay + 32.390 KD ×2 + fix ×2, `cloud/modal_finetune_kd.py`):
     milestone tốt nhất 15600 chỉ **+0,05 acc** (2,83 vs v4 2,78, đối đầu 30/45/25),
     15700/15800 TỤT dần (overfit) — giả thuyết pha loãng BỊ BÁC. 32k KD là quá ít,
     kể cả khi đậm đặc (`eval/judge_ft`).
   - **Rerank probe** (`eval/rerank_probe.py`): oracle 9 bản chỉ +5,2 chrF, LaBSE-rerank
     còn TỆ hơn greedy → chất lượng KHÔNG giấu trong weights, đòn bẩy decoding chết.
   - `>>fix<<` test tay 2 ca: 1 sửa đúng 1 sửa hỏng — không có nguồn sự thật, chỉ
     pattern-match; không cứu được lỗi thiếu kiến thức (lớp lỗi chủ đạo).
   **⇒ USER ĐÃ CHỐT (2026-07-21 tối): PHƯƠNG ÁN A — KD TOÀN PHẦN, 2 giai đoạn.**
   Chi tiết đầy đủ ở `PLAN_KD_JA2VI.md §7.3`: GĐ1 = 200k câu bằng gemini-flash-lite
   TRẢ PHÍ (~$5-20, rẻ hơn nhiều ước tính Haiku cũ vì 2.5-lite chỉ $0,10/$0,40
   per MTok) → train +2.500-3.000 step từ v4, mix KD 60-70%, **gate ≥ +0,3 acc**;
   GĐ2 nếu pass = dịch lại TOÀN BỘ 11,5M corpus (~$90-180) + pool 841k. KHÔNG
   spam thầy yếu (qwen-max/turbo/flash lỗi 11-17%, LaBSE không bắt được — §7.2).
   Một phần data sinh kèm ngữ cảnh (`ctx|||`) dạy context-awareness.
   **BLOCKER: chờ user bật billing 1 key Gemini** — xong là chạy pilot-audit 300
   câu rồi bung GĐ1. Nhánh B đề xuất song song (chưa chốt, `PLAN_KD_JA2VI.md
   §7.4`): pilot fine-tune Qwen3-1.7B/4B trên data KD sẵn có (~$5-15) = baseline
   CLAUDE.md §6 chưa từng làm — cho tier "hiểu ngôn ngữ" 1-2,5GB CPU.

   **📊 ERROR TAXONOMY 100 câu fail ja→vi (2026-07-21 tối, đề xuất reviewer ngoài,
   `eval/error_taxonomy_100.md` + `.json`):** phân loại nguyên nhân chính 100 câu
   fail (acc≤2) của v3 (66 hardbench + 34 FLORES): **sai nghĩa/vỡ cấu trúc 61%**,
   tên riêng/số liệu 12%, bỏ sót 8%, thuật ngữ 7%, keigo/register 6%,
   hallucination 5%, tiếng Việt vụng 1%. ⇒ **~74% là lỗi năng lực biểu diễn ngữ
   nghĩa** (câu ghép, passive, lồng vai — toàn từ phổ thông); chỉ ~25% là lớp
   data-nhắm-đích (glossary/keigo/domain) chạm tới được; idiom-mặt-chữ chỉ 9%.
   Hai pattern con: (a) **đảo vai/đảo phủ định 13%** — hiểu ngược 180°, nguy hiểm
   nhất, đáng làm metric riêng (đếm tỉ lệ lật chủ thể/phủ định trên probe cố
   định, rẻ hơn judge, nhạy hơn chrF); (b) lỗi tên riêng đa số là phiên âm
   katakana hỏng hệ thống (Miller→"gương", Mike→"micro"), glossary không cứu.
   Củng cố Phương án A: KD toàn phần đánh đúng khối SEM (dạy parse/giữ vai trên
   diện rộng); nếu KD toàn phần vẫn không phá được khối này → trần là sức chứa
   100M, chuyển hẳn sang 292M. Caveat: 1 người phân loại (Claude), 1 lượt, chưa
   kiểm chứng chéo — reviewer có thể tự phân loại lại mù từ file JSON để so khớp.

2. **Checkpoint/hạ tầng hiện có (2026-07-21):** 100M v4 step15500 = bản mạnh nhất
   nhánh 100M (`autosave-100m-v4`, v4_aa); ft15600 nhỉnh hơn không đáng kể
   (`autosave-100m-ft`). v3 step13000: `autosave-100m-v3`. 292M step30000 vẫn là
   nền 292M (`autosave-scale300m`, p_aa/p_ab). **2 fix tốc độ đã kiểm chứng trên
   L40S: `--pad-multiple 32` + `recompile_limit=64` (train.py tự set) → 0,89s/step
   ~50k tok/s sau warmup compile ~45' (so 2,65s/step của v3)** — warmup dài là
   một lần/container, đừng hoảng khi 30-80 step đầu chậm. Bug đã fix kèm:
   RoPE cache mặc định 256 crash với data `>>fix<<` 320 token (`--max-seq 384`);
   `import torch._dynamo` trong hàm gây UnboundLocalError (commit `cc87226`).
   Tốc độ inference CPU (WSL 6 luồng, i2_s 67,33MB): sinh 321 tok/s, ~155ms/câu.

3. **Data sẵn có cho bước tiếp:** 32.390 câu KD sạch (`data/synthetic/kd_clean/all_kd_v4.jsonl`),
   2.759 cặp fix (`data/synthetic/fix/all_fix_v3.jsonl`), pool JA 841k câu
   (`data/synthetic/kd/ja_pool.txt`) mới dùng ~35k → còn ~806k chưa dịch.
   Nguồn free đã cạn (DashScope vĩnh viễn, gemini-lite ~1k/ngày); OpenRouter
   free-tier: nạp $10 một lần mở 1.000 req/ngày (model mới nào cũng PHẢI qua
   pilot-audit 300 câu như quy trình cũ trước khi tin).

4. **292M (bidirectional) vẫn treo song song:** nền là `step30000` (KHÔNG dùng
   wave0/step32500). Nằm ở `p_aa`/`p_ab` trên `autosave-scale300m`.

5. Dọn dẹp không bắt buộc: `checkpoints_wave0/`, `premix_vong3a/`, `premix_ja2vi*/`,
   `premix_ft_*/`, `checkpoints_v3/`, `checkpoints_v4/`, `checkpoints_ft/` (local,
   đã `.gitignore`) — xoá được, mọi thứ đã an toàn trên GitHub Release.

## TL;DR

- **🔬 ĐỢT 0 XONG — KẾT LUẬN: KHÔNG DÙNG WAVE0, GIỮ STEP30000 (2026-07-20 tối).**
  Train +2500 step (30000→32500) trên Modal, LR êm (đỉnh 6e-5, không restart sốc) để phân
  biệt "nhiễu LR" vs "trần sức chứa" — nhưng ra kết quả THỨ BA không lường trước:
  **judge acc TB ja→vi TỤT từ 2.37 → 1.84** (vi→ja 1.35→1.06), **tệ hơn ở CẢ 10/10 domain**
  (it_deep −1.15, thanhngu −1.15, zeropronoun −0.75...). Đã kiểm chứng đây KHÔNG phải nhiễu
  giám khảo: 3 hệ tĩnh Google/Haiku/Fable (bản dịch cố định, chỉ đổi giám khảo giữa 2 lượt
  chấm) chỉ lệch ±0.1-0.15 giữa 2 lượt — 292M lệch −0.53, gấp 3-4 lần sàn nhiễu. **Chẩn đoán:
  OVERFIT** — loss train giảm (~1.0-1.1, thấp hơn hẳn các mốc trước) trong khi judge held-out
  tụt đều = đúng chữ ký overfit; train thêm trên CÙNG data vòng3a (không có câu mới) chỉ có
  hại vì corpus đã "vắt kiệt" từ vòng train trước. **Quyết định: mọi train tiếp theo dùng
  step30000 làm nền, KHÔNG dùng wave0/step32500.** Chi tiết + phương pháp kiểm chứng:
  `PLAN_KD_JA2VI.md` §2 W0.5 (cần cập nhật lại theo kết quả này — xem TODO).
  Script: `cloud/modal_train_wave0.py`. Checkpoint đã tải về local: `checkpoints_wave0/last_w0.pt`.

- **✅ TUYỂN THẦY KD ĐỢT 1 XONG (2026-07-20) — roster: gemini-flash-lite-latest + qwen-plus.**
  Benchmark 7 model free-quota (hardbench200 + judge mù 15 giám khảo, `eval/judge_teacher/`):
  gemini-lite thắng Google **10/10 domain ja→vi** (acc 4.90 vs Google 3.22, đối đầu theo câu
  170 thắng/21 hòa/9 thua/200) và thắng cả Haiku (4.10); qwen-plus theo sát (acc 4.72).
  **Pilot 300 câu/model + audit lớp 3 phân tầng (n=30-100/model) LOẠI qwen-max (16.7% lỗi,
  lẫn tiếng Trung + false-friend "エージェント→đại lý"), qwen-turbo (13%, đảo phủ định
  "貫く→xuyên thủng"), qwen-flash (11% xác nhận qua mẫu mở rộng n=100, false-friend
  "イメージ→hình ảnh" thay vì Docker image, "枯れている→lỗi thời" thay vì "đã ổn định"
  — ĐẢO hàm ý tích cực→tiêu cực).** Đã thử "khử câu lỗi" bằng verification (nhờ qwen-plus
  kiểm tra lại bản dịch qwen-max) — **THẤT BẠI hoàn toàn (0/5 lỗi đã biết bắt được, 2 báo
  nhầm)** — không có cách rẻ để cứu 3 model yếu, chỉ dùng LaBSE≥0.55 + rule filter + 2 thầy
  sạch. Trần khối lượng: qwen-plus ~12.000 câu (giới hạn input token, tổng cố định không
  theo ngày) + gemini-lite ~1.000 câu/ngày (2 key) → **~13.000 câu ngày đầu, ~19.000 câu
  sau 1 tuần, 0 đồng**. Đã thử thêm gemini-2.5-flash (hết quota 429 ngay), gemini-3-flash-preview
  (đúng nhưng 7-11s/câu, quá chậm), gemini-3.1-flash-live-preview/native-audio-dialog (chỉ
  Live API/WebSocket, không gọi được qua REST), gemini-3.5-flash (chất lượng tốt chrF 61.6
  nhưng hết quota sau 6-7 câu) — đều không dùng được. Pipeline sẵn sàng:
  `scripts/gen_kd_corpus.py` (dịch hàng loạt) → `scripts/filter_kd_corpus.py` (rule filter,
  đã bỏ check "..."/"anh chị" gây báo động giả) → `scripts/labse_score.py` (ngưỡng 0.55,
  KHÔNG phải 0.80 — đã hiệu chỉnh qua pilot, xem PLAN_KD_JA2VI.md).

- **🎯 PIVOT CHIẾN LƯỢC (2026-07-20, `PLAN_KD_JA2VI.md`):** user chốt **ja→vi là sản phẩm chính**. Sau 3 vòng data + scale, TB judge ja→vi đứng yên (2.40@19100 → 2.37@30000) — data niche chỉ đảo chỗ điểm (thanhngu +0.62, zeropronoun +0.58 NHƯNG keigo −0.78, nguphap −0.72, hop −0.55), gap lớn nhất với Google là năng lực lõi (nguphap −2.0, caudai −1.95, hop −1.9). Chuyển sang **sequence-level KD từ Haiku** (thắng Google 10/10 domain ja→vi trên hardbench, đối đầu 128/55/17): Đợt 0 = +2500 step LR êm + checkpoint averaging để phân biệt nhiễu-LR vs trần sức chứa (`cloud/modal_train_wave0.py`, ~$5); Đợt 1 = Haiku dịch 300k câu JA nhắm domain yếu (~$60-150 Batch API); Đợt 2 = train mix nghiêng ja→vi 70-75%. BỎ: data idiom kiểu vòng 3a, BT phục vụ vi→ja, bàn scale.
  **CẬP NHẬT: Đợt 1 đổi thầy từ Haiku trả phí sang gemini-lite+qwen-plus free (kết quả tốt
  hơn Haiku), Đợt 0 đã xong với kết luận bất ngờ (xem mục ĐỢT 0 ở trên) — Đợt 2 sẽ dùng
  step30000 làm nền, KHÔNG phải wave0.**

- **📎 GPU_PROVIDERS.md mới (2026-07-20):** khảo sát ~35 provider thuê GPU (khác API LLM
  sinh data ở PROVIDERS.md) — kết luận: không ai cho free thật/tự phục vụ/tái tục nhiều giờ;
  Modal $30/tháng vẫn là lựa chọn tốt nhất để burst A100 khi cần. Tham khảo nếu cần train
  ngoài local 3060Ti/Modal hiện tại.
- **✅ PHASE 1 (vòng 3a) XONG + GATE G1 TRƯỢT (2026-07-19):** 292M train 25000→30000 trên Modal L40S (lr-anchor 25000). Gate: judge acc 1.35 (vi→ja) / **2.37 (ja→vi)** vs Google 3.65/3.67 — thua cả 5/5 domain G1, đối đầu 12 thắng/19 hòa/169 thua. FLORES đã vượt 110M từ lâu (44.47/22.55 @19100) nhưng không phải thước đo Google. Kết quả: `eval/judge_4way_30000/`, so 19100: `eval/judge_4way_292m/`.
- **✅ BT VÒNG 3B XONG 100% (2026-07-19):** 841.039/841.039 câu JA mono dịch ja→vi bằng 292M@30000 trên Modal L40S — release **`bt-vong3b`** (bt_vong3b.ja/.vi, 92.9/92.6MB). Theo pivot trên: KHÔNG trộn vào mix (phục vụ vi→ja đã hạ ưu tiên); nguồn JA 841k tái dùng làm đầu vào KD Đợt 1.
- **📦 DATA VÒNG 3a XONG TOÀN BỘ (2026-07-18):** 66.3k record mới qua 5 lớp lọc (rule filter → Haiku 30 panel review → Fable phán xử → blacklist 139 → LaBSE) = 6.615 câu idiom/slang/tương phản/discourse-marker (provider free: 4 model Qwen quota riêng + 2 Gemini; GLM loại cả batch) + 50k cặp `ctx|||src` một-chiều (code, OpenSubtitles/TED) + 5.457 số liệu/phủ định (code) + 4.232 glossary-inject `[term=訳語]` + từ điển 1.247 mục (206 gốc Việt). **Premix `train-assets-vong3a` đã build (23,45M seq, new 18,7%, LaBSE GPU local) + upload.** Thẩm định độc lập Opus: `eval/DATA_REVIEW_VONG3_OPUS.md` (G1 cần thêm ctx||| + số liệu — ĐÃ làm; trần idiom ≤2k mục, đo term-rate trước khi nở). **Việc sau khi 292M đạt 25000: xem PLAN_RANKUP_292M.md §9** (gate 4-way → train vòng 3a +4-6k step lr-anchor 25000 → đo G1 + term-rate).
- **🗺️ PLAN LEO HẠNG (2026-07-17, `PLAN_RANKUP_292M.md`):** kế hoạch 4-5 tuần "đánh bại từng bậc" sau khi có 292M — thắng THẬT Google theo lộ trình domain (G1: ≥3 domain ja→vi khó; G2: toàn chiều ja→vi trên hardbench), thắng Haiku CHỈ trên glossary-bench nội bộ (term-accuracy), Fable chỉ thắng trục vận hành (tốc độ/0đ/offline — đo thành số). Haiku ($1/$5) được chốt làm THẦY sinh data/sửa lỗi thay Fable, ngân sách ước ~$150-250.
- **📊 BENCHMARK 4 HỆ + CLAUDE HAIKU (2026-07-17, `eval/hardbench_4way_haiku_report.md`):** thêm Claude Haiku 4.5 vào harness, chấm mù lại cả 4 hệ (15 giám khảo, 2 tiêu chí acc/nat). Kết quả acc vi→ja/ja→vi: 110M **1.22/1.90**, Google 3.88/3.73, **Haiku 4.69/4.45**, Fable 5.00/5.00; % dùng được (acc≥4): 6% / 60% / 92.5% / 100%. Haiku thắng Google 122/53/25 theo câu (mạnh nhất ở thành ngữ/slang/hội thoại — chỗ Google bắc cầu EN gãy), nhưng FLORES trung lập Haiku 37.2/51.1 vẫn dưới Google 42.6/53.5. Số 110M/Google tái lập khớp bản 16/07 → harness ổn định, dùng lại cho gate 292M. Tốc độ 110M đo lại máy công ty (Core Ultra 5 225H): **140 tok/s** @ 4-6 luồng (8 luồng 102, 12 luồng 21 — E-core làm hại); file 69MB. Haiku $1/$5 per MTok — ứng viên rẻ để sinh BT/data vòng 3 thay Fable.
- **⏸️ TRAIN 292M TẠM DỪNG — user đã XÓA node `O-1958273` (2026-07-16 ~23:40 giờ VN), MAI thuê node mới chạy tiếp.** Tiến độ đã an toàn trên release **`autosave-scale300m`**: bộ mới nhất = **`last_b` @ step 1375/25000** (loss ~2.67, `.step` up cuối 16:37 UTC = bộ đủ; bộ `last_a` @ step 970 là bản cũ hơn). Còn ~23,6k step ≈ **~31,5h GPU** (12,25k tok/s / 4,79s/step với `M300_MT=4096 M300_GA=32` + compile). **Thủ tục node mới (đã có sẵn, không phải nghĩ lại):** thuê node 12GB+ → clone repo + `gh auth login` → `apt-get install -y gcc` (Triton) → `bash cloud/restore_300m.sh` (tự tải premix bin từ `train-assets-vong2` + ghép bộ autosave mới nhất + đặt marker `.scale300m`) → `bash cloud/run_300m.sh` (tự resume từ last.pt, KHÔNG dọn nhầm nhờ marker) + bật lại daemon `cloud/backup_300m.sh` + watcher `WATCH_TAG=scale300m-step25000 bash cloud/watch_vong1.sh`.
- **✅ VÒNG 2 XONG (19000→23000, node `O-1957139`) + ĐÃ BENCHMARK (2026-07-16 chiều, i2_s local):**
  - chrF FLORES n=100: vi→ja **21.49** (±0 so 19000), ja→vi **42.27** (−0.6, trong nhiễu).
  - probe64: **họp vi→ja 27.4→35.4 (+8.0)** ⭐, câu khó vi→ja +6.5; NHƯNG **hội thoại vi→ja chỉ +0.4** (domain nhắm chính!) và loạt domain cũ tụt: IT vi→ja −3.7, câu khó ja→vi −5.9, họp ja→vi −4.4. Glossary term-trần đứng yên 28%.
  - **4/4 tín hiệu trần sức chứa BẬT (T1 T2 T3 T4)** — chi tiết `eval/capacity_log.md` + PLAN §3.5. Theo luật §5.1: **CHỐT SCALE from-scratch**; user chọn ~292M (bullet dưới). Data vòng 2 đã eyeball chống dương tính giả — sạch, không phải lỗi data.
- **📊 HARDBENCH 200 câu khó vs Google vs Claude (2026-07-16 tối, `eval/hardbench_report.md`):** bộ đề mới 200 câu tự soạn (10 domain khó × 2 chiều: keigo/thành ngữ/slang/zero-pronoun/số liệu/câu dài/ngữ pháp bẫy/IT/họp/hội thoại), 3 hệ chấm mù bởi 8 trọng tài Claude (0-5). Kết quả: **110M 1.43 (vi→ja) / 1.95 (ja→vi), Google 3.64/3.60, Claude 5.00/5.00**; tỉ lệ "dùng được" (≥4): 110M **5.5%**, Google 57.5%, Claude ~100%. Đối đầu 110M vs Google: 3 thắng/20 hòa/177 thua — 3 câu thắng ĐỀU là công sở IT ja→vi (đúng niche PLAN §9; Google dính lỗi bắc cầu EN: 議事録→"những phút giây", "gấu"→クマ). Bài học đo lường: **chrF đánh giá thấp khoảng cách thật trên câu khó** (ja→vi chỉ kém 3.5 chrF nhưng kém 1.65 điểm judge) → gate 292M nên chạy lại hardbench y hệt (lệnh trong report §7) chứ không chỉ FLORES. Vực sâu nhất: câu dài vi→ja (judge 1.00), slang/IT vi→ja (1.10) → data vòng 3 + BT dồn chiều vi→ja. **Bổ sung cùng phiên (report §8-10):** câu ngắn ja→vi 110M bám sát Google (probe64: 48.7 vs 52.2, thắng hội thoại 56.5 vs 52.3); FLORES n=100 trung lập 3 hệ: 110M 21.5/42.3, **Google 42.6/53.5** (vi→ja lần đầu đo), Claude 40.2/54.1. Ý tưởng "thinking/thầy-trò" của user đã chốt hướng thành **PLAN §4.3** (glossary-injection → thầy-trò lúc train → self-edit 2 lượt; KHÔNG CoT token).
- **🔄 RUN FROM-SCRATCH ~292M (khởi động 2026-07-16 tối trên node `O-1958273` — ĐÃ XÓA, xem bullet ⏸️ trên cùng để resume):** 291.8M params (d1152/16L/18H/ff3072), 0 → 25.000 step, LR 2.5e-4 (warmup 1000). **Cấu hình CHỐT sau 3 lần dò OOM: `M300_MT=4096 M300_GA=32` + compile** (budget 131k tok/step giữ nguyên) → **12,25k tok/s / 4,78s/step, tổng ~33h GPU (~1,4 ngày)**. Watcher: `WATCH_TAG=scale300m-step25000 bash cloud/watch_vong1.sh`.
  **Bài học node 12GB (đã ghi vào run_300m.sh):** eager OOM cả ở MT=4096 (STE `x.float()` tạo bản sao fp32 mỗi BitLinear — eager ngốn hơn compile nhiều); compile + MT=8192 cũng OOM; **compile + MT=4096 vừa khít**. Node thiếu gcc phải `apt-get install -y gcc` (Triton cần C compiler). Cần gạt dự phòng mới: `M300_GC=1` (gradient checkpointing). Node cũ O-1957139 (Blackwell, bệnh Triton `device not ready`) + n1: **tắt được, không còn gì độc nhất**.
  Code hỗ trợ đã test local: `train.py`/`convert_to_gguf.py` nhận dims (converter tự đọc cfg từ ckpt — watcher đóng gói 292M không cần sửa); premix bin đã upload lên `train-assets-vong2` (452MB) — node sau này prep chỉ cần `gh release download` + `run_300m.sh`.
  **🛟 AUTOSAVE chống node chết/hết tiền:** daemon `cloud/backup_300m.sh` (chạy nền cùng train) upload `last.pt` mỗi 30' lên release cố định **`autosave-scale300m`** — clobber tại chỗ, split 2 phần <2GB (file 3,5GB vượt hạn GitHub), 2 bộ a/b luân phiên (bộ up dở không phá bộ lành; file `.step` up cuối = chứng nhận đủ bộ). Node bay màu → node mới: clone + gh auth + **`bash cloud/restore_300m.sh`** (tự tải bin + ghép bộ mới nhất + đặt marker) rồi chạy lại run_300m.sh. Kèm vá mìn: run_300m.sh giờ có marker `.scale300m` — chạy lại script KHÔNG dọn nhầm last.pt 292M (trước đây sẽ làm train lại từ 0).
- **Gate nhận model 292M (sau ~2 ngày):** phải THẮNG 110M step23000 trên cùng harness — mốc phải vượt: FLORES vi→ja 21.49 / ja→vi 42.27; probe64 TB vi2ja 31.7 / ja2vi 48.7; glossary 28%. Thua = nút thắt là data → dồn sức BT (kế hoạch gốc 500k-1M câu JA, mới dùng 86k). Quyết định "thinking"/ngữ cảnh: KHÔNG CoT (nhỏ quá, phá latency); ngữ cảnh zero-pronoun làm ở vòng 3 bằng data format (không đổi arch) — PLAN §3.6.
- **👀 Việc user lúc rảnh (trong 2 ngày chờ):** eyeball `eval/probe64_step23000.jsonl` của 110M — làm mốc so sánh bằng mắt khi 292M ra lò.
- **Sự cố node n1 sáng nay: ĐÃ XỬ LÝ XONG** — n1.ckey.vn lỗi GPU/I-O ở ~step 19530, chuyển node mới + gói mix sẵn `bin_mix_vong2.tar.zst` (prep_vong2.sh chế độ PREMIX), KHÔNG mất data. (Backup khẩn cấp cũ trên release `backup-vong2-step<N>` chỉ còn giá trị tham khảo.)
  - **Hậu truyện n1 (tối 2026-07-16, phiên Claude máy công ty):** nguyên nhân "treo" là **NVML/nvidia-smi wedge + I/O stall tạm thời** — `watch.sh`/panel ckey chết vì query NVML, nhưng CUDA vẫn chạy → **n1 tự hồi và cũng train xong 19000→23000 (loss 1.3519)**, thành run TRÙNG LẶP với bản chính thức trên O-1957139. Checkpoint n1 đẩy lên tag **`backup-n1-step23000`** (ghi rõ "KHÔNG phải bản đã eval") — nhưng **n1 biến mất (~21:20, host key port 1638 ĐÃ ĐỔI — bị tắt/thu hồi) giữa lúc upload ⇒ tag này CÓ THỂ THIẾU asset, kiểm tra `gh release view backup-n1-step23000` trước khi tin**. Không sao: mọi thứ độc nhất đã nằm nơi khác (`last_final_23000.pt` @ `vong2-step23000`, premix @ `train-assets-vong2`). KHÔNG SSH lại port đó nữa (identity mới, có thể là máy người khác). Bài học vận hành: chẩn đoán node "treo" bằng lệnh nhẹ trước (`tail train.log`, `pgrep -af train.py`), bọc `timeout 8` cho nvidia-smi; SSH non-interactive nhớ `export PATH=/opt/conda/bin:$PATH`; `pkill/pgrep -f` phải dùng pattern bracket (`'[f]oo'`) kẻo tự khớp chính phiên SSH.
- **Gate sức chứa** (quyết định khi nào scale 110M→200M): PLAN_BUOC5 §5.1 + bảng điền từng vòng `eval/capacity_log.md`.
- **✅ VÒNG 1 XONG (14000→19000), gate PASS có điều kiện (2026-07-16):** probe64 IT vi→ja **+14.6 chrF**, câu khó +10.5, họp +7.9; FLORES không giảm (vi→ja 21.45, ja→vi 42.88); glossary term-trần 24%→28%. Assets đầy đủ trên Release `vong1-step19000`. Chi tiết: PLAN_BUOC5 §2.8.
- **Base FINAL step 14000 (mốc so sánh):** chrF FLORES n=100 (i2_s): vi→ja **21.31**, ja→vi **41.72** (Google ja→vi 54.0). Probe 64 câu: 14/64 ok (vi→ja 3/32, ja→vi 11/32). Model + checkpoint + data đầy đủ trên Release `v1.0-step14000` + `train-assets-step14000`.
- **Model chạy CPU thật:** 344-465 tok/s (5600X, 6 luồng, i2_s 68MB, RAM ~94MB), dịch tốt 2 chiều qua bitnet.cpp (llama-cli đã patch).
- **🎉 BUG CÂM: GIẢI XONG HOÀN TOÀN (2026-07-14). Model step4000 DỊCH ĐÚNG trên bitnet.cpp i2_s**: `猫が好きです。→Tôi thích mèo.`, `おはようございます。→Chào buổi sáng.`, `これはテストです。→Đây là bài kiểm tra.` Bench máy công ty: **432 tok/s** tg, 3345 tok/s pp (6 luồng). Là **HAI bug chồng nhau** (fix một cái vẫn câm — vì thế mới khó dò):
  - **Bug A (chỉ i2_s):** `llama-quantize` mặc định đa luồng ghi hỏng i2_s — scale per-tensor bị lạc chỗ, runtime đọc 0.0 → mọi matmul ternary = 0. **Fix: `llama-quantize <f32> <i2s> I2_S 1`** (số 1 = nthreads, bắt buộc).
  - **Bug B (MỌI quant, kể cả F32/F16/Q8_0):** `llama-cli` main.cpp:909 break `[end of text]` khi **token cuối của PROMPT** là EOG — mà format model bắt buộc prompt kết thúc `</s>` → thoát trước khi sinh token nào (kể cả `--ignore-eos`). Đây là quirk frontend llama-cli, KHÔNG phải lỗi model/runtime/converter. **Fix: patch 3 dòng** (`scripts/llama-cli-eog-prompt.patch`) — chỉ break khi EOG do model sinh ra. API llama.cpp thuần (llama-server, bindings) không dính.

## ✅ XÁC NHẬN THỰC NGHIỆM (2026-07-14, build bitnet.cpp trên máy công ty, model untrained)
Thí nghiệm A/B/C với converter HIỆN TẠI (untrained, prompt `<s>>>jpn<<Tôi thích mèo.` KHÔNG có `</s>` cuối):
- **F32**: babble `thiệt thiệt Tiện thiệt責任がある...` → graph/metadata/tokenizer ĐÚNG.
- **i2_s quantize `I2_S 1`** (single-thread): output **GIỐNG HỆT F32 từng token** → toàn bộ đường i2_s ĐÚNG. Hex-check: scale tại `n/4` = 0.015954 = đúng mean|w|.
- **i2_s quantize đa luồng** (lệnh cũ): `<pad><pad><pad>...` thoái hóa. Hex-check: scale = **-2.6e-19 ≈ 0** → mọi matmul ternary × 0.

**Bug B — cơ chế chính xác (đã xác nhận bằng --verbose-prompt):**
- llama-cli tokenize prompt `<s>>>jpn<<Tôi thích mèo.</s>` ra ĐÚNG `[2,5,344,555,3189,263,3]` — khớp PyTorch từng id. Tokenizer/converter vô can.
- Nhưng `sampling: 7 runs` = đúng số token prompt, 0 token sinh ra: main.cpp:909 `if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !params.interactive) break;` — sau khi nạp prompt, `embd.back()` chính là `</s>` của prompt → break luôn. `--ignore-eos` chỉ chặn EOS ở sampler, không chặn check này.
- Patch (đã lưu `scripts/llama-cli-eog-prompt.patch`, áp vào `3rdparty/llama.cpp/examples/main/main.cpp`): thêm `bool did_sample=false;` (dòng ~500), set `did_sample=true;` sau `embd.push_back(id)` ở nhánh sampling (~688), đổi điều kiện dòng 909 thành `if (did_sample && ...)`.

**Vì sao chẩn đoán trước đây lạc lối:** hai bug CHỒNG NHAU (fix A vẫn câm vì B; fix B vẫn câm vì A ở i2_s), và test "untrained cũng câm" không có giá trị (untrained + Bug B → câm là tất nhiên). Các quan sát "matmul ternary = 0" (Bug A) và "Q8_0 nonzero vẫn câm" (Bug B) thực ra đã trỏ đúng hai bug khác nhau.

**Ghi chú build trên máy mới (WSL Ubuntu):** cần `python3 utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32` trước cmake (sinh `include/bitnet-lut-kernels.h`); main branch hiện lỗi const ở `src/ggml-bitnet-mad.cpp:811` với clang 18 (vá: thêm `const` cho `y_col`). gguf-py của fork chưa biết type I2_S=36 và vỡ với numpy≥2 (Reader) — đọc file i2_s phải parse tay.

**Kết quả cuối (máy công ty, step4000, i2_s 68MB):** 6/6 câu ra bản dịch hợp lý; 4/6 khớp PyTorch từng ký tự; bench 432 tok/s (tg64, 6 threads). Pipeline chuẩn từ giờ:
```
python scripts/convert_to_gguf.py --ckpt <ckpt> --out dist/x_f32.gguf
llama-quantize dist/x_f32.gguf dist/x_i2s.gguf I2_S 1        # số 1 BẮT BUỘC
llama-cli (đã patch) -m dist/x_i2s.gguf --special -p "<s>>>jpn<<...</s>" --temp 0
```

## Kết luận điều tra bug câm (2026-07-14, đọc source microsoft/BitNet main + llama.cpp fork)
**Nguyên nhân i2_s (cơ chế đã xác minh trong source):**
- `llama-quantize` không truyền nthreads → mặc định `hardware_concurrency()` (llama.cpp:18752) → tensor 768×768 bị chia ~35 chunk chạy song song.
- `ggml_quantize_chunk` (ggml.c:22657) đặt chunk tại `start_row*row_size` với row_size = n_per_row BYTE — gấp 4 lần stride thật của dữ liệu 2-bit packed → dữ liệu rải rác, per-tensor scale bị đè/lạc chỗ.
- Runtime đọc scale duy nhất tại offset `ne00*ne01/4` (ggml.c:12457/13282) → trúng vùng chưa ghi = **0.0** → `out = (dot−act_sum)/act_scale × 0.0 = 0` cho MỌI phần tử — khớp chính xác quan sát "mọi matmul ternary ra 0".
- Pipeline chính chủ luôn quantize single-thread: `llama-quantize ... I2_S 1` (setup_env.py:139-146). Lệnh trong STATUS/docstring cũ của ta THIẾU số `1`.
- Lưu ý test md5 cũ: nếu input đưa vào llama-quantize đã là i2_s thì đi đường copy nguyên văn (llama.cpp:19015) — md5 khớp KHÔNG chứng minh gì về packing. Test đó không mâu thuẫn với kết luận trên.

**Đã kiểm chứng KHÔNG PHẢI nguyên nhân (đọc source, có bằng chứng dòng lệnh cụ thể):**
- ~~Thiếu output.weight~~: loader có fallback tied (llama.cpp:8724-8730), và `build_bitnet_158` dùng thẳng `tok_embd` làm lm_head (llama.cpp:15530) — GGUF 2B-4T chính chủ cũng KHÔNG có output.weight.
- ~~Pre-scale {-m,0,+m} sai~~: `quantize_i2_s` = per-tensor max|w| + sign + ngưỡng 0 là 1e-6 (ggml-bitnet-mad.cpp:59-72) → {-m,0,+m} tái tạo CHÍNH XÁC t·m. Converter chính chủ cũng pre-ternarize absmean y hệt (convert-hf-to-gguf-bitnet.py:970-975). Thiết kế của ta ĐÚNG.
- ~~Graph lệch~~: walk op-by-op build_bitnet_158 (llama.cpp:15389-15537) vs src/bitnet.py: NEOX rope, relu²·up, sub_norm trước wo/down, kq_scale 1/√64, không embedding scale — **0 khác biệt**. B158 KHÔNG load tensor *_scale nào (chỉ arch "bitnet" cũ mới load).
- ~~Token id~~: converter ta GHI tường minh bos=2/eos=3/unk=1/pad=0 (nếu thiếu thì SPM default bos=1/eos=2 → '<s>' của ta thành EOS — bẫy này ta đã né sẵn).

**Route "converter chính chủ" là BẪY cho model ta (bỏ hướng này):**
- `convert-hf-to-gguf-bitnet.py` emit arch **"bitnet"** → dispatch build_bitnet **SiLU** (llama.cpp:15239) — SAI với model relu² của ta.
- `convert-ms-to-gguf-bitnet.py` emit arch "bitnet-25" (OK graph) nhưng HARDCODE rope_freqs base 500000 dim 128 + vocab BPE — độc với model ta.
- Converter tự viết của ta được xác nhận tương đương semantics với flow chính chủ → GIỮ NGUYÊN, chỉ cần fix lệnh quantize.

## HƯỚNG TIẾP (retest với 2 fix)
1. Convert F32 GGUF bằng converter hiện tại (`--ckpt` checkpoint relu² mới) → `llama-quantize <f32> <i2s> I2_S 1` (**số 1 bắt buộc**) → `llama-cli` test.
2. Sanity gate trước khi test: đọc 4 byte tại offset `ne0*ne1/4` trong blob i2_s của `blk.0.attn_q` → float phải ≈ mean|w| của tensor PyTorch (≠ 0).
3. Nếu i2_s hết câm → xong, đo chrF + benchmark. Nếu F16 vẫn câm với converter HIỆN TẠI → mới cần dò tiếp (khi đó dump element-wise, KHÔNG dùng eval-callback sum).
4. Quantize và chạy PHẢI cùng một build (i2_s có 2 layout packing gated bởi ACT_PARALLEL trong gemm-config.h — trộn binary khác vintage là hỏng ngầm).

## Đã làm xong
1. **Data (Bước 1-3):** 5.40M cặp sạch ở `data/clean`, tokenizer SPM 32k, binarize → `data/bin`.
2. **Arch (Bước 4):** decoder-only BitNet b1.58, **FFN squared-ReLU** `down(relu(gate(x))²·up(x))` (KHÔNG SwiGLU — để khớp `build_bitnet_158` của bitnet.cpp), RoPE NEOX, RMSNorm SubLN, tied embed, d_model 768/12 layer/12 head/d_ff 2048 = 109.6M params.
3. **Converter** `scripts/convert_to_gguf.py`: arch `"bitnet-b1.58"`, áp `weight_quant` (ternary {-m,0,+m}) trước khi lưu, `add_rope_dimension_count`, cờ `--f16` (tải nhẹ), `--d-model/--d-ff/...` (test dims).
4. **Train lại relu² từ đầu trên cloud** (A4000). Xem "Cloud" bên dưới.

## Bug deploy — ĐÃ LOẠI TRỪ (đừng dò lại)
- Template: prompt đúng `<s>>>jpn<<{src}</s>` KHÔNG cách sau thẻ → token `[2,5,...,3]` chuẩn. Chạy `llama-cli --special`.
- Weight scheme: đã thêm weight_quant vào converter.
- Packing i2_s: `llama-quantize` của build này tạo file **byte-identical** với official (cùng md5 khi quantize model `large`).
- Kích thước: untrained ở dims 2B qua converter TÔI cũng câm ⇒ không phải dims.
- Activation outlier: attn_norm absmax 2.87, ratio 3.5, 765/768 phần tử ≠0 sau int8 → không crush.
- Embedding lookup: giá trị `inp_embd` khớp `embed[3189]` cả đầu lẫn cuối.
- i2_s: mọi matmul ternary ra **0** (kernel MAD). Q8_0: matmul ra nonzero nhưng vẫn câm.
- **Lưu ý:** `llama-eval-callback` báo "sum" KHÔNG đáng tin (0.0776 cho vector sum thật 0.4165) → không dùng "sum" để so lớp. Muốn so phải dump giá trị element hoặc viết harness khác.

## Scripts debug (đã có trong repo)
- `scripts/gguf_pytorch_check.py <gguf>`: nạp trọng số GGUF → PyTorch → generate (chứng minh model đúng).
- `scripts/test_ckpt_translate.py <ckpt.pt>`: test dịch nhanh 1 checkpoint (frozen = i2_s), chạy CPU.
- `scripts/compare_logits.py <gguf>`: top-8 token kế tiếp từ PyTorch.
- `scripts/trace_compare.py`: dump sum từng lớp layer-0 (LƯU Ý eval-callback sum không đáng tin).

## Cloud (A4000, ckey.vn)
- SSH `root@n2.ckey.vn -p 2430` (mật khẩu user cấp riêng). python = `/opt/conda/bin/python3` (torch 2.5.1+cu124).
- Training relu² đang chạy `bash cloud/run_cloud.sh` (nohup, tự resume/restart, tới step 14000). Checkpoint mới nhất: `~/Train-model-translate/checkpoints/last.pt` (lưu mỗi 100 step). Backup run SiLU cũ ở `checkpoints/_silu_backup/`.
- Convert TRÊN cloud rồi tải i2_s (67MB) / F16 (269MB, `--f16`) về. Egress cloud CHẬM (~100-350KB/s) → resume tải bằng `tail -c +$((offset+1)) file >> local`.
- Đẩy file lên cloud khi scp fail: `base64 -w0 file` rồi `echo <b64> | base64 -d > remote_file`.

## Lệnh nhanh (local, venv ~/BitNet/.venv_bitnet)
```
python scripts/convert_to_gguf.py --ckpt <ckpt> --out dist/x_f32.gguf   # bỏ --ckpt = untrained
~/BitNet/build/bin/llama-quantize dist/x_f32.gguf dist/x_i2s.gguf I2_S 1   # số 1 = nthreads, BẮT BUỘC (đa luồng ghi hỏng i2_s)
~/BitNet/build/bin/llama-cli -m dist/x_i2s.gguf --special --no-display-prompt \
  -p "<s>>>jpn<<Tôi thích mèo.</s>" -n 32 -t 6 --temp 0 --no-warmup
~/BitNet/build/bin/llama-bench -m dist/x_i2s.gguf -t 6 -p 128 -n 128
```
