# PLAN Bước 5 — Nâng chất lượng dịch VI↔JA lên mức "dùng chuẩn" cho IT / hội thoại / họp / đời thường

> Lập 2026-07-14, dựa trên **probe 64 câu × 4 domain** chạy trên model step-14000 (i2_s, bitnet.cpp)
> + chrF FLORES: vi→ja **21.3**, ja→vi **41.7** (Google ja→vi: 54.0).
>
> **⏱ CẬP NHẬT 2026-07-16 chiều — VÒNG 2 XONG & ĐÃ ĐO (§3.5): 4/4 tín hiệu trần sức chứa → SCALE FROM-SCRATCH. User chốt ~292M (d1152/16L/ff3072). Kế hoạch chạy + timeline ~2 ngày: §3.6; script `cloud/run_300m.sh`. Vòng 3 (§4 + ngữ cảnh) chạy trên model 292M.**

## 0. Hiện trạng đo được (probe nghiêm khắc chuẩn dịch chuyên nghiệp)

| Domain | vi→ja ok | ja→vi ok | Tổng |
|---|---|---|---|
| IT | 3/8 | 3/8 | 6/16 |
| Hội thoại | **0/8** | 3/8 | 3/16 |
| Họp/công sở | **0/8** | 4/8 | 4/16 |
| Câu khó (dài, số liệu) | **0/8** | 1/8 | 1/16 |
| **Tổng** | **3/32 (9%)** | **11/32 (34%)** | 14/64 (22%) |

**5 nhóm lỗi gốc (xếp theo tần suất trong probe):**
1. **Chiều VI→JA sụp đổ** (sai_nghia 34, bo_sot 30 lần): câu Nhật ra cụt/vụn/hallucinate (`microservice→貴金属`, `sinh nhật→日本`). Chiều sinh tiếng Nhật thiếu data sạch.
2. **Thuật ngữ & từ mượn bị phá** (14 lần): katakana → phiên âm rác (`ライブラリ→Librali`, `システム→cysteinm`, `レストラン→Leastran`); từ EN xen trong câu Việt (deploy, pull request) làm câu vỡ.
3. **Số liệu / ngày giờ / phủ định mất hoặc sai** (so_lieu 8+, nhiều bo_sot): `12,5%→3.5%`, `thứ Tư→月曜日`, `午後→buổi trưa`, "vẫn KHÔNG tìm ra" → câu khẳng định. **Loại lỗi nguy hiểm nhất khi dùng cho họp.**
4. **Ngữ pháp/thì-thể** (19 lần): ý định tương lai → quá khứ, "sắp" mất sắc thái, câu cụt thiếu động từ/てください.
5. **Register/keigo sai** (5 lần): họp cần keigo nhưng ra thể thường; 申し訳ございません → "Xin lỗi" cụt.

## 1. Mục tiêu số hoá (sau 3 vòng)

| Chỉ số | Hiện tại | Đích vòng 3 |
|---|---|---|
| chrF ja→vi (FLORES devtest) | 41.7 | **≥ 48** (~90% Google) |
| chrF vi→ja (FLORES devtest) | 21.3 | **≥ 30** |
| Probe 64 câu — tỉ lệ ok | 22% | **≥ 55%** |
| Thuật ngữ IT trong glossary dịch đúng | (rác) | **≥ 90%** trên test set thuật ngữ |
| Số liệu/ngày giờ giữ đúng | hay mất | **≥ 95%** trên test set số liệu |

Không kỳ vọng vượt Google Translate (model 110M) — đích là **"tốt, dùng được"** đúng CLAUDE.md, và ĐẶC BIỆT mạnh hơn Google ở **thuật ngữ IT nội bộ** nhờ glossary riêng.

## 2. VÒNG 1 — Thuật ngữ IT + cứu chiều VI→JA *(tác động lớn nhất, làm trước)*

### 2.1 Data glossary (user cấp glossary JA-VI)
- Mỗi thuật ngữ → sinh **8-12 câu ví dụ** bằng LLM (ngữ cảnh thật: bug report, code review, deploy, hỏi đáp kỹ thuật), cả 2 chiều.
- Bắt buộc phủ 3 dạng viết: katakana (`ライブラリ`), EN nguyên dạng (`library`), Việt (`thư viện`) — trị tận gốc lỗi `Librali`.
- Ước lượng: 1.000 thuật ngữ × 10 câu ≈ **10k cặp**; oversample ×3 khi train.
- Giữ riêng 10% thuật ngữ làm **test set thuật ngữ** (không train).

### 2.2 Data IT docs (user cấp tài liệu đã dịch chuẩn)
- Tách câu (JA theo 。, VI theo .!?) → align bằng **LaBSE ≥ 0.75** → cặp câu domain chất lượng cao.
- Giữ riêng ~500 câu làm **IT-test** (không train).

### 2.3 Back-translation cứu chiều VI→JA *(chìa khoá của vòng này)*
- Nguyên tắc: muốn giỏi SINH tiếng Nhật thì **target JA phải là câu Nhật thật**.
- Lấy **500k–1M câu JA monolingual sạch** (từ data thô sẵn có: news/wiki/TED; ưu tiên văn phong hội thoại + công sở).
- Dùng chính model (chiều mạnh ja→vi, 41.7) dịch ra VI trên GPU → cặp (VI tổng hợp → **JA thật**).
- **Chỉ train chiều vi→ja** với các cặp này (không đảo chiều).

### 2.4 Data code-switching VI-EN + copy-through
- Probe cho thấy câu Việt chứa từ EN (deploy, backup, review…) làm model vỡ. Sinh **5-10k cặp** câu Việt trộn từ EN ↔ Nhật chuẩn (dân IT Việt nói chuyện kiểu này hằng ngày).
- **Copy-through augmentation** (sinh bằng CODE, ~0 token LLM): câu chứa chuỗi phải **GIỮ NGUYÊN VĂN** khi dịch — tên riêng lạ, mã version (v2.1.3), mã ticket (JIRA-1234), URL, tên file, chuỗi ngẫu nhiên. Dạy model kỹ năng "gặp cái lạ thì CHÉP, đừng bịa" → trị cả lớp lỗi phiên âm rác (`Librali`, `cysteinm`, `Leastran`). ~5k cặp, vài trăm khung câu là đủ.

### 2.5 Train
- Resume từ `cloud_backup/last_final_14000.pt` (có optimizer, đã verify).
- Mix: **65% replay data cũ + 35% data mới** (chống catastrophic forgetting); trong data mới oversample glossary ×3.
- +**5.000 step**, LR restart 1e-4 → cosine về 1e-5. A4000: ~5h.

### Gate vòng 1 (không đạt thì không sang vòng 2)
- chrF FLORES **không giảm** ở cả 2 chiều; vi→ja **≥ +4 điểm**.
- Test set thuật ngữ ≥ 70% đúng; re-run probe 64 câu: IT ok ≥ 9/16.

### 2.6 TRẠNG THÁI THU THẬP DATA — ✅ XONG (2026-07-15)

Toàn bộ data nhắm đích Vòng 1 đã sinh/gom (nằm trong `data/synthetic/`, `data/glossary/` — gitignore, đẩy lên **GitHub Release** không vào git).

| Khối | Cặp | Nguồn / script |
|---|---:|---|
| glossary_sents (câu ví dụ theo thuật ngữ) | 104.404 | `gen_via_api.py` (Gemini 2 key) + recover từ journal |
| ja_indomain_bt (câu JA cho back-translation) | 86.172 | `harvest_ja_indomain.py` (Qiita API) — *chưa dịch, để BT trên GPU* |
| it_docs (OPUS localization ja-vi) | 27.144 | GNOME/KDE/Ubuntu/PHP tách câu |
| glossary_direct (cặp term thẳng) | 17.951 | `glossary_merged.csv` (MS-terminology 16.459 + wikidata + user 427 + brse 76) |
| phase2 (code-switch + Q&A 指摘事項-style HỢP PHÁP) | 8.781 | `gen_phase2.py` (cs 4.656 + qa 4.125) |
| copythrough (giữ nguyên version/ticket/URL) | 3.448 | `gen_copythrough.py` (sinh bằng code) |
| codeswitch (VI-EN ↔ JA, sinh bằng code) | 201 | `gen_codeswitch.py` |
| **TỔNG nhắm đích** | **248.101** | *(+ base OPUS 5.4M cho replay)* |

**Glossary tổng:** 18.027 thuật ngữ (`data/glossary/glossary_merged.csv`); đã tách 10% test-set (`glossary_test*.{csv,txt}`, 1.674 term — KHÔNG train).

**Chạy trên GPU node — ĐÃ ĐÓNG GÓI THÀNH 1 LỆNH** `bash cloud/prep_vong1.sh`:
1. **Back-translation** (`cloud/backtranslate.py`): model dịch `ja_indomain_bt.ja` (86k JA thật) → VI → cặp (VI tổng hợp → JA thật), chỉ chiều vi→ja.
2. **Lọc + trộn** (`scripts/mix_and_binarize.py`): LaBSE ≥ 0.8 cho free-text synthetic + dedup + loại cặp trùng test-term; oversample glossary ×3; nhân data mới đạt ~30% → **trộn ở mức binary** với base (nối token stream, khỏi tokenize lại 5.4M base).
3. **Resume train** (`cloud/run_vong1.sh`): từ `last_final_14000.pt` +5.000 step (→19000) với **restart LR** (`train.py --lr-anchor 14000`, 1e-4→1e-5) — không restart thì LR ~min, không học.
4. Xong → convert GGUF → eval (probe64 + chrF FLORES) → gate dưới.

*Verify local: 7 nguồn load OK (161k cặp thô), loại 12k trùng test-term, seqs_for/tgt_start khớp `binarize.py`. bt.* sinh trên node.*

### 2.7 ĐANG TRAIN — cập nhật 2026-07-15 ~15:00

**Số thực của prep (đã chạy xong trên node RTX 5060 Ti 16GB):**
- **BT full 86.172 câu** → giữ **55.846 cặp** vi→ja (65%). Tốc độ 63 câu/s nhờ `backtranslate.py` bản BATCH (nhóm câu cùng độ dài, x20 so với batch=1 cũ chỉ 3 câu/s).
- **LaBSE ≥0.8 loại 45.874/129.959 cặp free-text (35%!)** → data mới sạch 189.670 cặp. (Lần mix đầu KHÔNG có LaBSE do thiếu sentence-transformers — đã dừng, khôi phục base, làm lại. LaBSE chạy trong venv CPU riêng `.venv_labse` vì cài vào conda global làm vỡ torchvision/transformers; dùng `MIX_PY=.venv_labse/bin/python3 bash cloud/prep_vong1.sh`.)
- **Mix:** base 10.79M + new 3,83M = **14,63M seq (new 26.2%)**, glossary hiệu dụng ×18, dev giữ nguyên base.

**Train:** resume 14105 → **19000**, LR restart 1e-4 (anchor 14000, warmup 200) ✓ xác nhận trong log; `VONG1_MT=16384 VONG1_GA=8` (16GB VRAM chứa thoải mái, không OOM); **~22,7k tok/s, 2.6s/step** (GPU 98% util, 56°C không throttle — đã thử cả 4096/32, 8192/16, 16384/8 đều ~20-23k ⇒ 22k là ~TRẦN của 5060 Ti, đừng kỳ vọng 30k như 4070 Ti). ETA ~3.5-4h.

**Tự động sau khi xong:** `cloud/watch_vong1.sh` (đang chạy nền, PID-lock) canh `VONG1 COMPLETE` → tự đóng gói (ckpt fp16, **GGUF F16**, bt.ja/vi, fp32+optimizer để resume Vòng 2, train.log) → **GitHub Release `vong1-step19000`**.

### 2.8 KẾT QUẢ GATE VÒNG 1 — đo 2026-07-16 (i2_s 68MB, bitnet.cpp, greedy)

**chrF FLORES devtest (n=100)** — `eval/eval_one.py`, cùng harness mọi mốc trước:
| Chiều | 14000 | 19000 | Δ |
|---|---|---|---|
| ja→vi | 41.72 | **42.88** | +1.2 ✓ không giảm |
| vi→ja | 21.31 | **21.45** | +0.1 ✓ không giảm (✗ chưa +4 trên FLORES) |

**probe64 — chrF trung bình theo domain×chiều** (`eval/run_probe64.py`, chi tiết từng câu ở `eval/probe64_step*.jsonl`):
| Domain | vi→ja 14k→19k | ja→vi 14k→19k |
|---|---|---|
| **IT** | 24.3 → **38.9 (+14.6)** ⭐ | 42.6 → 47.7 (+5.1) |
| Câu khó | 21.0 → **31.5 (+10.5)** | 49.6 → **58.7 (+9.1)** |
| Họp | 19.5 → **27.4 (+7.9)** | 42.6 → 45.1 (+2.5) |
| Hội thoại | 16.4 → 17.7 (+1.3) | 54.4 → 53.4 (−1.0) |
| **TB toàn probe** | 20.3 → **28.9 (+8.6)** | 47.3 → **51.2 (+3.9)** |

**Glossary test-set** (200 term held-out, JA term trần → VI): 24% → **28%** (✗ gate 70% — nhưng phép đo khắc nghiệt: term chưa từng thấy, không ngữ cảnh; cần đo lại bằng term-trong-câu ở vòng sau).

**Kết luận gate:** data nhắm đích ăn ĐÚNG chỗ nhắm — vi→ja IT +14.6, câu khó +10.5, họp +7.9; FLORES (news/wiki, ngoài domain nhắm) không giảm. Hội thoại đứng yên (đúng — là mục tiêu Vòng 2). Điểm yếu lộ ra: generalize term chưa thấy còn kém (28%).
→ **QUYẾT ĐỊNH: PASS có điều kiện — sang Vòng 2**; mang theo 2 việc: (a) đo thuật ngữ bằng term-trong-câu, (b) vi→ja FLORES tổng quát vẫn thấp (21.5) — kỳ vọng nhích tiếp nhờ data hội thoại Vòng 2. *(Chuẩn cuối vẫn là USER eyeball probe64 — file `eval/probe64_step19000.jsonl` sẵn để đọc.)*

## 3. VÒNG 2 — Hội thoại & Họp (register + keigo)

> **✅ DATA VÒNG 2 CHỐT 2026-07-16: 13.858 cặp sạch** (11 mode; release `train-assets-vong2`) — qua đủ 4 lớp QUALITY_GATE (xem `eval/QUALITY_GATE.md`: Claude xoá ~1.700 cặp/gloss lỗi qua các đợt đọc, blacklist 260 mục). Kiểm kê: idh 4.532 (476 idiom đã duyệt), ht 1.381, am 1.262 (ẩm thực/văn hoá — MỚI), dn+dn2 1.898, pb 1.055, id 1.052, pk 819 (phủ định kép — kéo sớm từ V3), hop 794, ps 571, g2 494.
> **Train:** `bash cloud/prep_vong2.sh` trên node (mặc định PREMIX: tải bin mix sẵn `bin_mix_vong2.tar.zst` + ckpt 19000 → train ngay 19000→23000, LR restart anchor 19000; `PREMIX=0` để tự mix như cũ). Gate ở §3.4.
> **🔄 ĐANG TRAIN (2026-07-16 ~11:45, node mới `O-1957139` sau sự cố n1):** step ~21100/23000, loss ~1.42, 25,1k tok/s / 2.32s/step, ETA ~13:00. Trước khi xong: bật watcher `WATCH_TAG=vong2-step23000 nohup bash cloud/watch_vong1.sh > checkpoints/watch.log 2>&1 &` — checklist sau-train ở STATUS.md TL;DR.

### 3.1 Data sinh bằng LLM (~1.5-2M token output ≈ 25-35k cặp)
- **Hội thoại đời thường** (theo khung JLPT N3→N1 đã có): rủ rê/hẹn, gọi món (phở bò! trà đá!), mua sắm, hỏi đường, thời tiết, sức khỏe, gia đình. Chú trọng: câu 2 vế ("Hôm nay rảnh không? Đi ăn nhé"), sắc thái **sắp/định/nhớ...nhé** ↔ 〜そう/つもり/忘れずに.
- **Họp & công sở**: mở/chốt họp, dời lịch, báo cáo tiến độ (%), deadline/納期, biên bản/議事録, phê duyệt/承認, xin phép vắng mặt. **Mỗi câu sinh 2 biến thể register**: keigo (丁寧語/謙譲語) + thể thường, gắn đúng ngữ cảnh.
- **Ẩm thực/văn hoá Việt**: danh mục món ăn, địa danh → trị lỗi `trà đá→石のお茶` (dịch word-by-word).
- **Phrasebook công thức** (~1.000 câu, ~50k token LLM): câu chào/công thức văn hoá dịch theo **CHỨC NĂNG** chứ không literal — お疲れ様です (≠"bạn đã mệt rồi"), よろしくお願いします (≠"xin đối xử tốt với tôi"), 頑張って, お世話になっております, mở/kết họp, mở/kết email, cảm ơn/xin lỗi khách. Rẻ mà ăn điểm cực lớn cho use case họp/công sở.
- **Hội thoại theo persona** (quan hệ rõ: sếp↔nhân viên, đồng nghiệp ngang hàng, bạn thân, khách hàng): dạy **xưng hô tiếng Việt đúng quan hệ** (anh/em/chị/mình/bạn) thay vì "tôi/bạn" đều tăm tắp như Google. Đây là chỗ model chuyên domain **vượt được Google** trong công sở IT.
- **Từ đa nghĩa tương phản**: ~200 từ bẫy JA (大丈夫, 結構, いい, ジョブ…) + ~200 từ VI (đá, cơm, được, nhà…), mỗi nghĩa 5-8 câu ngữ cảnh khác nhau cho ra bản dịch khác nhau.
- **[KÉO TỪ VÒNG 3 LÊN — theo phân tích 48 câu `eval/analysis_javi48_vong1.md`] Phủ định kép** ないことはない/なくもない/話せないこともない — model đang dịch **NGƯỢC nghĩa** (3/3 câu FAIL), nguy hiểm nhất về nghĩa; data sinh code+LLM rẻ nên làm sớm.
- **[MỚI] Glossary đợt 2 — term dev-workflow hiếm**: リベース(rebase), ホットフィックス, ロールアウト, カナリア, チェリーピック… (48 câu lộ hallucinate `リベース→"dính cơ sở dữ liệu"`); ~200-300 term + câu ví dụ.
- **[MỚI] Slang list hội thoại** ~100 từ: やばい・まじで・それな・寝坊した・ドタキャン… gắn vào phần hội thoại đời thường.

### 3.2 Tận dụng data sẵn có
- OpenSubtitles VI-JA (hội thoại tự nhiên, đã có trong data thô): lọc lại lấy top LaBSE ≥ 0.8, oversample ×2 phần hội thoại ngắn.

### 3.3 Train
- +**4.000 step**, mix 70% replay (gồm data vòng 1) + 30% mới.

### 3.4 Gate vòng 2
- Probe: HoiThoai ok ≥ 8/16, Hop ok ≥ 8/16; register keigo đúng trong bộ test họp; chrF không giảm (mốc 19000: vi→ja 21.45 / ja→vi 42.88).
- Việc mang theo từ gate Vòng 1: (a) đo glossary bằng **term-trong-câu** thay vì term trần; (b) check T3 — vi→ja FLORES sau vòng 2 vẫn < 27 là 1 tín hiệu trần.
- Điền dòng Vòng 2 vào `eval/capacity_log.md` (T1–T4, §5.1) rồi mới quyết Vòng 3 hay scale 200M.

### 3.5 KẾT QUẢ GATE VÒNG 2 — đo 2026-07-16 (i2_s 68MB, bitnet.cpp, greedy, cùng harness mọi mốc)

**chrF FLORES devtest (n=100)** — `eval/eval_one.py`:
| Chiều | 19000 | 23000 | Δ |
|---|---|---|---|
| ja→vi | 42.88 | 42.27 | −0.6 (trong nhiễu, không vi phạm gate) |
| vi→ja | 21.45 | **21.49** | +0.0 → vẫn < 27 ⇒ **tín hiệu T3** |

**probe64 — chrF trung bình theo domain×chiều** (`eval/probe64_step23000.jsonl` để đọc bằng mắt):
| Domain | vi→ja 19k→23k | ja→vi 19k→23k |
|---|---|---|
| **Hội thoại (nhắm)** | 17.7 → 18.1 (**+0.4** ✗) | 53.4 → 56.5 (+3.1) |
| **Họp (nhắm)** | 27.4 → **35.4 (+8.0)** ⭐ | 45.1 → 40.7 (**−4.4**) |
| IT (cũ) | 38.9 → 35.2 (**−3.7**) | 47.7 → 44.8 (−2.9) |
| Câu khó | 31.5 → **38.0 (+6.5)** | 58.7 → 52.8 (**−5.9**) |
| **TB toàn probe** | 28.9 → 31.7 (+2.8) | 51.2 → 48.7 (−2.5) |

**Glossary term-trần** (200 term held-out): 28% → **28%** (đứng yên; phép đo term-trong-câu vẫn nợ).
**Loss:** TB 500 step cuối **1.384** — vòng 1 ~1.39, chênh 0.006 < 0.02 ⇒ **tín hiệu T4**.

**Kết luận gate:** mẫu hình "học mới ĐÈ kiến thức cũ" hiện rõ — chỗ nhắm có ăn (họp vi→ja +8.0 nhờ data keigo/register; phủ định kép kéo câu khó vi→ja +6.5) nhưng trả giá bằng tụt loạt domain cũ (IT vi→ja −3.7, câu khó ja→vi −5.9, họp ja→vi −4.4) dù replay 70%; hội thoại vi→ja gần đứng yên (+0.4) dù 5,9k cặp ht+idh sạch (đã eyeball 10 cặp ngẫu nhiên chống dương tính giả — data KHÔNG bẩn).
→ **4/4 tín hiệu trần (T1+T2+T3+T4) — theo luật §5.1: CHỐT SCALE from-scratch.** Giữ 110M step23000 làm (a) teacher back-translation, (b) mốc đối chứng — model mới phải THẮNG nó trên cùng probe64. *(Chuẩn cuối vẫn là USER eyeball `eval/probe64_step23000.jsonl`.)*
→ **QUYẾT ĐỊNH SIZE (user, 2026-07-16): lên thẳng ~292M** (thay vì 200M của §5.1) để dư sức chứa cho vòng 3–5, chấp nhận i2_s ~150MB / ~150–200 tok/s CPU (vượt spec 100–200M của CLAUDE.md — đã cân nhắc; vẫn thừa cho re-translation). Kế hoạch chạy: §3.6.

### 3.6 SCALE ~292M — kế hoạch chạy (đang triển khai 2026-07-16)

- **Config:** d_model **1152** / **16 layer** / **18 head** (head_dim 64 — khớp graph build_bitnet_158) / FFN **3072**, vocab 32k, max_seq 256 → **291.8M params** (đã smoke test forward + converter tự đọc dims từ cfg ckpt).
- **Train:** from-scratch trên node hiện tại (16GB), dùng ngay `data/bin` premix vòng 2 (= base + vòng 1 + vòng 2, đủ toàn bộ data). Script **`cloud/run_300m.sh`**: 25.000 step, token budget 131k/step (MT 8192 × GA 16), LR 2.5e-4 → 2.5e-5 (warmup 1000), milestone mỗi 2000. Có **guard**: tự upload `last_final_23000.pt` (110M) lên release nếu thiếu rồi mới dọn `checkpoints/` → `checkpoints_110m/`.
- **Ước lượng:** compute/token ≈ 3× của 110M. **Số thật (node 4070 Ti 12GB, compile + `M300_MT=4096 M300_GA=32`): 12,25k tok/s / 4,78s/step → ETA ~33h.** VRAM 12GB: eager OOM cả MT=4096 (STE tạo bản sao fp32) — bắt buộc compile + MT≤4096, hoặc `M300_GC=1`.
- **Đóng gói:** tái dùng `watch_vong1.sh` với `WATCH_TAG=scale300m-step25000` (run_300m.sh ghi cùng marker).
- **Gate nhận model:** 292M phải **thắng 110M step23000** trên cùng probe64 + chrF FLORES (mốc: vi→ja 21.49 / ja→vi 42.27; probe TB vi2ja 31.7 / ja2vi 48.7). Thua = data là nút thắt, không phải size (khi đó dồn sức nở data BT).
- **KHÔNG làm "thinking"** (CoT): 300M quá nhỏ để reasoning có ích, phá latency (×5 token), data đắt — sức mạnh model dịch nhỏ đến từ data. **Dịch có ngữ cảnh** (zero-pronoun §9#1): làm ở **vòng 3 trên model 292M** — chỉ là quy ước format data (token sẵn có, sinh bằng code từ cặp câu liền kề OpenSubtitles/TED), không cần đổi tokenizer/arch nên không hoãn from-scratch.
- Vòng 3 (data số liệu/phủ định/câu phức §4 + ngữ cảnh + BT nở thêm) chạy TRÊN model 292M sau khi qua gate.

## 4. VÒNG 3 — Câu dài, số liệu, phủ định, tổng kết

### 4.1 Data nhắm lỗi tồn đọng (từ eyeball vòng 1-2 của user)
- **Số liệu/ngày giờ**: sinh 10-15k cặp template biến số thật (%, tiền VND/JPY, ngày-tháng-năm, giờ, thứ, phiên bản v1.2.3, deadline) — bắt buộc số ở nguồn xuất hiện đúng ở đích. Sinh bằng CODE (~0 token LLM), chỉ cần vài trăm khung câu. **Bắt buộc phủ quy đổi 万/億** (`10万円`=100.000 yên, KHÔNG phải 10.000!), loại từ (本/枚/台 ↔ con/cái/chiếc), niên hiệu (令和6年=2024).
- **Phủ định**: cặp câu khẳng định/phủ định tối thiểu ("tìm ra" vs "vẫn không tìm ra") — trị lỗi mất 「〜ていません」. Quét thêm **danh sách ngữ pháp JLPT N2-N1** (đã có từ Bước 1) cho các mẫu phủ định lắt léo: 〜わけではない (không hẳn là), しか〜ない (chỉ), 〜ないことはない (không phải không).
- **Câu phức nhiều mệnh đề**: điều kiện nếu-thì-còn-nếu, nhượng bộ mặc-dù, nối tuy-nhưng; cấu trúc "sự kiện + địa điểm" (kỷ niệm X TẠI khách sạn ≠ thành lập khách sạn). Thêm augmentation bằng code: **nối cặp câu ngắn sẵn có thành câu ghép** (miễn phí).
- Câu dài 30-60 token (max_seq 256 vẫn dư).

### 4.2 Train + tổng kết
- +**4.000 step** → đo full: FLORES (n=1012), 3 test set domain, probe 64 câu, so Google.
- Đóng gói bản release mới: convert → `llama-quantize I2_S 1` → i2_s 68MB → tag GitHub Release v1.1.

### 4.3 Kỹ thuật "thầy–trò" (chốt hướng 2026-07-16 tối, từ ý tưởng "thinking" của user; đo bằng hardbench200 + probe64)

Bối cảnh: user đề xuất "mode thinking chạy trong khối kiến thức nhỏ" + "người thầy sửa bài". CoT token thật thì KHÔNG làm (đã chốt §3.6: phá latency ×5, 300M quá nhỏ). Ba bản hiện thực hóa được, xếp theo thứ tự pilot:

1. **Glossary/idiom-injection (terminology-aware NMT)** — "khối kiến thức nhỏ" tra bằng CODE lúc chạy, ~0 chi phí: quét câu nguồn theo glossary + từ điển thành ngữ/slang + bảng quy đổi 万/億, nhét gợi ý `[gấu=恋人]` vào đầu prompt; train model vòng 3 biết dùng gợi ý (thêm mẫu có-gợi-ý vào data). Ăn thẳng vào 3 lỗi hardbench: glossary kẹt 28%, thành ngữ/slang mặt chữ, sai hàng số.
2. **Thầy–trò lúc TRAIN (error-targeted distillation)** — trò dịch → thầy (LLM API, hạ tầng `gen_any.sh` sẵn) chấm + viết bản sửa → train tiếp trên bản sửa. Data bám đúng phân bố lỗi thật. Chi phí inference 0. Đây là bản trung thành nhất với metaphor "thầy dạy trò tới khi thành thạo".
3. **Tự-sửa 2 lượt lúc INFER (APE)** — ưu tiên thử **CHÍNH MODEL đó tự làm thầy** trước (single-model self-edit): thêm task tag mới (vd `>>fix<<`) vào training, input = câu gốc + bản nháp, target = ref; lúc chạy: lượt 1 dịch, lượt 2 cùng model chạy mode fix. KHÔNG cần train/deploy model riêng, vẫn 1 file GGUF, latency ~2× (chấp nhận được ở 400+ tok/s). Data train task fix gần như miễn phí: (src, draft = output trò trên data sẵn, ref). Giới hạn phải biết: tự-sửa chỉ vá được fluency/sót vế/ngôi — KHÔNG vá được lỗ hổng tri thức (không biết "gấu"=恋人 thì mode fix cũng không biết). Thầy TÁCH RIÊNG (1-2 model phụ, thầy VI cho ja→vi + thầy JA cho vi→ja) chỉ đáng làm nếu self-edit có tín hiệu nhưng kịch trần — vì thầy riêng chỉ hơn khi có data/size khác trò.

Kỳ vọng đã thống nhất: KHÔNG chạm Claude trên đề khó mở (tri thức thế giới ~1000× scale, không pipeline 300M nào bù); đích thực = **vượt Google trong niche công sở IT** (đã thắng hội thoại ja→vi probe64) + bám sát domain chung. Trình tự: gate 292M trước → data vòng 3 + (1) → pilot (2) → chỉ khi chưa đủ mới (3).

## 5. Quy trình chuẩn mỗi vòng (lặp lại y hệt)

```
sinh/gom data → lọc (dedup + LaBSE) → binarize trộn mix →
resume train trên cloud (~4-6h/vòng) →
eval tự động: chrF FLORES + domain test + probe 64 câu (workflow có sẵn) →
USER eyeball 50 câu (30-60 phút) → chốt gate → vòng kế
```

- Eval chạy i2_s trên bitnet.cpp (số thật của sản phẩm cuối, không phải PyTorch).
- Probe 64 câu tái dùng làm regression test — cùng bộ câu, so ok-rate qua từng vòng.

### 5.1 GATE SỨC CHỨA — cơ sở quyết định scale 200M (điền sau MỖI vòng, ~10 phút)

> Nguyên tắc: scale chỉ có lãi khi nút thắt là SỨC CHỨA của 110M params, không phải data.
> Vòng 1 đã chứng minh pipeline data hoạt động (+8..+15 chrF đúng domain nhắm) → từ giờ,
> nếu data nhắm đích đạt chuẩn mà KHÔNG ăn điểm, nghi phạm chính là trần model.

Sau khi eval xong vòng N (probe64 + chrF FLORES + glossary test), điền 1 dòng vào
`eval/capacity_log.md` rồi đếm **tín hiệu trần**:

- **T1 — Data nhắm không ăn:** domain nhắm có ≥800 cặp sạch (qua đủ 4 lớp QUALITY_GATE + LaBSE, có oversample) mà Δprobe domain đó **< +3 chrF**. (Đối chứng vòng 1: IT +14.6, câu khó +10.5 với điều kiện tương đương.)
- **T2 — Học mới đè kiến thức cũ:** bất kỳ domain cũ nào tụt **> 2 chrF** dù replay ≥ 65% — model hết chỗ trống, phải ghi đè cái cũ để chứa cái mới.
- **T3 — Mốc tuyệt đối (§7):** sau Vòng 2 mà vi→ja FLORES vẫn **< 27**.
- **T4 — Loss chạm sàn:** trung bình loss 500 step cuối (lấy `grep -E "^step " checkpoints/train.log | tail -100`, tính tay hoặc awk) của 2 vòng liên tiếp chênh **< 0.02** — model không nén thêm được gì dù data mới đã vào.

**Quyết định:**
- **0–1 tín hiệu** → chưa phải trần. Tiếp vòng data kế (rẻ hơn scale nhiều lần).
- **≥2 tín hiệu trong cùng 1 vòng, HOẶC T1 lặp lại 2 vòng liên tiếp** → chốt scale ~200M: d_model 1024 / 16 layer / FFN 2816 (khởi điểm gợi ý), train **from-scratch** 2–3 ngày trên toàn bộ data mix hiện có (KHÔNG resume được từ ckpt 110M — khác kích thước ma trận). Giữ ckpt 110M làm (a) teacher back-translation, (b) mốc đối chứng: model 200M phải THẮNG 110M trên cùng probe64 mới được nhận.
- **Chống dương tính giả:** 1 vòng data kém chất lượng cũng bật T1. Trước khi kết tội model, eyeball 10 cặp ngẫu nhiên của mode không ăn điểm — nếu data bẩn/lệch domain thì lỗi ở data, sửa data trước, chưa tính là tín hiệu trần.

## 6. Việc cần USER làm
1. **Cấp glossary JA-VI** (csv/tsv/xlsx đều được) — cần cho Vòng 1.
2. **Cấp tài liệu IT đã dịch** (song ngữ hoặc 2 file riêng) — cần cho Vòng 1.
3. **Chốt ngân sách LLM token** cho sinh data (ước tổng ~3-4M token output cho cả 3 vòng).
4. **Eyeball 50 câu/vòng** (tôi chuẩn bị sẵn file chấm, chỉ cần đánh dấu đạt/không + ghi chú ngắn).
5. Thuê node GPU khi train (A4000/4070Ti, ~5h/vòng; hoặc train local 3060 Ti chậm hơn ~3-4x).

## 7. Rủi ro & đối sách
- **Quên kiến thức cũ** khi train tiếp → luôn mix ≥ 65% replay; gate "chrF không giảm".
- **Synthetic bẩn** → lọc LaBSE ≥ 0.8 với data sinh + dedup trước khi trộn; giữ log nguồn từng cặp.
- **Plateau do trần 110M params**: nếu vòng 2 xong mà vi→ja < 27 → cân nhắc **scale ~200M** (vẫn trong đích CLAUDE.md; A4000 16GB train được, from-scratch ~2-3 ngày với data mix cuối) — quyết sau, không làm sớm.
- **Lệch build bitnet.cpp**: mọi bản i2_s phải quantize bằng `I2_S 1` single-thread + chạy đúng build đã patch (STATUS.md).

## 8. Timeline dự kiến
| Vòng | Việc | Thời gian |
|---|---|---|
| 1 | glossary+docs+BT data (1-1.5 ngày) → train 5k step (~5h) → eval+eyeball | **2-3 ngày** |
| 2 | sinh data hội thoại/họp (~1 ngày) → train 4k (~4h) → eval+eyeball | **2 ngày** |
| 3 | data số liệu/câu phức (~0.5 ngày) → train 4k → eval full + release v1.1 | **1.5-2 ngày** |

**Tổng: ~1 tuần làm việc** (phần lớn thời gian là train chạy nền + chờ user eyeball).

Thời gian train thuần theo GPU (effective batch 131k token/step): A4000 ~3.5-3.8s/step, **4070 Ti ~2.2-2.5s/step (~1.5x nhanh hơn, VRAM 12GB đủ — run 14k chỉ chiếm 7.5GB)**. Mỗi vòng gọn trong 1 buổi thuê node ~4-5h (setup 15' + train + convert/tải 68MB + tắt). Kiểm nhiệt GPU 10 phút đầu (node A4000 cũ từng throttle 94°C mất nửa tốc độ).

## 9. Bản đồ lỗi đặc thù JA↔VI (các hệ khác & Google cũng dính) → công thức data

| # | Lỗi | Google có dính? | Data khắc phục | Vòng |
|---|---|---|---|---|
| 1 | **Chủ ngữ ẩn tiếng Nhật** (zero pronoun): 会議に出られません — ai? | CÓ — đoán bừa ngôi ("tôi"/"anh ấy") | Cặp kèm **câu ngữ cảnh đứng trước** + câu JA chủ ngữ ẩn ↔ VI đại từ tường minh đủ các ngôi | Nâng cấp sau 3 vòng (đổi format input) |
| 2 | **Xưng hô tiếng Việt** (anh/em/chị/mình/bạn) | CÓ — ra "tôi/bạn" robot | Hội thoại **persona quan hệ rõ** (sếp↔nhân viên, bạn thân, khách) | Vòng 2 |
| 3 | **Keigo bị ủi phẳng** register | CÓ | Mỗi nội dung 2-3 biến thể register | Vòng 2 |
| 4 | **Phủ định cuối câu & lắt léo** (〜わけではない, しか〜ない) | Thỉnh thoảng | Cặp tối thiểu ±phủ định + quét ngữ pháp JLPT N2-N1 | Vòng 3 |
| 5 | **万/億, loại từ, niên hiệu** (10万円=100.000¥!) | Ít nhưng có | Template số liệu quy đổi tường minh (sinh bằng code) | Vòng 3 |
| 6 | **Từ đa nghĩa** (大丈夫=được/thôi khỏi; trà đá→石のお茶) | CÓ | Cặp tương phản nghĩa (~200 từ bẫy mỗi chiều × 5-8 câu/nghĩa) | Vòng 2 |
| 7 | **Câu công thức văn hoá** (お疲れ様→"bạn đã mệt rồi"!) | CÓ — dịch literal | Phrasebook chức năng ~1k câu | Vòng 2 |
| 8 | **Tên riêng/chuỗi lạ** (Librali, cysteinm) | KHÔNG — Google rất mạnh mảng này | **Copy-through augmentation**: chuỗi lạ/version/ticket giữ nguyên văn | Vòng 1 |
| 9 | **Câu dài SOV↔SVO, mệnh đề quan hệ lồng** | Đỡ hơn nhưng vẫn dính | Câu phức + nối câu ngắn thành câu ghép (code) | Vòng 3 |

**Chỗ model chuyên domain có thể VƯỢT Google:** #2 #7 (+#8 nhờ glossary riêng) trong ngữ cảnh công sở IT — Google không dám đoán quan hệ xưng hô và luôn dịch literal câu công thức, còn mình chọn được domain hẹp.

## 10. Ghi chú dùng streaming STT (re-translation) — cho giai đoạn tích hợp

Model dịch nguyên câu nhưng dùng được với STT streaming bằng chiến lược **re-translation** (đúng thiết kế CLAUDE.md §5, không cần train simultaneous):
- **Tốc độ dư sức**: câu 30 token dịch lại hết ~0.1-0.15s (344-465 tok/s + pp 2900 tok/s); llama.cpp cache prefix nên càng rẻ.
- **Luật vàng JA→VI**: tiếng Nhật SOV — động từ và **PHỦ ĐỊNH nằm CUỐI câu** (参加でき…ません lật nghĩa ở chữ cuối). KHÔNG hiển thị bản "chốt" khi câu Nhật chưa hết; chỉ commit khi STT ra **final** / gặp 、。/ ngắt hơi. Partial hiện mờ + "…".
- **Chống nháy**: local agreement — chỉ giữ phần đầu bản dịch trùng nhau giữa 2 lần re-translate liên tiếp.
- Câu STT dài → cắt theo ranh giới mệnh đề, giữ dưới max_seq 256.
- Nâng cấp tuỳ chọn sau 3 vòng: **prefix-augmentation** (cắt prefix align 2 bên làm ~5-10% data phụ) để partial mượt hơn.
