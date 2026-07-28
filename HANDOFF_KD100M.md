# HANDOFF — BitNet 152M ja→vi

## 0-OCTIES. ⭐⭐⭐ VÒNG 6 ĐANG CHẠY (2026-07-28 01:20) — ĐỌC MỤC NÀY TRƯỚC

### Phát hiện lật ngược cả hướng đi: THUẬT NGỮ CHỈ LÀ 3% LỖI

Phân loại 59 câu v5 hỏng mà **Google dịch được** (`eval/error_taxonomy_v5.json`):

| Nguyên nhân | Lỗi chính |
|---|---|
| Cấu trúc câu (câu chẻ, mệnh đề lồng, sai trợ từ) | **37%** |
| Chọn sai nghĩa từ THÔNG THƯỜNG | **22%** |
| Ẩn chủ ngữ → sai đại từ tiếng Việt | 10% |
| Quán ngữ 8% · phủ định 5% · katakana 5% · thêm-bớt nghĩa 5% | |
| **Thuật ngữ chuyên ngành** | **3%** |

⇒ Cả tuần đo độ phủ thuật ngữ + đào 393M câu chỉ nhắm vào 3% nguyên nhân.

### Mật độ data KHÔNG dự đoán chất lượng (`scripts/skill_density.py`)

| Kỹ năng | % corpus | v5 | Google |
|---|---|---|---|
| keigo | **4,0%** | 80% | 80% ✅ ngang |
| katakana | **23,2%** | 75% | 95% ❌ thừa data vẫn hỏng |
| ẩn chủ ngữ | 22,5% | 65% | 85% ❌ |
| mệnh đề lồng | 5,5% | 50% | 95% |
| slang | 0,63% | 45% | 90% |

⇒ Hai loại can thiệp: (A) đói data → bơm lên ~4% như keigo; (B) đủ data mà hỏng →
bơm thêm cùng loại VÔ ÍCH, phải đổi CÁCH CHỌN.

### Kho CC-100 gốc: `ja.txt.xz` = 69,3GB = **392,8 TRIỆU câu**, trước chỉ dùng 1,7%

Đã đào hết 2 lượt (mỗi lượt 7-22 phút):
- `mined_rare.txt` 216.179 câu (thuật ngữ 54 miền)
- `mined_skills.txt` 2.033.769 câu (6 mặt trận lỗi + katakana + từ đa nghĩa)
- Lọc rác 2,0% → `v6_todo.txt` **2.204.272 câu** đang KD dịch

**Tốc độ đào — đã tối ưu 4 bước, đừng lặp lại sai lầm:**
| Cách | Toàn kho 69GB |
|---|---|
| `lzma` Python + `pyahocorasick` | 2,8 giờ |
| `xz -dc` + `pyahocorasick` | 100 phút |
| `xz -dc` + `ahocorasick_rs` (Rust) | 35 phút |
| **`xz -dc` + Rust + 10-13 tiến trình** | **7-22 phút** |

GPU/CUDA KHÔNG giúp — đây là đi bộ trên máy trạng thái, không thiếu FLOP.

### ⚠️ BẪY REGEX TIẾNG NHẬT — mất trọn một lượt đào
Mẫu 2 ký tự để trần gần như LUÔN là chuỗi con của từ khác:
`それな`→「それなり」(21% số khớp) · `きも`→「読み書きも」(21%) · `マジ`→「マジック」 ·
`沼`→「沼津」 · `おつ`→「おつまみ」 · `まい`→「〜てしまい」(28%).
**65% câu "slang" và 28% câu "phủ định" là rác.** Phải neo bằng đuôi biến cách/dấu câu.
Bản đã sửa: `scripts/mine_skills2.py` (đã kiểm mẫu, sạch).

### Model CHƯA train xong
Loss v5 vẫn giảm **−0,006/1000 step** khi dừng ở step 14.000. Dừng vì hết tiền, không
phải hội tụ ⇒ train thêm step là đòn bẩy rẻ nhất chưa dùng.

### Chuỗi tự động đang chạy
`scripts/overnight_v6.sh` (nền): chờ KD (~6h, 96 câu/s) → `merge_v6.py` → split →
binarize → upload Modal → `train_v6` 16.400 step (~$16,4, hết sạch ngân sách) →
`auto_eval_v6.sh` tự chấm + **bench TRONG MIỀN**.
Backup checkpoint mỗi 5 phút về `D:/Bit-Translate-data/checkpoints_v6/`.
Kết quả gom ở **`logs/KETQUA_V6.txt`**. Kế hoạch vòng sau: **`PLAN_V7.md`**.

---

# HANDOFF — Pilot KD 100M ja→vi (session 2026-07-25)

> File bàn giao để tiếp tục ở phiên CLI khác. Đọc mục 8 (Bước tiếp theo) trước, rồi
> mục 6 (gotchas) để không vấp lại lỗi đã gặp. Ngôn ngữ dự án: tiếng Việt.

---

## 0-SEPTIES. ⭐⭐ VÒNG 5 + PHÁT HIỆN LẬT NGƯỢC: BENCH ĐO SAI MIỀN (2026-07-27)

### Kết quả bench TED 200 câu (chấm mù, v4/v5/Google cùng phiên)

| | Google | v4 | v5 |
|---|---|---|---|
| Tổng (n=200) | 87% | 64% | 66% |
| ngắn (n=100) | 85% | 75% | 77% |
| dài (n=100) | 89% | 53% | 54% |

McNemar v4-v5: p=0,755 (tổng), p=1,000 (dài) → **KHÔNG khác biệt**.
Nhìn riêng bảng này thì vòng 5 thất bại hoàn toàn dù can thiệp rất mạnh:
chuỗi ≥110 token 3,39%→10,80%, token trong chuỗi dài 12,9%→33,2%, +2,45M cặp thật.

### ⚠️ NHƯNG BẢNG TRÊN ĐO SAI MIỀN — đây là bài học lớn nhất phiên này

Data thêm vào là **biên bản Quốc hội**; bench là **TED talk**. Đo bằng phép thử
"model có fit nổi chính data nó được train không" (100 câu Quốc hội dài, TB 159 ký
tự, so với đích Gemini):

| | câu dài Quốc hội 159 ký tự |
|---|---|
| **v4** | **6/22 = 27%** |
| **v5** | **16/22 = 73%** |

McNemar: v5 hơn 12 câu, v4 hơn 2 câu → **p=0,016, KHÁC BIỆT THẬT**.
Ví dụ: `西村国務大臣` v4 "Bộ trưởng Ngoại giao **Tây Thôn**" → v5 "Bộ trưởng Nishimura";
`越前市長` v4 "Thị trưởng **Việt Nam**" → v5 "Thị trưởng Echizen"; `CLT` v4 "**hóa chất**
trực giao" → v5 "gỗ ghép thanh vuông góc"; `平成17年` v4 "**2017**" → v5 "2005".

⇒ **v5 học được RẤT nhiều (27%→73%), bench TED không nhìn thấy.**

### ✅ PHÉP THỬ SẠCH — câu Quốc hội CHƯA MODEL NÀO TỪNG THẤY (chốt hạ)

Phản biện với bảng trên: 100 câu đó v5 đã gặp khi train (2,75 epoch) còn v4 thì chưa
→ một phần 73% có thể là ghi nhớ. Nên lấy 100 câu từ **572.282 câu Quốc hội nằm NGOÀI
`v5_planC.txt` và ngoài mọi `kd_v5*.jsonl` / `kd_cc100.jsonl`** (130-220 ký tự), Gemini
dịch làm đích. Cả hai model đều mù → đo đúng khả năng khái quát hoá TRONG MIỀN.

- Nguồn: `D:/Bit-Translate-data/raw/heldout_kokkai.txt`, đích `heldout_probe.jsonl`
- Bản dịch: `eval/bench_held_v5.jsonl`, `eval/bench_held_v4.jsonl`

| | 22 câu chưa từng thấy | KTC95 |
|---|---|---|
| **v5** | **21/22 = 95%** | [78%, 99%] |
| **v4** | **3/22 = 14%** | [5%, 33%] |

McNemar: **18 câu v5 đúng-v4 sai, 0 câu ngược lại → p = 7,6e-06.**

Lỗi điển hình của v4 mà v5 sửa hết:
`林大臣` → v4 "Bộ trưởng **Bộ Lâm nghiệp**" / v5 "Bộ trưởng Hayashi" ·
`野山` → v4 "**Noyama**" (tưởng địa danh) / v5 "vùng núi non" ·
`船舶リサイクル条約` → v4 "Hiệp ước Tái chế **Tư cách và Tư pháp**" / v5 "Công ước Tái chế Tàu biển" ·
`平成24年度から28年度` → v4 "**từ năm 24 đến 28 tuổi**" / v5 "năm 2012 và 2016" ·
`民法717条1項` → v4 "**Khoản 7 điều 71**" / v5 "Khoản 1 Điều 717" ·
`岡部参考人` → v4 "**các quan chức ở Okabe**" / v5 "nhân chứng Okabe" ·
`道州制` → v4 "**hệ thống pháp quyền**" / v5 "hệ thống đạo châu" ·
`キャリアパス` → v4 "**hỗ trợ công tác sau bán hàng**" / v5 (bỏ, không bịa).
v4 còn lặp cụm vô hạn ("hai mươi năm, hai mươi năm, hai mươi năm") — v5 hết hẳn.

⚠️ **Lưu ý hiệu chuẩn:** 95% này chấm trên **toàn văn**, còn bench TED 66% chấm theo
tiêu chí phiên khác. KHÔNG so 95% với 66%. Cái so được là **v5 vs v4 cùng phiên, cùng
thước** → 95% vs 14%. Khoảng cách đó là thật.

⇒ **KẾT LUẬN CUỐI: không phải ghi nhớ, không phải trần dung lượng. Vòng 5 là bước
nhảy lớn nhất từ trước đến nay — nhưng chỉ TRONG MIỀN đã bơm data.** Hướng "bơm data
theo miền" ĐÚNG. Vấn đề còn lại là **độ phủ miền**, không phải năng lực model.

### ❌ HAI KẾT LUẬN CỦA TÔI BỊ BÁC TRONG PHIÊN NÀY — đừng lặp lại

**1. "Chạm trần dung lượng 152M"** — SAI. Suy diễn từ hai kết quả âm (grow 12L→18L
không giúp câu dài; thêm data dài không giúp). Nhưng kết quả âm KHÔNG phân biệt được
"không thể" với "chưa đúng cách". Bằng chứng bác: model đạt 73% trên câu 159 ký tự —
fit được data train nghĩa là THỪA dung lượng biểu diễn.

**2. "Lượng tử hoá 1,58-bit cắt mất biểu diễn"** — SAI. User phản biện đúng: BitNet
b1.58 ở 7B dịch tốt. Paper BitNet cho thấy ternary ngang FP16 cùng số tham số **từ ~3B
trở lên**; dưới ngưỡng mới có khoảng cách. Phát biểu đúng phải là "152M quá nhỏ, và ở
quy mô nhỏ ternary lấy thêm một phần" — không phải "ternary là thủ phạm".

Cũng cần nhớ: **lúc TRAIN đã là ternary rồi** (`BitLinear.forward` gọi `weight_quant`
+ `activation_quant`, master FP chỉ để tích gradient). GGUF không làm model kém đi.

### PHÉP THỬ ĐÚNG để kết luận "đã đạt trần" (dùng lại cho mọi vòng sau)
1. **Fit tập train** — không fit nổi data mình được dạy = thiếu dung lượng thật.
   Fit tốt mà bench kém = vấn đề KHÁI QUÁT HOÁ, thêm data vẫn còn cửa. RẺ, làm trước.
2. **Đường cong theo checkpoint** — bão hoà sớm = hết đà; còn leo = chưa hội tụ.
3. **Bench TRONG MIỀN vừa bơm data** — nếu không hơn ở đây thì mới là thất bại thật.
4. FP16 cùng kích thước — chỉ làm sau khi 1-3 xong (tốn credit).

### Số liệu corpus v5
13.144.706 cặp / 667M token / 2,75 epoch. v4 cũ 80,1% + KD thật mới 18,6% +
quán ngữ ×3 1,3%. Train 14.000 step, loss 2,17→2,028, `rc=0`, ~$13,5.
Model: `D:/Bit-Translate-data/checkpoints_v5/v5_avg.pt` (trung bình 7 mốc 12500-14000).

---

## 0-SEXIES. ⭐ KẾT QUẢ VÒNG 4 + NGUỒN DATA VÒNG 5 (2026-07-26, khuya)

### Model tốt nhất: `D:/Bit-Translate-data/checkpoints_v4/v4_avg5.pt`
Train dừng ở step 5620/6000 (hết credit; LR đã 8,76e-6 ≈ đích 8e-6 nên không mất gì).
`v4_avg5` = trung bình step3500..5500, hơn `v4_avg` (3 mốc) trên CẢ HAI thước đo.

| | v3 | v4 | Google |
|---|---|---|---|
| Bench câu thật, ĐÚNG NGHĨA (n=200, chấm mù cùng phiên) | 58% | **64%** | 86% |
| — ngắn (n=100) | 70% | **75%** | 85% |
| — dài (n=100) | 47% | **52%** | 88% |
| grammar_probe | 71/80 | **72/80** (avg5) | — |

v4 vs v3: p=0,155 — **chưa đủ bằng chứng thống kê** (McNemar: v4 hơn 25 câu, v3 hơn 15).
Nguyên lý bước A đúng ở cấp vi mô (thuật ngữ được bơm data đi từ dịch bậy sang đúng,
mỗi từ có 242-362 câu trong data mới) nhưng bench TED ít thuật ngữ chuyên ngành nên
tổng thể chỉ dịch chuyển +6.

⚠️ **Thang chấm lệch giữa các phiên**: chấm lại 50 câu cũ cho v3 66%→60%, Google 84%→84%.
LUÔN chấm hai hệ trong CÙNG một phiên. Ghi chú cũ "v3 ngang Google ở câu ngắn" là SAI
(n=50 quá nhỏ); n=100 cho Google 85% vs v4 75%, p<0,001.

### 🐞 BUG LỚN: prompt GGUF thiếu EOS — mọi số đo bitnet.cpp cũ đều SAI
Định dạng train: `[BOS] >>vie<< <ja> [EOS] <vi> [EOS]` — **EOS là DẤU NGĂN CÁCH**.
`translate_bench.py::run_gguf` gọi `llama-cli -p ">>vie<< {src}"` → thiếu EOS →
v4_avg5 ra `..........`. Đã chứng minh GGUF KHÔNG hỏng bằng cách nạp trọng số từ
file GGUF ngược vào PyTorch (dịch hoàn hảo).

| cách gọi | kết quả |
|---|---|
| `llama-cli` thiếu EOS | `..........` |
| `llama-cli -p "...</s>"` | rỗng (cli dừng ngay tại EOS) |
| **llama-server + truyền TOKEN ID** | **đúng** |

⇒ số "292M GGUF 41,8 chrF" ở §3 và mọi kết quả `--gguf` cũ đều đo dưới prompt sai.

### bitnet.cpp: 77,6MB, 326-354 tok/s (6 lõi)
```
python scripts/convert_to_gguf.py --ckpt .../v4_avg5.pt --out D:/.../dist/v4_avg5_f32.gguf
wsl: ~/BitNet-test/build/bin/llama-quantize v4_avg5_f32.gguf v4_avg5_i2s.gguf I2_S 1
wsl: setsid nohup ~/BitNet-test/build/bin/llama-server -m .../v4_avg5_i2s.gguf \
       -c 2048 -np 4 -t 6 --host 127.0.0.1 --port 8081 </dev/null &
```
Ổ E ĐẦY (40GB) → mọi file lớn ghi `D:/Bit-Translate-data/`.

### NGUỒN DATA MỚI CHO VÒNG 5 (đều CHƯA qua Gemini)

| nguồn | số câu | ghi chú |
|---|---|---|
| **CC-100** `raw/cc100_pick.txt` | **5.230.697** | lọc 458M dòng, giữ 1,14%, `--min-score 15` |
| **Quốc hội** `raw/kokkai_ja.txt` | ~3-4M (đang thu) | `scripts/harvest_kokkai.py`, API mở |

Quốc hội đo trên mẫu: **0/833 câu trùng** 17,4M hash corpus cũ; p50 = 62 ký tự,
31% câu ≥80 ký tự (corpus v4: 3,4% chuỗi ≥110 token). Đúng chỗ model gãy.
Phải lọc văn khuôn mẫu nghị trường (「賛成の諸君の起立を求めます」) — xem `PROC` trong script.
⚠️ ngày cuối tháng phải dùng `calendar.monthrange`; `2010-02-29` → HTTP 400, mất im lặng cả tháng.

### BỘ DÒ LỖI TỰ ĐỘNG (`mt_diagnose.py` + `diag_report.py`)
Dịch 10k câu Quốc hội bằng v4_avg5 rồi dò bằng luật — KHÔNG cần judge, chạy được
trên hàng triệu câu. Kết quả 3k câu đầu **khớp bench chấm tay một cách độc lập**:

| dải ký tự | có cờ lỗi | | bench chấm tay |
|---|---|---|---|
| 0-40 | 23,4% | ↔ | câu ngắn sai 25% |
| 150+ | 44,0% | ↔ | câu dài sai 48% |

Ba bộ dò làm toàn bộ việc, đều tăng theo độ dài: `lech_so` 4,5→15,0%,
`mat_cau_hoi` 3,5→16,7%, `mat_phu_dinh` 3,5→13,0%.
Hai bộ vô dụng: `sot_tieng_nhat` (1/3067), `cut_cau` (11) — model không mắc.
⚠️ `lech_so` bản đầu chỉ bắt chữ số Ả Rập → chỉ 1 cờ; thêm SỐ HÁN (`十分の一`,
`二割`) mới lên 312. Detector đặt trong `diag_report.py` để cải tiến KHÔNG phải dịch lại.
⚠️ `roi_noi_dung`/`bia_them` dùng ngưỡng phân vị p5/p97 → theo định nghĩa luôn gắn
cờ ~5%/~3%; chỉ đọc được PHÂN BỐ LỆCH giữa các dải, không phải trị tuyệt đối.

---

## 0-QUINQUIES. ⭐ VÒNG 4 ĐANG CHẠY + PHÁT HIỆN ĐỔI HƯỚNG (2026-07-26, tối)

**Đang train**: app `ap-ZmiWOzNwFQFvzPUxny2tl1`, `train_v4` trong
`cloud/modal_train_100m_kd.py` — 18L/152,1M từ `v3_avg`, 6000 step, lr 8e-5,
`bin_v4` (12,56M seq / 475M token). Backup 10 phút/lần về
`D:/Bit-Translate-data/checkpoints_v4`; `scripts/auto_eval_v4.sh` tự chấm khi xong.

### Corpus v4 khác v3 ở đâu (`logs/merge_v4.log`)

| | vòng 3 | vòng 4 |
|---|---|---|
| Data THẬT | 10,52M (89,6%) | **11,45M (91,0%)** — thêm 922k OPUS chưa dùng |
| Tổng hợp phong cách | 1,225M (**10,42%**) | 408k (**3,2%**) |
| Tổng hợp NHẮM ĐÍCH | 0 | **723k (5,7%)** = termgap ×6 + sciterm ×2 |

Lệnh tái lập:
```
python scripts/merge_niche_corpus.py --oversample 1 --os-map "termgap=6,sciterm=2" \
  --extra-real D:/Bit-Translate-data/raw/kd_os_new.jsonl \
  --out D:/Bit-Translate-data/kd_filtered_niche_v4.jsonl
python scripts/prep_clean_split.py D:/Bit-Translate-data/kd_filtered_niche_v4.jsonl \
  D:/Bit-Translate-data/clean_v4
python scripts/binarize_ja2vi.py --clean D:/Bit-Translate-data/clean_v4 \
  --out D:/Bit-Translate-data/bin_v4
```

### Data theo DANH SÁCH PHỦ (thay cho sinh theo chủ đề tự nghĩ)
`measure_term_coverage.py` → 8.041/11.933 thuật ngữ glossary xuất hiện ≤5 lần
(2.343 chưa từng). `gen_term_gap.py` sinh 100.388 câu phủ **7.449/8.041 (92,6%)**
trong 22 phút, bản dịch tiếng Việt LẤY TỪ GLOSSARY. Sau khi nhân bản: trung bình
**78 lần/thuật ngữ** trong corpus, chỉ còn 560 thuật ngữ ≤5 lần.

### ⚠️ HAI PHÁT HIỆN LÀM ĐỔI HƯỚNG BƯỚC B

**(1) Thuật ngữ chỉ là ~1/3 lỗi.** Đọc tay 17 câu v3 sai nghĩa trên bench câu thật:
cấu trúc ~8 câu, thuật ngữ ~5 câu, rơi nội dung câu dài ~3 câu. ⇒ trần của vòng 4
trên bench thật là khoảng **66% → 72-76%**, không hơn.

**(2) Model KHÔNG thiếu kiến thức ngữ pháp — nó mất dấu cấu trúc khi câu dài.**
`scripts/grammar_probe.py` (mới, 80 phép thử, chấm bằng regex nên chạy lại miễn phí)
đo v3: **câu ngắn 96% (52/54), câu dài 79% (11/14)** trên CÙNG những điểm ngữ pháp.

| điểm ngữ pháp | ngắn | dài |
|---|---|---|
| phạm vi từ…đến (から〜まで) | 5/5 | 0/1 |
| trích dẫn nghe nói (そうです) | 4/4 | 0/1 |
| hướng cho/nhận (もらう) | 4/4 | 0/1 |

Câu dài tái lập ĐÚNG lỗi trên bench thật: `乗ってもらいました` → "họ đã cho tôi đi
chiếc Porsche". ⇒ **Bơm data dạy ngữ pháp là vô ích.** Nút thắt là dung lượng/chú ý
trên câu dài. Bộ probe đã mở rộng thành thang 4 mức (ngắn/vừa/dài/rất dài) để đo
NGƯỠNG GÃY, chạy tự động trong `auto_eval_v4.sh`.

### ⭐ (3) TĂNG CHIỀU SÂU KHÔNG TRỊ ĐƯỢC CÂU DÀI — chốt hướng vòng 5
Chạy `grammar_probe.py` (80 phép thử) trên cả ba đời model:

| model | ngắn (n=54) | KHÔNG ngắn (n=26) | tổng |
|---|---|---|---|
| v1 12L/109,6M | 87% | 65% | 80% |
| v2 12L/109,6M | 93% | 73% | 86% |
| v3 **18L/152,1M** | **96%** | **73%** | 89% |

v2→v3 thêm 6 lớp (+39% tham số) + 8000 step: câu ngắn 93%→96%, câu dài ĐỨNG YÊN,
và **5/7 phép thử trượt trùng nhau**. ⇒ grow tiếp 18L→24L sẽ lặp lại đúng kết quả này.

Nguyên nhân đo được — phân bố độ dài chuỗi (ja+vi) trong bin:

| | bin_v3 | bin_v4 |
|---|---|---|
| < 40 token | 68,0% | 69,7% |
| ≥ 110 token (mức probe "dài" bắt đầu gãy) | 3,89% | **3,39%** |
| ≥ 170 token (mức "rất dài") | 0,64% | 0,60% |

Mode `long` tổng hợp vòng 3 chỉ thêm 20k cặp = 0,17% corpus. Vòng 4 còn làm tỷ lệ
tệ đi chút vì 922k câu OPUS phụ đề rất ngắn.

**VÒNG 5**: cân lại PHÂN BỐ ĐỘ DÀI, KHÔNG tăng model.
  1. Oversample 426k chuỗi ≥110 token sẵn có ×5 → ~13% corpus (miễn phí, làm ngay).
  2. CC-100 (`filter_cc100.py` đã cho +3,0 điểm cho câu 60-150 ký tự) → KD dịch.
File mốc: `eval/probe_v1.jsonl`, `probe_v2.jsonl`, `probe_v3.jsonl`.

### Bench câu thật: n=50 là KHÔNG ĐỦ
`eval/aggregate_bench_judge.py` (mới) tính Wilson + McNemar. Với n=50, ngay cả
khoảng cách Google 84% vs v3 66% cũng **p=0,066 — chưa đủ bằng chứng**. Vòng 4 phải
chấm mù **cả 200 câu**, v3 và v4 **trong CÙNG một phiên** (thang judge không hiệu
chuẩn giữa các phiên). Dựng panel:
```
python scripts/translate_bench.py D:/Bit-Translate-data/checkpoints_v4/v4_avg.pt --label v4
python eval/build_panels_bench.py eval/judge_v4 v3=eval/bench_v3.jsonl \
    v4=eval/bench_v4.jsonl google=eval/bench_google.jsonl
# judge chấm 0/1/2 -> eval/judge_v4/judges/panel_N.json
python eval/aggregate_bench_judge.py eval/judge_v4
```

**Mốc phải vượt** (n=50, `eval/bench_meaning.json`): v3 tổng 66% / ngắn 80% / dài 57%;
Google 84% / 90% / 80%; Haiku 94%.

### Script mới phiên này
`gen_term_gap.py`, `measure_term_coverage.py`, `init_round.py`, `grammar_probe.py`,
`auto_eval_v4.sh`, `eval/build_panels_bench.py`, `eval/aggregate_bench_judge.py`;
`merge_niche_corpus.py` thêm `--extra-real` + `--os-map`.

### Đang chạy nền
CC-100 lọc có chủ đích (`filter_cc100.py`): 1,5M/4M câu — nhiên liệu vòng 5, cần KD
dịch sau khi lọc xong.

### Nợ kỹ thuật đã biết
`prep_clean_split.py` chia dev theo `id`, mà mỗi bản nhân bản có id khác nhau → một
câu termgap có thể vừa ở dev vừa ở train. Dev vì vậy chỉ là đồng hồ báo overfit,
KHÔNG so được giữa các vòng. (Vòng 3 cũng vậy nên ít nhất là nhất quán.)

---

## 0. TL;DR
- **Đã dịch KD 11,88M câu ja→vi bằng Gemini Live API → lọc còn 10,5M → train 100M
  ja→vi from-scratch → đóng GGUF chạy CPU 214 tok/s (71MB).**
- **Pilot THẮNG ĐẬM 292M cũ** (judge 3,57 vs 2,40; dùng được 6%→55%) **nhưng còn
  kém Google (4,02) / Haiku (4,51)**. Yếu ở: thành ngữ, keigo, câu dài.
- **Đang làm dở:** sinh thêm data idiom/keigo (REST Gemini) để enrich rồi train
  lại 100M — test giả thuyết "nút thắt là DATA, không phải dung lượng".

---

## 1. Pipeline đã hoàn thành (theo thứ tự)

| Bước | Script | Output | Số liệu |
|---|---|---|---|
| KD dịch | `scripts/run_kd_batch.py` (+ `run_kd_5keys_parallel.py`) | `data/synthetic/kd_clean/kd_gemini3_final.jsonl` | 11.883.869 record (11,88M nguồn + 223 orphan) |
| Rule filter + NFKC | `scripts/filter_kd_full.py` | `data/synthetic/kd_clean/kd_filtered.jsonl` | 10.525.121 cặp (88,57%) |
| Tách train/dev | `scripts/prep_clean_split.py` | `data/clean/{train,dev}.{ja,vi}` | train 10.522.046 / dev 3.075 |
| Binarize ja→vi | `scripts/binarize_ja2vi.py` | `data/bin/{train,dev}.{tokens.u16,index.npy}` | 10.509.854 seq, **412M token**, mean 39,2 tok/seq |
| Train 100M | `cloud/modal_train_100m_kd.py` | Modal volume `vija-100m-kd-vol` | 15.000 step, loss cuối 0,757 |
| Convert GGUF | `scripts/convert_to_gguf.py` + `llama-quantize` | `dist/kd100m_i2s.gguf` (71MB) | i2_s 1.58-bit |
| Eval | `scripts/hardbench_ckpt.py`, `judge_kd_ja2vi/` | `eval/hardbench_kd100m_*` | xem mục 3 |

**Teacher KD:** `gemini-3.1-flash-live-preview` (Live API, response_modalities=AUDIO
+ output_transcription — lấy text qua phiên âm audio).
**Format binary ja→vi thuần:** `[BOS=2, >>vie<<=4, ja_ids, EOS=3, vi_ids, EOS=3]`.
KHÔNG có `>>jpn<<`(5), KHÔNG `>>fix<<`(32000), KHÔNG `>>thinking<<`. vocab=32001.

---

## 2. Cấu hình train 100M (đã chạy)
- Modal account: **`nguyenda190497`** (tài khoản free $30 — KHÁC account cũ
  `nguyentuanngai` còn $9,28). Volume: `vija-100m-kd-vol`.
- Dims: `--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048 --vocab-size 32001` = 109,6M params.
- Args: `--max-tokens 8192 --grad-accum 16` (=131k token/step) `--compile --max-seq 256
  --pad-multiple 32 --lr 3e-4 --min-lr 3e-5 --warmup 1000 --max-steps 15000` (from-scratch, không lr-anchor).
- ~5 epoch (412M×~4,8). Tốc độ L40S ~57k tok/s = **0,9 s/step**, ~4h, ~$8-14.
- Loss: 10,5 (step0) → 0,81 (step11k) → **0,757 (step15k, phẳng từ ~step11k)**.

---

## 3. Kết quả eval (100 câu ja2vi hardbench)

**chrF (sacrebleu):**
| Hệ | chrF |
|---|---|
| KD-100M PyTorch step11000 | **43,3** |
| KD-100M i2_s GGUF step15000 (CPU) | 41,8 |
| Haiku | 41,5 |
| Google | 36,1 |
| 292M cũ (step30000) | 34,7 |
| 100M v4 cũ | 34,1 |

**Judge mù (Claude-Opus chấm blind, acc 0-5, % dùng được = acc≥4):**
| Hệ | acc | % dùng được |
|---|---|---|
| Fable(Claude) | 5,00 | 100% |
| Haiku | 4,51 | 93% |
| Google | 4,02 | 75% |
| **KD-100M** | **3,57** | **55%** |

Đối đầu KD vs Google: 20 thắng/35 hòa/45 thua. vs Haiku: 11/29/60.
Cũ để so: 292M judge ja2vi ~2,40 ; 100M cũ ~1,90 / ~6% dùng được.

**⚠️ BÀI HỌC LỚN: chrF NỐNG ĐIỂM model nhỏ.** chrF nói KD thắng Google/Haiku;
judge mù cho thấy KD THUA cả hai. Luôn dùng judge để so với hệ khác, không tin chrF.

**Tốc độ CPU (bitnet.cpp i2_s, Ryzen 5 5600X 6 threads):** tg **214 tok/s**, pp 1469
tok/s, file 71MB, RAM 114MB. → Đạt mục tiêu CLAUDE.md "CPU siêu nhẹ".

---

## 4. Phân tích 45% câu fail (acc≤3)
Fail theo domain: thành ngữ **9/10**, keigo **7/10**, câu dài **7/10**, slang 6/10,
hop 4, it_deep/ngữ pháp 3, hội thoại/số liệu/zero-pronoun 2 (các domain lõi TỐT ≥80%).

4 nhóm nguyên nhân:
1. **Câu dài → sụp cấu trúc** (共働き→"nơi làm việc chung", chèn "người đi nước ngoài" vô nghĩa). Nghi *dung lượng*.
2. **Thành ngữ → dịch literal** (元も子もない→"không có đứa con", 腹を割って→"mổ bụng"). *Data hiếm*.
3. **Keigo → sai chủ ngữ ẩn / đảo hướng kính ngữ** (検討させていただく→"mong quý khách xem xét"). *Data hiếm (2,6%)*.
4. **Đảo nghĩa/polarity + từ hiếm hỏng** (câu nhờ→câu phủ định; TBD→"TMD"; リハーサル→"real").
Ngoài ra: **độ tự nhiên thấp** (nat 3,18 vs Google 3,45, Haiku 3,88) — nhiều câu đúng nhưng cứng.

---

## 5. Nút thắt: CHƯA chứng minh được là dung lượng
**Tôi (assistant) đã từng nói quá "100M kịch trần dung lượng" — KHÔNG có bằng chứng.**
- Loss phẳng chỉ chứng minh *run này hội tụ*, không phải trần dung lượng.
- Bằng chứng NGHIÊNG VỀ DATA: (a) train loss 0,75 thấp = model *đủ sức* fit data
  (không underfit); (b) fail tập trung ở domain HIẾM (idiom 0,x%, keigo 2,6%).
- **Chỉ 1 domain (câu dài) nghi dung lượng.**
- Bằng chứng quyết định cần (CHƯA có): train 200M cùng data → so train-loss + usable
  rate; HOẶC bơm data idiom/keigo vào 100M → nếu usable tăng thì nút thắt là data.

**200M trên data hiện tại bị ĐÓI DATA:** cần ~4 tỷ token (20 tok/param) = ~10 epoch
trên 412M → ~16h, ~$30 (ăn trọn budget) + lặp nhiều. Train ít epoch hơn → 200M
undertrained, có thể không hơn 100M. ⇒ **enrich data TRƯỚC khi scale 200M.**

---

## 6. GOTCHAS kỹ thuật (đọc kỹ để không vấp lại)
1. **WSL lệnh nhiều dòng bị mangle** qua Git Bash→wsl.exe (im lặng, không output).
   → Viết file `.sh` rồi chạy: `wsl -e bash -lc "sed -i 's/\r$//' /mnt/e/.../x.sh && bash /mnt/e/.../x.sh"`.
   Lệnh WSL 1 dòng (dùng `;`) thì OK.
2. **`llama-quantize <f32> <i2s> I2_S 1`** — số `1` (single-thread) BẮT BUỘC, bỏ đi
   model CÂM (matmul ternary ra 0). Xem `scripts/convert_to_gguf.py` docstring.
3. **chrF nống điểm model nhỏ** → luôn judge để so hệ khác.
4. **Key Gemini format `AQ.Ab8RN...`** dùng được CẢ Live API LẪN REST generate_content.
   (Key format `AIzaSy...` user paste hôm trước bị "leaked"/invalid — Google tự khóa.)
5. **Quota 1011 (RESOURCE_EXHAUSTED)** khi chạy ~200 luồng/đẩy mạnh → per-window
   quota, giảm luồng thì tự hồi. Giữ mức vừa (6 key × 20 = 120 luồng chạy ổn).
6. **BATCH_SIZE>1 (gom câu)** vô ích khi đã kịch TPM (token/phút): gom 2 câu = 1,7×
   token/phút → vỡ trần. Khi quota-bound dùng batch-1. (`run_kd_batch.py` có bug
   2-session/worker đã fix bằng lazy single-fallback.)
7. **Modal đổi account:** `modal token new --profile X` → `modal profile activate X`.
   Data phải UPLOAD LẠI trên account mới (volume không chia sẻ giữa account).
   Upload: `set PYTHONUTF8=1 && modal volume put vija-100m-kd-vol data/bin bin`.
8. **Windows console:** luôn `PYTHONUTF8=1` (cp1252 không in được emoji/CJK).
9. **Môi trường:** Windows python = torch CPU-only (không CUDA), có gguf/sentencepiece/
   sacrebleu. WSL (`/home/tuent/BitNet-test/build/bin`) = có BitNet build (llama-cli/
   quantize/bench/server) NHƯNG không torch/sacrebleu. WSL đọc được `/mnt/e/Bit-Translate`.
10. **llama-server health:** dùng `curl -sf` (trả 503 khi đang load model; `-s` không
    fail trên 503 → query sớm bị 503).
11. **cmd.exe (không phải bash):** đặt env bằng `set VAR=val` (không dấu cách quanh =),
    KHÔNG dùng `VAR=val cmd`.

---

## 7. File & checkpoint quan trọng
**Local:**
- Model: `checkpoints/kd_step15000.pt` (cuối), `kd_step11000.pt` (đã judge), `kd_last.pt`
  (resume state 1,3GB), `kd_train.log`.
- GGUF: `dist/kd100m_f32.gguf` (439MB), `dist/kd100m_i2s.gguf` (71MB — bản chạy CPU).
- Data: `data/synthetic/kd_clean/kd_filtered.jsonl` (10,5M), `data/clean/`, `data/bin/`.
- Eval: `eval/hardbench_kd100m_step11000.jsonl`, `eval/hardbench_kd100m_i2s_hyps.jsonl`,
  `eval/judge_kd_ja2vi/` (panels + judges + key).
**Modal (account nguyenda190497, volume vija-100m-kd-vol):** 15 milestone step1000-15000 + last.pt.

**Lệnh hữu ích:**
```bash
# Test dịch nhanh 1 checkpoint (CPU, PyTorch):
PYTHONUTF8=1 python scripts/translate_ckpt.py checkpoints/kd_step15000.pt
# Bench tốc độ CPU (WSL):
wsl -e bash -lc "/home/tuent/BitNet-test/build/bin/llama-bench -m /mnt/e/Bit-Translate/dist/kd100m_i2s.gguf -t 6 -p 128 -n 128"
# Hardbench chrF qua PyTorch checkpoint:
PYTHONUTF8=1 python scripts/hardbench_ckpt.py checkpoints/kd_step15000.pt
```

---

## 0-QUATER. ⭐ KẾT QUẢ VÒNG 3 — 83% DÙNG ĐƯỢC, VƯỢT HAIKU (2026-07-26)

Chi tiết: `eval/judge_v3/RESULT_v3.json`. Checkpoint: `D:\Bit-Translate-data\checkpoints_v3\v3_avg.pt`.

| | v2 | **v3 (152,1M)** | Google | Haiku | Fable |
|---|---|---|---|---|---|
| acc | 3,82 | **4,29** | 2,92 | 3,70 | 4,98 |
| nat | 3,76 | **4,07** | 2,86 | 3,56 | 4,99 |
| % dùng được | 65% | **83%** | 26% | 69% | 100% |
| chrF | 46,8 | **48,7** | 36,1 | 41,5 | — |

Đối đầu v3 vs v2: **39 thắng / 47 hòa / 14 thua**. Loss 2,0953 → 2,0358. Train 8.000
step × 1,27s = 2,8h, ~$7,7.

Domain acc (v2→v3): **thanhngu 2,6→4,3 (+1,7)** · hoithoai 3,3→4,2 · zeropronoun 3,8→4,7 ·
caudai 3,0→3,7 · hop 3,6→4,0 · solieu 4,4→4,8 · nguphap 4,2→4,5 · slang 4,1→4,1 ·
it_deep 4,3→4,1 · **keigo 4,9→4,5**.

### ⭐ BÀI HỌC QUAN TRỌNG NHẤT: ĐỘ PHỦ SEED > SỐ LƯỢNG CÂU
- Vòng 2: 30.778 cặp trên **1.241** quán ngữ ⇒ thanhngu chỉ **+0,8**.
- Vòng 3: 95.782 cặp trên **4.691** quán ngữ ⇒ **+1,7**.

Cùng loại data, khác độ phủ, khác hẳn kết quả. Lỗi cũ đã sạch: `朝飯前`→"dễ như ăn cháo",
`猫の手も借りたい`→"bận đến mức chẳng kịp thở", `二の足を踏む`→"dậm chân tại chỗ",
`頭が上がらない`→"không thể ngẩng cao đầu". ⇒ **Khi một domain KHÔNG lên sau khi bơm data,
kiểm tra ĐỘ PHỦ SEED trước, đừng vội kết luận là giới hạn dung lượng.**

### Về dung lượng — CHƯA kết luận được
`scripts/check_new_layers.py`: 6 block mới đạt **20,1%** độ lớn block cũ (trước train
0,0%), tăng dần theo tầng (17,2% → 26,2%). Tức model CÓ dùng chỗ mới nhưng chưa đầy.
`caudai` 3,0→3,7, tiệm cận Haiku 3,8 nhưng chưa vượt. ⇒ Còn dư địa: train thêm step
trên 18L, hoặc grow tiếp 18L→24L (195M, ~$8).

### Hạn chế khi trích số này
1. Grow + data mới cùng lúc ⇒ KHÔNG tách được đóng góp của từng cái.
2. Judge do assistant chấm, KHÔNG mù hoàn toàn (3 hệ kia giữ nguyên bản dịch nên nhận
   ra được cột KD). Vẫn so được vì chấm CÙNG PHIÊN, cùng tay chấm với v2.

### Đang chạy nền (miễn phí)
KD dịch **934.524 câu Nhật mới** (lọc từ OPUS OpenSubtitles, đã loại 986k trùng corpus
cũ + 1,09M rác phụ đề) → `D:\Bit-Translate-data\raw\kd_os_new.jsonl`. Dùng cho vòng 4.

---

## 0-TER. VÒNG 3 — MỞ RỘNG ĐỘ SÂU 12L→18L (2026-07-26)

**Quyết định chốt:** KHÔNG train 200M từ đầu ($21, rủi ro cao) mà **mở rộng độ sâu
12L→18L = 152,1M** (`scripts/grow_depth.py`), ~$7.

Vì sao: giữ `d_model=768` nên **kế thừa 100% weights** của `v2_avg3`. Chèn 6 block mới
với zero-init `attn.wo`+`ffn.down` ⇒ block mới là IDENTITY. Đã VERIFY hai lớp:
sai lệch logits = **0.000e+00** và hardbench **100/100 câu dịch y hệt**, chrF 46,8.
Khả thi với BitNet vì `weight_quant` có `clamp(min=1e-5)` nên weight 0 ra 0, không NaN.
Block mới rải đều (sau mỗi 2 block cũ) chứ không dồn cuối.

| | 200M from-scratch | **grow 18L** |
|---|---|---|
| Params | 251,0M (d1024/16L — KHÔNG phải 200M) | **152,1M** |
| Điểm khởi đầu | chrF 0 | **chrF 46,8 / 65% dùng được** |
| Token đã học tổng | ~770M | **~1,38 tỷ** (kế thừa 970M) |
| Thời gian / chi phí | 8,8h / ~$21 | **2,8h / ~$7** |
| Chuyển account | 2 lần thủ công | không |

**Data vòng 3** (`bash scripts/prep_v3.sh`, oversample **3** để giữ niche ~10% như
vòng 2): harvest thêm **+3.160 quán ngữ** (tổng **4.666**, gấp 3,8× seed 1.223) bằng
`scripts/harvest_idioms_live.py` (bản cũ `gen_idiom_harvest.py` ĐÃ HỎNG: cần
OPENAI_API_KEY không có + model 404). Sinh thêm: idiom 95k · **conv** 40k · **zeropron**
30k (2 mode mới trong `gen_niche_kd.py`).

**Ba chỉ số quyết định sau vòng 3:**
1. **Layer mới có học không** — đo norm `attn.wo`/`ffn.down` của 6 block mới; nếu vẫn
   ~0 thì dung lượng KHÔNG phải nút thắt ⇒ dừng scale, dồn hết vào data.
2. `caudai` có nhích khỏi **3,0** không (Google 3,4 · Haiku 3,8).
3. `thanhngu` có vượt **2,6** không (kiểm chứng giả thuyết "thiếu độ phủ quán ngữ").

**QUYẾT ĐỊNH CỦA USER — KHÔNG ĐỀ XUẤT LẠI:**
- **BỎ HẲN chiều ngược vi→ja.** Dù nó cho +451M token miễn phí, model chỉ phục vụ
  ja→vi; thêm chiều ngược chia dung lượng (vốn đã căng) cho việc không dùng đến.
- Nguồn câu Nhật mới: OPUS OpenSubtitles ja (3,14M câu) **trùng 48,6%** với corpus cũ
  ⇒ chỉ +1,6M câu mới (+13% token). Tải sẵn ở `D:\Bit-Translate-data\raw\os_ja.txt.gz`.
  CC-100 ja mới là nguồn lớn (statmt.org, truy cập được) nhưng tốn công tải + lọc.
- 422k cặp data cũ nằm rải trong `data/synthetic/` — **KHÔNG gộp**: chỉ +3% và có ~20k
  cặp từ `qwen-max/turbo/flash` mà PROVIDERS.md đã CẤM (11-16,7% lỗi).

**Backup khi hết credit:** `bash scripts/backup_ckpt_loop.sh checkpoints_v3 600` chạy
song song — tự tải milestone mới về ổ D + kiểm tra corrupt. Khi chuyển account chỉ cần
`stepN.pt` (~600MB) chứ KHÔNG cần `last.pt` (train.py chịu được thiếu optimizer state).

---

## 0-BIS. KẾT QUẢ VÒNG 2 (enrich niche) — ĐỌC TRƯỚC TIÊN

**46% → 65% dùng được** (judge mù 100 câu, cùng phiên, cùng tay chấm).
Chi tiết: `eval/judge_v2/RESULT_v2.json`; mốc: `eval/judge_avg5/BASELINE_avg5.json`.

| | avg5 (trước) | v2_avg3 (sau) | Google | Haiku | Fable |
|---|---|---|---|---|---|
| acc | 2,99 | **3,82** | 2,92 | 3,70 | 4,98 |
| nat | 2,99 | 3,76 | 2,86 | 3,56 | 4,99 |
| % dùng được | 46% | **65%** | 26% | 69% | 100% |

Đối đầu từng câu: v2 thắng 57 / hòa 27 / thua 16. chrF 44,4 → 46,8.
⇒ **KD-v2 vượt Haiku về acc, gần bằng về % dùng được, vượt Google rõ.**

Domain acc (avg5→v2): keigo 2,7→**4,9** · slang 2,4→**4,1** · caudai 2,0→3,0 ·
hop 2,7→3,6 · it_deep 3,4→4,3 · thanhngu 1,8→2,6 · zeropronoun 3,3→3,8 ·
nguphap 3,8→4,2 · hoithoai 3,0→3,3 · solieu 4,8→4,4.

**Trả lời câu hỏi nút thắt (§5 cũ):**
- **DATA là nút thắt ở keigo + slang** — 2 domain bơm data cải thiện cách biệt (+2,2, +1,7)
  so với domain không bơm cao nhất (+0,9). Enrich tiếp là rẻ và hiệu quả.
- **THÀNH NGỮ cần phủ RỘNG hơn chứ không phải nhiều câu hơn** — chỉ +0,8, vẫn thấp nhất
  (2,6) dù bơm 30.778 cặp; vì 1.241 quán ngữ seed KHÔNG trùng quán ngữ trong hardbench.
  ⇒ vòng sau phải harvest thêm hàng nghìn quán ngữ mới (mở rộng `idiom_glosses.jsonl`),
  không phải sinh thêm câu cho cùng 1.241 cái cũ.
- **CÂU DÀI nghiêng DUNG LƯỢNG** — +1,0 lên 3,0, VẪN kém Google 3,4 / Haiku 3,8 dù có
  20.422 cặp riêng; lỗi còn nguyên là sụp chủ thể. Cộng `solieu` tụt −0,4 (phải đánh đổi
  domain mạnh) ⇒ 100M đã căng. **Đây là chỗ đáng cân nhắc 200M**, giờ đã có data để nuôi.

**⚠️ HAI HẠN CHẾ khi trích số này:** (1) resume chứ không from-scratch ⇒ KHÔNG tách được
phần do data vs phần do train thêm 4000 step; (2) lần chấm v2 KHÔNG mù hoàn toàn — 3 hệ
kia giữ nguyên bản dịch nên nhận ra được cột KD.

**Checkpoint:** `D:\Bit-Translate-data\checkpoints_v2\v2_avg3.pt` (bản tốt nhất) ·
`v2_step4000.pt` (46,3 chrF, kém avg3 46,8) · Modal volume `checkpoints_v2/`.

---

## 8-BIS. CẬP NHẬT PHIÊN 2 (2026-07-25, chiều) — ĐỌC MỤC NÀY THAY §8 CŨ

### a) Data niche ĐÃ SINH XONG (Live API, không phải REST)
`scripts/gen_niche_kd.py` — sinh cặp ja→vi cho domain hiếm, **6 mode**:
`idiom | keigo | slang | negation | katakana | long`.

| Mode | Cặp (qua QC) | Thời gian | Kiểm chứng chất lượng |
|---|---|---|---|
| keigo | **80.665** | 16,7′ | 57 pattern × 18 tình huống, nhóm CONTRAST (させていただく vs していただく) trị lỗi đảo hướng — mẫu dịch đúng chiều |
| negation | **42.594** | 5,4′+2,6′ | 284 pattern; 二重否定 đúng; `request` 9,7% (lượt đầu chỉ 3,2% → thêm `NEG_REQUEST`×3) |
| slang | **40.794** | 5,7′ | 4.134 từ slang khác nhau; emoji cân bằng ja↔vi (chỉ 40 cặp lệch, đã loại) |
| idiom | **30.778** | 3,9′ | 1.241 quán ngữ × ~25 câu × 16 ngữ cảnh; seed `data/synthetic/gen/idiom_glosses.jsonl` |
| katakana | **25.805** | 4,4′ | 890 từ; wasei-eigo đúng nghĩa Nhật (マンション→"căn hộ chung cư", ペーパードライバー→"có bằng nhưng không lái") |
| long | **20.422** | 13,0′ | ja tb 93 ký tự, 91,7% ≥80; dịch giữ đủ mệnh đề |
| **TỔNG** | **241.058** | ~55′ | 0 lỗi API toàn bộ |

Output: `D:\Bit-Translate-data\gen_niche\` (idiom+keigo còn ở `data/synthetic/gen_niche/`;
`merge_niche_corpus.py` đọc CẢ HAI thư mục qua `--niche-dirs`).

**Corpus đã gộp** (`merge_niche_corpus.py --oversample 4`):
`D:\Bit-Translate-data\kd_filtered_niche.jsonl` — 10.525.064 gốc + 964.220 niche×4
= **11.489.284 cặp, niche 8,39%** (loại 57 câu ja gốc trùng niche). Rồi:
```
python scripts/prep_clean_split.py D:/Bit-Translate-data/kd_filtered_niche.jsonl D:/Bit-Translate-data/clean_v2
python scripts/binarize_ja2vi.py --clean D:/Bit-Translate-data/clean_v2 --out D:/Bit-Translate-data/bin_v2
modal volume put vija-100m-kd-vol D:/Bit-Translate-data/bin_v2 bin_v2
```

**⚠️ BẮT BUỘC DÙNG `--backend live` (mặc định).** Đo thật:
- Live (`gemini-3.1-flash-live-preview`, 60 session): **7.000 cặp/phút, 0 lỗi**.
- REST `gemini-flash-lite-latest`: 2.374/phút rồi bị RPM chặn → 217/phút, 425 lỗi.
- REST `gemini-flash-latest`: **BẪY**, alias `gemini-3.6-flash`, free tier **20 req/NGÀY**.

Live chỉ nhận `response_modalities=["AUDIO"]` (TEXT → `1007 not supported`; 4 model Live
khác đều 1008 not found). Text lấy qua `output_audio_transcription` — **đã kiểm chứng đó
là text GỐC, không phải ASR**: nó giữ nguyên dấu `{`, `"`, ```` ```json ```` và kanji
chính xác. Rác lẫn tiếng Việt chỉ 0,05% (là ghi chú thừa cuối câu, đã chặn bằng QC
`ja_lan_tieng_viet`).

### b) ĐÃ THỬ: beam search VÔ ÍCH, checkpoint averaging CÓ LỢI
| Decode / ckpt | chrF | slang | nguphap | thanhngu | caudai |
|---|---|---|---|---|---|
| greedy step15000 | 43,7 | 26,0 | 45,3 | 33,9 | 43,0 |
| beam 5 (α=0,6) | 43,6 | 26,6 | 48,3 | 33,6 | 43,4 |
| **avg5 (11000-15000)** | **44,4** | **30,1** | **49,0** | **35,7** | **44,1** |
| avg8 (8000-15000) | 42,8 | 27,3 | 46,9 | 32,3 | 45,2 |
| step11000 | 43,3 | 27,7 | 46,4 | 34,2 | 44,0 |

- **Beam 5 không cải thiện** (24 thắng/51 hòa/25 thua từng câu, không câu nào lệch >15
  chrF, độ dài hyp 105 vs 104). ⇒ model mắc *model error*, không phải *search error*.
  Đã có `generate_beam` trong `src/bitnet.py` + `--beam` trong `hardbench_ckpt.py` nếu
  cần đo lại, nhưng **đừng đầu tư thêm vào decode**.
- **avg5 được +0,7 chrF miễn phí**, slang +4,1. Dùng `scripts/avg_ckpt.py`. Average trên
  MASTER WEIGHTS FP rồi mới ép ternary.
- **Dải average có điểm ngọt** (đã quét đủ): avg3 44,1 · avg4 44,1 · **avg5 44,4** ·
  avg8 42,8. avg8 kéo về step8000 (LR còn cao) TỆ hơn cả ckpt đơn. Chỉ average sau khi
  loss đã phẳng. ⇒ **dùng `kd_avg5.pt`** làm bản eval/đóng GGUF.
- avg5 thắng ở CẢ **dev loss** (150 seq, ls=0): avg5 **0,9058** < step11000 0,9605 <
  step15000 0,9890 — averaging tốt hơn thật, không phải đánh đổi loss lấy generalization.
- **Đọc loss vòng 2 cho đúng**: step 10 của `train_v2` là **2,72**, KHÔNG phải dấu hiệu
  hỏng. = label smoothing 0,1 (+~0,35) + model gặp data THẬT SỰ MỚI (91,6% data cũ ở
  loss ~0,9 cộng 8,39% niche ở loss ~10 ≈ 1,66; cộng ls ≈ 2,0, cùng cỡ 2,72). Loss cao
  ở đây là TIN TỐT — chứng minh niche mang thông tin mới. Nếu nó ~1,0 mới đáng lo (data
  mới trùng data cũ ⇒ cả vòng vô nghĩa).
- Hệ quả cho §5: beam **không cứu được câu dài** (+0,4) ⇒ giả thuyết "câu dài nghi dung
  lượng" vẫn đứng, không phải lỗi decode.
- Trả lời câu "step11000 hay 15000 tốt hơn": chênh nhau chỉ +0,4 (43,3 vs 43,7) —
  **không cần chọn, average cả dải hơn cả hai**.

### b-BIS) MỐC JUDGE MÙ 100 CÂU cho `kd_avg5` (trước vòng enrich)
Lưu ở `eval/judge_avg5/BASELINE_avg5.json`; panel ở `eval/judge_avg5/panels/`
(dựng bằng `eval/build_panels_ja2vi.py <hyps> <outdir>` — arg 2 BẮT BUỘC nếu không
muốn ghi đè `panel_key.json` cũ).

| Hệ | acc | nat | % dùng được |
|---|---|---|---|
| KD-avg5 | 2,99 | 2,99 | **46%** |
| Google | 2,92 | 2,86 | 26% |
| Haiku | 3,70 | 3,56 | 69% |
| Fable | 4,98 | 4,99 | 100% |

Domain acc (KD/Google/Haiku) — **yếu**: thanhngu 1,8/2,2/3,4 · caudai 2,0/3,4/3,8 ·
slang 2,4/2,6/3,2 · keigo 2,7/2,9/3,1 · hop 2,7/2,9/3,9. **Mạnh**: solieu 4,8/4,0/4,8 ·
nguphap 3,8/3,2/3,8 · it_deep 3,4/3,0/3,7 · zeropronoun 3,3/2,9/3,7 · hoithoai 3,0/2,1/3,6.

**⚠️ THANG ĐO KHÔNG HIỆU CHUẨN GIỮA PHIÊN**: cùng hyp Google được 4,02 phiên trước,
2,92 phiên này. KHÔNG so số tuyệt đối với mốc cũ (3,57/55%); chỉ so tương đối giữa các
hệ trong CÙNG lần chấm. So trước/sau train thì phải chấm cả hai trong cùng phiên.

**⚠️ BẰNG CHỨNG MẠNH NHẤT cho "chrF nống điểm"**: ở `caudai`, chrF cho KD 44,1 (vượt
Google 42,8) nhưng judge cho 2,0 vs 3,4 (thua nặng) — bản dịch dài trùng nhiều ký tự
nên chrF cao dù cấu trúc sụp. ⇒ Kết luận "beam +0,4 chrF ở caudai" KHÔNG đáng tin.

KD phân bố **hai cực** (nhiều 5đ lẫn nhiều 1đ) trong khi Google dồn 2-3đ. Lỗi 1đ đúng
loại data niche vừa sinh: `元も子もない`→"không có con", `TBD`→"TMD", `リハーサル`→"really
staging", `見送らせて`→"tiễn biệt quý khách", `猫の手も借りたい`→"mượn tay của mèo".

### c) ĐÃ THÊM cho lần train tới
- `--label-smoothing` (`scripts/train.py`, `BitNetConfig`, cả 2 nhánh CE; tương thích
  ngược với ckpt cũ). Modal `BASE_ARGS` đã bật **0.1**. ⚠️ Loss in ra sẽ CAO hơn run cũ
  (0,757) khoảng +0,05-0,1 dù model không tệ hơn — **đừng so loss chéo giữa 2 run**.
- `--bin-dir` (train.py) + `--clean/--out` (binarize_ja2vi.py) để dùng data pha 2.
- `modal_train_100m_kd.py::finetune_p2` — **curriculum pha 2**: fine-tune nhẹ trên tập
  niche mật độ cao, lr 5e-5, `--lr-anchor` để LR không nhảy lại đỉnh, tự backup bản pha 1
  thành `phase1_final.pt`.

### d) ĐỘ PHỦ CORPUS đo thật (mẫu 300k dòng `kd_filtered.jsonl`)
| Domain | Độ phủ | chrF pilot |
|---|---|---|
| slang | **0,32%** | 26,0 (thấp nhất, domain duy nhất THUA Google) |
| phủ định phức (わけではない…) | **0,06%** | — (nhóm lỗi #4 đảo nghĩa) |
| câu dài ≥80 ký tự | 7,64% | 43,0 (Haiku 48,2) |

### e) GOTCHAS MỚI
12. **Ổ E chỉ 40GB, đã đầy 100%** → file lớn mới ghi vào `D:\Bit-Translate-data\`
    (user chốt dùng ổ D thay vì xóa). Các script đều có cờ `--outdir/--bin-dir/--out`.
13. **`modal volume get` có thể ra file corrupt IM LẶNG** (gặp thật với step8000:
    `PytorchStreamReader ... invalid header`) → luôn `torch.load` kiểm tra sau khi tải.
14. **Git Bash `/d/...`** chỉ được chuyển thành `D:/...` khi ở **argv**; nhúng trong
    string literal của `python -c` thì Python Windows không hiểu → dùng `D:/...`.
15. **Live API 60+ session** gây "timed out during opening handshake" → cần cổng
    `asyncio.Semaphore(12)` + stagger + **requeue task khi lỗi** (không requeue = mất
    trắng request đó).
16. **`torch.compile` warmup ~30 phút với data mới** (vs ~5 phút run cũ) vì phân bố độ
    dài rộng hơn (slang 8-40 ký tự + long 60-157) → `pad-multiple 32` sinh nhiều shape
    (B,T) hơn ⇒ nhiều lượt recompile. ĐỪNG kill sớm: tốc độ VỀ ĐÚNG 0,91 s/step
    (54k tok/s) từ step ~40. Mốc để đối chiếu: step10 91s → step20 48s → step30 28s →
    step40 0,90s.

---

## 8. BƯỚC TIẾP THEO (§8 cũ — phần data idiom/keigo ĐÃ XONG, xem 8-BIS)

**Đang xây: generator sinh data idiom/keigo qua REST Gemini** (chưa viết xong file).
Mục tiêu enrich data để test giả thuyết "nút thắt = data coverage".

**Kế hoạch data cần sinh (đòn bẩy tăng usable rate):**
- Thành ngữ: ~20-50k cặp (hiện ~0,x%).
- Keigo: ~50-100k cặp (hiện 2,6%).
- (Câu dài: BỎ QUA sinh thêm — corpus đã 21%, fail nghiêng dung lượng; xử lý bằng
  200M hoặc oversample câu dài sẵn có.)

**Cách sinh (đã kiểm chứng chạy tốt):**
- REST `client.models.generate_content(model="gemini-flash-latest", contents=prompt,
  config=GenerateContentConfig(response_mime_type="application/json", temperature=1.0))`.
- Chất lượng idiom sinh thử XUẤT SẮC (猫に小判→"đàn gảy tai trâu", 三日坊主→"cả thèm
  chóng chán"...). Structured JSON `[{ja, vi, idiom}]`.
- Cần: song song nhiều key (10 key trong .env), dedup theo câu ja, QC nhẹ
  (dùng `check()` trong `filter_kd_full.py`), lặp đến đủ số lượng. Đa dạng: seed
  danh sách idiom/kịch bản keigo trước (tránh sinh lặp mấy idiom phổ biến).

**Sau khi có data:**
1. Gộp gen_idiom + gen_keigo vào 10,5M → rule filter → binarize lại (`binarize_ja2vi.py`).
   Cân nhắc oversample phần niche (×3-5) để nó có trọng số đáng kể trong mix.
2. Train lại 100M from-scratch (hoặc resume) trên data giàu hơn (`modal_train_100m_kd.py`).
3. Eval hardbench + **judge lại** → so usable rate với 55% cũ.
   - Nếu thành ngữ/keigo TĂNG rõ → **xác nhận nút thắt là DATA**, tiếp tục enrich (rẻ).
   - Nếu phẳng dù data đã phủ → nghiêng *dung lượng*, lúc đó scale 200M (với data giàu hơn).

**Sau nữa (khi data đủ lớn):** scale 200M (`--d-model 1152 --n-layers 16 --n-heads 18
--d-ff 3072` ~290M, hoặc d1024/16L ~200M) — chỉ khi đã bớt đói data.

---

## 9. Quyết định & sở thích user (giữ nguyên)
- **ja→vi là sản phẩm CHÍNH**; pilot train **ja→vi 1 chiều thôi**, KHÔNG `>>fix<<`,
  KHÔNG `>>thinking<<`, auto-detect tiếng Nhật (chỉ thẻ `>>vie<<`).
- User **vặn kỹ, không thích nói quá** — phải trung thực về bằng chứng (đã sửa vụ
  overclaim "kịch trần dung lượng").
- Ý thức chi phí Modal (budget $30 trên account mới).
- Ưu tiên: rẻ + test giả thuyết trước khi đổ compute lớn (enrich data trước, 200M sau).
