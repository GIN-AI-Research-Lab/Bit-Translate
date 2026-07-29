# TỔNG KẾT MODEL — vòng 6 (2026-07-28)

> Mọi con số dưới đây là **đo thật**, kèm nguồn. Chỗ nào chưa đo thì ghi rõ **CHƯA ĐO**
> chứ không suy diễn. Chỗ nào không đạt ý nghĩa thống kê thì ghi rõ là nhiễu.

## ✅ CẬP NHẬT 2026-07-29: đã tìm ra + SỬA XONG — bản deploy đạt mức PyTorch

Nguyên nhân 16-22 điểm mất KHÔNG phải lượng tử hoá mà là **tokenizer**: spm là UNIGRAM,
llama.cpp chạy greedy-merge kiểu BPE → xé vụn từ trước khi vào model (ISSUES.md #4).
Fix: đưa TOKEN IDS tokenize sẵn qua llama-server `/completion`. Xác minh 200 câu cùng
phiên: **GGUF i2_s 74% vs PyTorch 76% (McNemar p=0,134 — ngang nhau); câu ngắn 85% =
Google 85%**. Model 78MB / 330 tok/s / CPU offline giờ THẬT SỰ đạt các số dưới đây.
Phần "18 điểm" bên dưới giữ nguyên làm hồ sơ lịch sử — đã lỗi thời.

## ⚠️ (LỖI THỜI — xem cập nhật trên) lượng tử hoá i2_s làm mất 18 ĐIỂM

Chấm mù **4 hệ trong CÙNG MỘT PHIÊN**, 200 câu (`eval/judge_final/RESULT.json`):

| | Google | **v6 PyTorch** (avg7) | v6 GGUF i2_s (avg7) | GGUF step16400 |
|---|---:|---:|---:|---:|
| **TỔNG** (n=200) | 86% | **74%** | **56%** | 54% |
| **câu NGẮN** | 82% | **86%** ← thắng Google | **66%** | 63% |
| **câu DÀI** | 91% | **62%** | **45%** | 46% |

| McNemar | kết quả |
|---|---|
| PyTorch vs GGUF (**cùng checkpoint avg7**) | 44 hơn / 7 kém → **p=0,000, mất 18 điểm** |
| GGUF avg7 vs GGUF step16400 | 16 vs 14 → **p=0,855, KHÔNG khác nhau** |
| PyTorch vs Google | 20 hơn / 45 kém → p=0,003 |

**Nguyên nhân đã cô lập được: LƯỢNG TỬ HOÁ, không phải chọn checkpoint.** Cùng một
`avg7` mà PyTorch 74% còn GGUF 56%; đổi checkpoint trong dạng GGUF thì không đổi gì.

Hệ quả cho cả hướng đi: **bản PyTorch THẮNG Google ở câu ngắn (86% vs 82%) nhưng chưa
deploy được**; bản deploy được thì kém Google 20 điểm. Đây là chỗ chặn hướng "đóng gói cái
đang thắng". Xem **`ISSUES.md` #4**.

Các con số ở §2 dưới đây là của **bản PyTorch** — đúng cho nghiên cứu, KHÔNG phải bản deploy.

---

## 1. Model là gì

| | |
|---|---|
| Kiến trúc | Transformer encoder-decoder, **18 lớp, d_model 768, FFN 2048, 12 heads** |
| Tham số | **152,1M** — trong đó 127,5M ternary + 49,2M embedding/lm_head |
| Lượng tử | **BitNet b1.58** — weight ternary (−1/0/+1), train from-scratch (QAT + STE) |
| Vocab | SentencePiece 32.001, dùng chung VI+JA, thẻ hướng `>>vie<<` / `>>jpn<<` |
| `max_seq` | **256 token cho input + output CỘNG LẠI** (bảng RoPE 256 hàng) |
| Chiều tối ưu | **ja→vi** (vi→ja đã hạ ưu tiên từ vòng 3) |
| Checkpoint | `v6_avg.pt` = trung bình 7 mốc step 15000–16400 |

**Cách train**: knowledge distillation — 100% data là bản dịch do LLM thầy sinh, từ câu
tiếng Nhật đơn ngữ đào trong CC-100 (392,8M câu). Thầy: `gemini-3.1-flash-live-preview`.
Corpus vòng 6: **15,28M cặp / 858,4M token**. Train 16.400 step (≈2,50 epoch) trên
Modal L40S, ~$16.

---

## 2. Chất lượng dịch — so Google Translate

Bench 200 câu tiếng Nhật thật (TED/OpenSubtitles), cân bằng 10 đặc trưng × 20 câu,
100 ngắn / 100 dài. **LLM-judge chấm MÙ** (các hệ xáo thành A/B/C, không có bản tham
chiếu — judge chấm thẳng từ câu nguồn). Thang **0/1/2 về ĐÚNG NGHĨA**, `%` = tỉ lệ đạt 2đ.
**Cả 3 hệ chấm trong CÙNG MỘT PHIÊN** (thang judge không hiệu chuẩn được giữa các phiên).

| | Google | v5 | **v6** | McNemar v6 vs Google |
|---|---:|---:|---:|---|
| **câu NGẮN** (n=100) | 78% | 80% | **81%** | p=0,710 → **ngang nhau** |
| **câu DÀI** (n=100) | 87% | 55% | **64%** | p=0,001 → **Google hơn thật** |
| **TỔNG** (n=200) | 82% | 68% | 72% | p=0,025 → Google hơn thật |

### Ba kết luận đứng vững

1. **Câu ngắn: ngang Google.** 81% vs 78%, p=0,710 — thống kê không phân biệt được.
2. **Câu dài: kém Google rõ rệt.** −23 điểm, p=0,001.
3. **v6 hơn v5: CHƯA chứng minh được.** Tổng p=0,144 · dài p=0,124 · ngắn p=1,000.

### Điểm theo từng đặc trưng — TOÀN BỘ LÀ NHIỄU, đừng dùng để quyết định

| đặc trưng | v5 | v6 | Google | v6 vs GG: p |
|---|---:|---:|---:|---:|
| plain | 70% | 90% | 85% | 1,00 |
| negation | 75% | 95% | 90% | 1,00 |
| idiom | 65% | 75% | 60% | 0,45 |
| slang | 50% | 60% | 80% | 0,34 |
| clause | 55% | 65% | 90% | 0,13 |
| question | 70% | 70% | 85% | 0,37 |
| katakana | 70% | 70% | 90% | 0,29 |
| zeropron | 65% | 60% | 85% | 0,18 |
| keigo | 70% | 65% | 75% | 0,75 |
| number | 85% | 75% | 85% | 0,72 |

n=20 mỗi ô → KTC 95% khoảng **±20 điểm**. Ô `negation 95% vs 90%` thực chất chỉ là
**2 câu hơn / 1 câu kém**. Không ô nào đạt p<0,05, kể cả những ô Google hơn nhiều.

### ⚠️ Thiên vị đã biết của phép đo này

Judge là **Gemini**, mà data train v6 do **Gemini** sinh ra → v6 viết theo văn phong
Gemini và Gemini đang chấm. Bằng chứng: Google được **82%** ở phiên này nhưng **87%** ở
phiên chấm trước bằng judge khác. Nên khoảng cách thật với Google **có thể lớn hơn** con
số trong bảng.

---

## 3. So với Claude Haiku — CHƯA ĐO ĐƯỢC

**Không có phép so cùng phiên nào giữa v6 và Haiku.** Đừng dùng bảng nào để kết luận.

Hai mảnh dữ liệu rời rạc hiện có:

| nguồn | nội dung | vì sao không dùng được |
|---|---|---|
| `STATUS.md` 4-way (2026-07-17) | acc ja→vi: **110M 1,90** · Google 3,73 · **Haiku 4,45**; dùng được: 6% / 60% / **92,5%** | Đó là model **110M vòng cũ**, không phải v6. Model đã đổi kiến trúc (12L→18L) và đổi toàn bộ data từ đó |
| hardbench chrF | v6_avg **49,5** · Google 36,1 · Haiku **41,5** | chrF **không dùng để so giữa các hệ** — xem §6 |

**Việc cần làm để có số thật**: dịch bench 200 câu bằng Haiku rồi chấm mù 4 hệ
(v6/Google/Haiku + 1 mốc) trong một phiên. Chưa làm.

---

## 4. Tốc độ, dung lượng, cấu hình chạy

### Đo thật — đường PyTorch (dùng để nghiên cứu, KHÔNG phải bản triển khai)

Trên **Intel Core Ultra 5 225H**, 5 luồng, đã cắm sạc, FP32 eager:

| loại câu | token vào | token ra | giây | tok/s |
|---|---:|---:|---:|---:|
| ngắn (12 ký tự) | 8 | 7 | 0,26 | 26,5 |
| vừa (33 ký tự) | 18 | 19 | 0,57 | 33,5 |
| dài (110 ký tự) | 55 | 69 | 2,31 | 29,9 |
| **trung bình** | | | **1,05 s/câu** | **30,3** |

### Đường triển khai thật — GGUF i2_s + bitnet.cpp, **ĐÃ ĐO 2026-07-28**

Convert `step16400.pt` → GGUF F32 → `llama-quantize I2_S 1` → `llama-bench`.
Trên **Intel Core Ultra 5 225H**, đã cắm sạc:

| | |
|---|---|
| **File trên đĩa** | **77,56 MB** (từ 337 MB F16) |
| **RAM khi chạy** | **124,43 MiB** |
| **Sinh token** | **353 tok/s** @ 8 luồng · **345** @ 6 luồng |
| **Xử lý prompt** | **2.705 tok/s** @ 8 luồng |
| So với PyTorch FP32 | **nhanh hơn 11,7×** (353 vs 30,3) |

### Số luồng — đường cong đo thật, khác hẳn số cũ trong `STATUS.md`

| luồng | 4 | 5 | 6 | **8** | 10 | 12 | **14** |
|---|---:|---:|---:|---:|---:|---:|---:|
| tok/s | 301 | 301 | 345 | **353** | 171 | 208 | **0,32** |

- **Điểm ngọt: 6–8 luồng.**
- 10–12 luồng: tụt và **phương sai rất lớn** (±54, ±71) — E-core làm hại
- **14 luồng (dùng hết lõi): sụp còn 0,32 tok/s — chậm hơn 8 luồng 1.100 lần.** Tuyệt đối
  không để runtime tự chọn số luồng.

⚠️ Số cũ trong `STATUS.md` (140 tok/s @ 4–6, 102 @ 8, 21 @ 12) là của **model 110M vòng
cũ với bản build bitnet.cpp cũ** — đo lại hôm nay cho **2–2,5× nhanh hơn** và điểm ngọt
dịch từ 4–6 sang **6–8 luồng**. Dùng số mới.

### Cấu hình tối thiểu để chạy

| | |
|---|---|
| CPU | Bất kỳ CPU có **AVX2**. Không cần GPU |
| RAM | **~125 MB** cho model + KV cache |
| Đĩa | **78 MB** |
| Số luồng | **6–8**, đặt cứng bằng `-t 8`. Xem bảng trên |
| Nguồn điện | Laptop **phải cắm sạc**: trên pin clock tụt 1,70 → 1,22 GHz |
| Mạng | Không cần — chạy hoàn toàn offline |
| Chi phí | 0đ/câu |

Lệnh chạy:

```bash
llama-cli -m v6_s16400_i2s.gguf -p ">>vie<< <câu tiếng Nhật>" -n 200 -t 8 --temp 0
```

### ⚠️ Bản `v6_avg.pt` KHÔNG convert được — xem `ISSUES.md` #2

Checkpoint tốt nhất (`v6_avg`, trung bình 7 mốc, dev loss 2,0761) **sinh ra GGUF hỏng**:
output là rác `ùcccccc…`. Mốc đơn `step16400.pt` thì chạy tốt. Nên **bản deploy hiện tại
là `step16400`, không phải `avg7`** — tức mất phần −0,0125 dev loss mà averaging mua được.

**Lưu ý về song song**: inference là **memory-bandwidth bound**. Chạy 2 tiến trình model
cùng lúc KHÔNG nhanh hơn — đo thật: một job bị bỏ đói xuống 0,74 lõi và chậm **5,6×**.
Chạy tuần tự.

---

## 5. Vấn đề còn mở: câu dài

Khoảng cách −23 điểm ở câu dài là **năng lực model thật**, không phải lỗi kỹ thuật có thể
vá. Bốn hướng đã thử hoặc đã loại bằng số liệu:

| hướng | kết quả |
|---|---|
| **Tách câu lúc inference** (cách Google làm) | **ĐÃ ĐO, KHÔNG ĂN.** 100 câu Quốc hội có ref: nguyên câu chrF **57,28** · tách thận trọng **57,38** (+0,10 = nhiễu) · tách mọi dấu `、` **54,74** (−2,54, tệ hơn) |
| **Nới `max_seq`** | Chỉ **0,10%** câu thật vượt 256 token (đo trên 120.000 câu biên bản Quốc hội: trung vị 80, p99 220). Không phải nút thắt |
| **Ghép cặp ngắn thành cặp dài** | Ghép-4: **32,4% vượt 256** bị `binarize` vứt, trung vị 210 token **chưa chạm** vùng 224 cần lấp |
| **Thinking / self-edit / rerank** | Oracle chọn bản tốt nhất trong 9 bản chỉ **+5,2 chrF** ⇒ *"chất lượng KHÔNG giấu trong weights"*. `>>fix<<` chỉ pattern-match |

**Vì sao tách câu không ăn**: cả 100 câu đều **vừa** `max_seq`, model dịch trọn không bị
cắt. "Câu dài" không phải vấn đề độ dài — nó khó vì **nhiều mệnh đề lồng nhau**, và không
thủ thuật đóng gói nào sửa được chuyện hiểu sai.

Chi tiết đầy đủ: **`ISSUES.md` #1**.

---

## 6. Vì sao KHÔNG dùng chrF để so với hệ khác

Cùng một model, đổi người viết bản tham chiếu là **đổi luôn dấu**:

| bộ đo | ref do ai viết | model | Google | chênh |
|---|---|---:|---:|---:|
| hardbench | LLM soạn | **49,5** | 36,1 | **+13,3** |
| FLORES-200 ja→vi | người dịch chuyên nghiệp | 42,3 | **53,5** | **−11,2** |

Và chrF xếp hệ tốt hơn hẳn xuống thấp hơn: Haiku (**92,5%** dùng được) có FLORES chrF
**51,1**, thấp hơn Google (60% dùng được) **53,5**.

⇒ chrF chỉ dùng làm **canh hồi quy giữa các vòng trên cùng bộ đề**. Ở vai đó nó hữu ích:
hardbench v3 48,0 → v5 47,7 → **v6 49,5**.

---

## 7. Thước đo phụ

| thước đo | v5 | **v6** | ghi chú |
|---|---:|---:|---|
| Dò ngữ pháp 80 phép thử (tự động) | 71/80 | **76/80** | Đứng yên ở 71 suốt v3→v5. v6 **sửa 7 phép thử câu DÀI**, hỏng 2 câu NGẮN |
| hardbench chrF | 47,7 | **49,5** | canh hồi quy, không so hệ khác |
| dev loss (held-out) | — | **2,0761** | Lần đầu dự án có. ⚠️ **bị rò 12,56%** — xem dưới |

### ⚠️ Dev set bị rò 12,56%

**558/4.444 câu dev có bản sao `ja` y hệt trong train** (506 trùng 1 lần, 52 trùng 2 lần).
Nguyên nhân: corpus có nhân bản chủ ý từ các vòng trước (termgap ×6, quán ngữ ×3), mà
`prep_clean_split.py` chia theo hash `id` nên bản sao rơi cả hai bên. **Mọi kết luận dựa
trên dev loss ở biên độ 0,01–0,03 phải xem lại.** Chưa lọc lại.

### Đường cong dev loss (chưa lọc rò)

| step | epoch | dev_loss | Δ/1000 |
|---:|---:|---:|---:|
| 2000 | 0,31 | 2,1846 | – |
| 6000 | 0,92 | 2,1505 | −0,0099 |
| 10000 | 1,53 | 2,1221 | −0,0067 |
| 14000 | 2,14 | 2,0946 | −0,0061 |
| 16000 | 2,44 | 2,0896 | **−0,0025** |
| 16400 | 2,50 | 2,0886 | **−0,0025** |
| **avg7** | – | **2,0761** | — |

Hai điều đọc được: (a) dev **phẳng hẳn** ở 2.400 step cuối trong khi train loss ở đó lại
dốc nhất (−0,0126) — chỉ loss train thì tưởng đang học nhanh nhất; (b) **trung bình 7
checkpoint mua được −0,0125 dev loss, miễn phí** — tốt hơn mọi checkpoint đơn lẻ.

---

## 8. Đánh giá thẳng

**Đã đạt được**: một model **152M ternary, ~81MB, chạy CPU offline 0đ/câu, dịch ja→vi
ngang Google Translate ở câu ngắn** (81% vs 78%, p=0,710). Đó là kết quả thật và đứng
vững qua phép kiểm bắt cặp.

**Chưa đạt**: câu dài kém Google 23 điểm (p=0,001), và bốn hướng vá rẻ đều đã bị số liệu
loại. Đây là giới hạn dung lượng, không phải lỗi kỹ thuật.

**Sáu vòng data đã chứng minh**: bơm data làm điểm **trong miền** nhảy (v5: Quốc hội
14%→95%) nhưng bench chung **đứng yên hoặc trong nhiễu** (v3 hardbench +18 mà bench thật
không đổi; v6 tổng +4 với p=0,144). Cộng thêm: 81% câu hỏng có **≥2 nguyên nhân** nên sửa
trọn một mặt trận chỉ cứu 2–3 câu trên 59.

**Hai đường còn lại**:
- **Đóng gói cái đang thắng** — convert GGUF, chạy bitnet.cpp, dùng cho câu ngắn/vừa.
  Ba trục Google không cạnh tranh được: **offline, 0đ, ~81MB**.
- **Tăng dung lượng** — đường rẻ đã kiểm chứng là chèn block identity để mở sâu (rẻ hơn
  3× so với train lại). Ước **$150–250** theo `PLAN_RANKUP_292M.md`, không phải $9.

**Không nên**: vòng data thứ 7. Đó là đường sáu vòng vừa rồi đã đi, mỗi bước đều trong nhiễu.

---

## 9. Nguồn số liệu

| số | file |
|---|---|
| Điểm judge từng câu, 3 hệ | `eval/judge_v6/RESULT.json` + `eval/bench_new.jsonl` |
| Bản dịch v6 bench 200 câu | `eval/bench_v6.jsonl` |
| 120 câu Quốc hội held-out | `eval/bench_held_v6.jsonl` |
| Test tách câu | `eval/split_test_v6.jsonl` |
| Đường cong dev loss | `eval/dev_curve_v6.json` |
| Dò ngữ pháp | `eval/probe_v6.jsonl` |
| Mật độ kỹ năng corpus | `eval/skill_density.json`, `eval/skill_density_v6new.json` |
| Phân loại nguyên nhân lỗi | `eval/error_taxonomy_v5.json` |
| Log train + báo cáo gộp | `logs/KETQUA_V6.txt`, `D:/Bit-Translate-data/checkpoints_v6/train.log` |
| Phản biện độc lập | `eval/peer_review_v7.md` (nemotron) |
| Issue câu dài | `ISSUES.md` #1 |
| Phân loại artifact nặng/nhẹ | `ARTIFACTS.md` |
