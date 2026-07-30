# ISSUES — vấn đề đã xác định, CHƯA giải quyết

> Mỗi issue phải có: số đo, cách kiểm lại, và cái gì đã thử/đã loại. Không ghi phỏng đoán.

---

## #1 — Câu dài: vị trí token >160 gần như chưa được luyện (MỞ — có tiến triển V7A)

**Phát hiện 2026-07-28, sau vòng 6.**

### Tiến triển V7A (2026-07-30) — CHƯA ĐÓNG

Vòng V7A bơm data câu dài (đào theo độ dài + ghép liền kề, `PLAN_V7A.md` §3a). Kết quả
đo trên bench 200b chấm mù cùng phiên (`eval/judge_200b_claude/KETQUA.md`):

| acc==2 câu dài (n=100) | v6 | gate | **v7a i2s** |
|---|---:|---:|---:|
| | 67% | 65% | **73%** |

**+6 điểm so v6, +8 so gate — đúng chỗ nhắm.** Nhưng vẫn dưới câu ngắn 11 điểm
(73 vs 84) và nat==2 câu dài mới 64% ⇒ issue **giữ trạng thái MỞ**. Bench 200b câu dài
= 96–228 ký tự (trung vị 150, đo trên `bench_opus200b.jsonl`) — khác bench TED cũ
(72–77 ký tự) nên lần này CÓ đo được vùng dài; câu Quốc hội thật vẫn nên kiểm thêm
bằng `bench_held_*.jsonl` như ghi ở dưới.

### Số đo

Độ phủ vị trí trong `bin_v6` (15.284.511 chuỗi, dài trung bình 56,2 token):

| vị trí token | số chuỗi phủ tới đó | % corpus |
|---:|---:|---:|
| 32 | 9.579.413 | 62,67% |
| 64 | 4.342.143 | 28,41% |
| 96 | 2.484.702 | 16,26% |
| 128 | 1.481.337 | 9,69% |
| 160 | 734.086 | 4,80% |
| 192 | 268.404 | 1,76% |
| 224 | 58.740 | **0,38%** |
| 256 | 685 | **0,00%** |

`max_seq = 256` là trần **cứng** (bảng RoPE có 256 hàng) cho **input + output CỘNG LẠI**, vì
`generate_cached` giữ cả hai trong một chuỗi. Tỉ lệ token VI/JA đo trên 15,28M cặp:
**1,12** (trung vị), 1,41 (p90). Với ~2,05 ký tự/token ở văn phong Quốc hội, câu Nhật quá
**~245 ký tự** sẽ bị cắt output.

### Không phải khan hiếm nguồn — là chưa bao giờ chọn theo độ dài

`kokkai_ja.txt`, 2 triệu dòng đầu: **7,9% câu ≥150 ký tự, 1,3% ≥200 ký tự**
(158.646 và 25.144 câu). CC-100 còn 392,8M câu. Các script đào (`mine_skills2.py`,
`mine_rare.py`) chọn theo **KHUÔN MẪU** (mệnh đề lồng, slang, phủ định), **chưa lần nào
chọn theo ĐỘ DÀI**.

### Đã đo được gì về chất lượng thực tế

Trên 3 câu Quốc hội dài nhất (198–209 ký tự, 12–14 dấu `、`, không có `。` ở giữa):
`v6_avg` dịch **trọn cả ba, không bị cắt** (tổng 197/218/224 token, vừa dưới 256).
So với Gemini trên cùng 3 câu: v6 **rơi từ và sai thuật ngữ**
(`税理士/会計士` → "kế toán thuế / kế toán tăng"; mất `融資だと`, mất `税収減`),
Gemini đúng cả. ⇒ **v6 không gãy vì độ dài, mà kém vì năng lực.**

⚠️ Bench 200 câu **KHÔNG đo được** issue này: câu "dài" nhất trong đó chỉ **72–77 ký tự**,
còn Quốc hội held-out là 130–218. Muốn theo dõi issue này phải dùng `bench_held_*.jsonl`.

### Bốn hướng, xếp theo giá trị/chi phí

1. **Ghép cặp ngắn thành cặp dài** — nối 3-4 cặp có sẵn bằng `、`/`そして`. **$0, không tốn
   quota KD**, nguồn vô hạn. Sửa được độ phủ vị trí + dạy "câu dài thì dừng ở đâu".
   KHÔNG dạy được cấu trúc lồng thật (câu ghép không có phụ thuộc chéo mệnh đề).
2. **Đào theo độ dài rồi KD** — cách duy nhất dạy cấu trúc lồng thật. Câu dài tốn ~4×
   token/cặp, nên 200k cặp dài ≈ 800k cặp ngắn về quota. 200k đủ để nhân 4× độ phủ vị trí 224.
3. **Tách câu lúc inference** (cách Google làm) — không train lại. Rủi ro: tiếng Nhật đặt
   động từ CUỐI câu, vị ngữ cuối chi phối các mệnh đề trước; cắt ngang là mất mối đó.
   **Phải đo trước khi tin** — kiểm rẻ trên 120 câu `heldout_kokkai.txt`.
4. **Nới `max_seq` bằng RoPE interpolation — ĐỪNG LÀM TRƯỚC.** Cửa sổ 256 hiện tại còn chưa
   lấp đầy (vị trí 224 mới 0,38%); nới lên 512 chỉ làm vùng chưa luyện rộng thêm.
   Chỉ xét sau khi (1) hoặc (2) đã lấp vùng 160–256.

### Đã sửa (không phải giải quyết issue, chỉ hết crash)

`src/bitnet.py generate_cached()` giờ dừng gọn khi `pos + T > max_seq` thay vì nổ
`RuntimeError: size of tensor a (0) must match tensor b (768)`. Lỗi cũ tiềm ẩn ở **mọi**
caller (`translate_bench.py`, `hardbench_ckpt.py`, `translate_txt.py`, demo) — chỉ chưa lộ
vì input ngắn hơn. Nguyên nhân crash là **`--max-new 200` + prompt 102 token = 302 > 256**,
tức độ dài SINH RA, không phải độ dài đọc vào.

### Cách kiểm lại issue

```bash
python scripts/translate_txt.py <ckpt> D:/Bit-Translate-data/raw/heldout_kokkai.txt \
    eval/bench_held_vN.jsonl --label vN
```
Rồi đếm % câu có output bị cắt (không kết thúc bằng dấu câu), và so độ phủ vị trí bằng
đoạn đo trong mục "Số đo" ở trên.

---

## #2 — ~~Checkpoint trung bình không convert được~~ **ĐÃ BÁC — CHẨN ĐOÁN SAI**

**Kết luận ban đầu 2026-07-28 là SAI. Bác cùng ngày.**

Tôi kết luận `v6_avg.pt` sinh GGUF hỏng, dựng cả bảng 6 giả thuyết để "cô lập lỗi".
Toàn bộ dựa trên tiền đề sai: tôi test bằng **prompt thiếu cả `<s>` và `</s>`**.
Với prompt đúng, `v6_avg_i2s.gguf` chạy **hoàn hảo** và còn dịch tốt hơn `step16400`
(`資料` → "tài liệu của cuộc họp", đúng; step16400 → "lịch trình họp", sai).

Sai lầm phương pháp: tôi test `v4_avg5_i2s.gguf` thấy nó ra tiếng Việt nên tin prompt
đúng. Nhưng v4 chạy được với prompt sai chỉ là **may** — nó vẫn sai nghĩa, chỉ trông như
tiếng Việt nên tôi không nhận ra. **Bài học: xác nhận invocation bằng một câu đã biết đáp
án, đừng chỉ xem output có "trông giống ngôn ngữ đích" hay không.**

Xem issue thật ở **#3** dưới đây.

---

## #3 — Prompt GGUF phải có ĐỦ `<s>` và `</s>` (ĐÃ SỬA)

`tokenizer.ggml.add_bos_token = false` trong GGUF, nên `llama-cli` **không tự thêm `<s>`**,
trong khi model được train trên `[bos, >>vie<<, ja..., eos]`.

Format đúng: `<s>>>vie<< <câu tiếng Nhật></s>`

| thiếu gì | hậu quả đo thật |
|---|---|
| thiếu `</s>` | dịch **sai nghĩa** + chèn ký tự rác `ù` ở đầu |
| thiếu `<s>` | **35% câu (70/200) sinh ra RỖNG** |
| thiếu cả hai | rác hoàn toàn (`ùcccccc…`) — chính là thứ làm tôi chẩn đoán sai #2 |

⚠️ **MỌI số đo qua đường GGUF trước 2026-07-28 đều dùng prompt thiếu cả hai token** —
gồm bench 292M vòng trước. **Không dùng được, phải đo lại.**

Đã sửa trong `scripts/translate_bench.py`, kèm hai lỗi phụ:
- Cờ `-no-cnv` không tồn tại trong bản build hiện tại (`invalid argument`)
- `split(prompt)` không khớp vì llama-cli chuẩn hoá khoảng trắng → hyp dính cả prompt.
  Phải neo theo 10 ký tự cuối câu nguồn và lấy dòng **không rỗng** đầu tiên

---

## #4 — llama.cpp TOKENIZE SAI: tokenizer là UNIGRAM, llama.cpp chạy greedy-merge kiểu BPE (ĐÃ SỬA + XÁC MINH 2026-07-29)

### 🎯 NGUYÊN NHÂN GỐC — xác nhận NHÂN QUẢ HAI CHIỀU

`spm_vija_32k.model` là **UNIGRAM** (`trainer_spec.model_type=1`, Viterbi tối ưu toàn cục).
GGUF ghi `tokenizer.ggml.model="llama"` → llama.cpp dùng `llm_tokenizer_spm` = **greedy
bigram-merge kiểu BPE** (`llama-vocab.cpp:203`), không bao giờ tái tạo được phân đoạn
unigram. Từ thường gặp bị XÉ VỤN trước khi vào model:

| từ | sentencepiece (đúng, format train) | llama.cpp (vỡ) |
|---|---|---|
| してください | **1594** (1 token) | して+く+だ+さい (4 token) |
| ツイート | **17062** | ツ+イー+ト |
| に対して | **2044** | に+対+して |
| チャンピオン | **13513** | チャン+ピ+オン |

Kèm lỗi phụ cùng nguồn: **token `▁`(262) bị ĐÚP** sau `>>vie<<` (llama.cpp không chạy
NFKC/remove_extra_whitespaces của sentencepiece; space thật trong prompt + dummy-prefix).
Đo 30 câu đầu bench: **ids lệch 30/30** (24 resegmentation thật), llama sinh 518 token vs
434 chuẩn (+19,4%). Chuỗi token out-of-distribution nằm TRƯỚC mọi phép nhân trọng số ⇒
mọi format (i2_s/Q8_0/Q4_0/F16) mất y hệt — khớp toàn bộ quan sát.

### Phép thử vàng hai chiều (n=30, so text trực tiếp)

| phép thử | identical | chrF |
|---|---:|---:|
| Harness check: PyTorch regen vs bench đã lưu | 30/30 | 100,0 |
| **XUÔI**: PyTorch + **ids-của-llama** vs GGUF-hỏng đã lưu | **24/30 (80%)** | **96,1** |
| đối chứng: PyTorch-chuẩn vs GGUF-hỏng | 7/30 (23%) | 73,5 |
| **NGƯỢC**: GGUF i2_s + **ids-chuẩn** (qua llama-server /completion) vs PyTorch đã lưu | **26/30 (87%)** | **97,0** |

⇒ Chỉ đổi ids: PyTorch tái tạo gần nguyên văn output hỏng; GGUF hồi phục về mức PyTorch
**trên chính runtime bitnet.cpp**. Phần số học/kernel/graph VÔ TỘI. Phần dư 4-6/30 câu là
lệch từ đồng nghĩa do nhiễu i2_s + KV f16 — vô hại.

### ✅ FIX ĐÃ ÁP + XÁC MINH TRÊN 200 CÂU

`run_gguf` trong `scripts/translate_bench.py` đã viết lại: llama-server + POST
`/completion` với MẢNG token ids (tokenize bằng sentencepiece + normalize_for_model,
y hệt lúc train). Bench 200 câu chấm mù CÙNG PHIÊN 4 hệ (`eval/judge_fixed/`):

| | TỔNG | câu ngắn | câu dài |
|---|---:|---:|---:|
| Google | 86% | 85% | 88% |
| PyTorch (v6pt) | 76% | 85% | 66% |
| **GGUF i2_s qua ids đúng (v6fixed)** | **74%** | **85%** | 62% |
| GGUF tokenizer hỏng (i2sold, đối chứng) | 58% | 69% | 48% |

McNemar `v6fixed vs v6pt`: 0 hơn / 4 kém → **p=0,134, không phân biệt được** — bản deploy
đã đạt đúng mức PyTorch. `v6fixed vs i2sold`: 39/9, p=0,000 → fix mua lại **+16 điểm**.
**Câu ngắn bản deploy: 85% = Google 85%.** Tốc độ bench: 200 câu/53s (llama-cli cũ: 2.569s).

### FIX gốc (đã áp)

**Không bao giờ để llama-cli/llama-server tự tokenize câu nguồn.** Sửa `run_gguf` trong
`scripts/translate_bench.py`: bỏ subprocess llama-cli với chuỗi; chạy `llama-server -m
<gguf> -t 4 -c 256` một lần, mỗi câu POST `/completion` với `"prompt": [2,4] +
sp.encode(normalize_for_model(src)) + [3]` (MẢNG ids), temp 0. Bỏ luôn anchor-split parsing.
Bonus: 30 câu/3s vì model nạp 1 lần. Code mẫu đã chạy được: scratchpad workflow
`server_client.py`. Dài hạn (CHƯA KIỂM): ghi `tokenizer.ggml.model="t5"` để dùng
`llm_tokenizer_ugm` (unigram native trong llama.cpp) — phải kiểm special-token trước.

**Mọi số GGUF đã công bố (64%, 54-56%) đo qua tokenizer hỏng → ĐANG ĐÁNH GIÁ THẤP model.
Sau fix phải bench 200 câu + chấm mù CÙNG PHIÊN với PyTorch; kỳ vọng ~80%.**

### Hành trình loại trừ (giữ để không ai đi lại)

### ❌ Giả thuyết "sai hàm kích hoạt" — BỊ BÁC (2026-07-29)

Tôi đọc `models/bitnet.cpp` của bản llama.cpp **MỚI** thấy `LLM_FFN_SILU`, rồi kết luận
llama.cpp tính SiLU trong khi model train bằng ReLU². **Sai** — bản đang CHẠY là bản cũ
(`~/BitNet-test`), và `llama.cpp:15501` của nó **đã dùng `LLM_FFN_RELU_SQR`**, tức đúng.

Sai lầm: đọc code của bản A rồi kết luận cho bản B đang chạy.

**Nhưng có phụ phẩm đáng ghi:** bản llama.cpp MỚI (submodule `390c307`) dùng `LLM_FFN_SILU`
trong `models/bitnet.cpp:133` — đó là **hồi quy của upstream**. Ai nâng version phải patch
lại thành `LLM_FFN_RELU_SQR`, nếu không sẽ tự tạo ra đúng lỗi này.

### Nguyên nhân THẬT vẫn chưa biết

| | hàm kích hoạt FFN |
|---|---|
| Model được TRAIN (`src/bitnet.py:24`) | **`F.relu(g_pre).square()`** — ReLU bình phương |
| llama.cpp TÍNH (`models/bitnet.cpp:133`, và `build_bitnet` bản cũ) | **`LLM_FFN_SILU`** — SiLU/Swish |

**Sai hàm kích hoạt.** Không phải lượng tử hoá, không phải convert, không phải kernel.

### Cách sửa — MỘT DÒNG

```c
LLM_FFN_SILU  →  LLM_FFN_RELU_SQR
```
Enum đã có sẵn: `llama-graph.h:46`, xử lý ở `llama-graph.cpp:1690`.
⚠️ **Cả bản cũ VÀ bản mới đều dùng SILU** → nâng version KHÔNG tự sửa được, phải patch.

### Vì sao giải thích được TẤT CẢ quan sát

| quan sát | giải thích |
|---|---|
| i2_s 54%, Q8_0 54%, Q4_0 52% — mất như nhau | hàm kích hoạt không phụ thuộc định dạng lưu |
| GGUF→PyTorch = 80%, p=1,000 so checkpoint | converter đúng; PyTorch dùng ReLU² |
| Tắt activation *quantization* vẫn 80% | đó là chuyện khác hẳn |
| Output hợp lý chứ không thành rác | ReLU² và SiLU đều trơn, dương → suy giảm chứ không vỡ |
| 14 commit upstream không có fix | cả hai bản đều SILU |

### Đã loại bằng thực nghiệm trước khi tìm ra (giữ lại để không ai đi lại)



**Phát hiện 2026-07-28.** Cùng một checkpoint `v6_avg`, chỉ khác đường chạy:

| | TỔNG | câu ngắn | câu dài |
|---|---:|---:|---:|
| **PyTorch** (`freeze_for_inference`, ternary + GEMM fp32) | **74%** | **86%** | **62%** |
| **GGUF i2_s** (bitnet.cpp, ternary packed 2-bit) | **56%** | 66% | 45% |
| Google (đối chiếu) | 86% | 82% | 91% |

McNemar `PyTorch vs GGUF`: **44 câu hơn / 7 câu kém → p=0,000**. Chấm mù cùng phiên,
4 hệ, judge Gemini. Nguồn: `eval/judge_final/RESULT.json`.

**Đối chứng loại trừ nguyên nhân "chọn sai checkpoint"**: `avg7` GGUF vs `step16400` GGUF
= 16 vs 14 câu, **p=0,855 → không khác nhau**. Nên 18 điểm mất KHÔNG do checkpoint.

### Vì sao nghiêm trọng

Cả tiền đề của BitNet là ternary mất rất ít chất lượng. PyTorch **cũng đang chạy ternary**
(`freeze_for_inference` gọi `weight_quant`), vậy 18 điểm mất phải đến từ chỗ khác:
- Lượng tử hoá **activation** 8-bit trong kernel i2_s
- Hoặc cách tính **scale per-tensor** của `llama-quantize` khác `weight_quant` của dự án
- Hoặc lệch layout khi pack 2-bit

Đây là thứ chặn toàn bộ hướng "đóng gói cái đang thắng": bản PyTorch **thắng Google ở câu
ngắn (86% vs 82%)** nhưng không deploy được; bản deploy được thì kém Google 20 điểm.

### Giả thuyết ĐÃ THỬ VÀ BỊ BÁC — đừng lặp lại

**"i2_s bỏ hết các số 0"** — BÁC bằng thực nghiệm. Lý lẽ ban đầu: tìm thấy trong
`ggml-quants.c` đoạn `i2_scale = max` (absmax) + `if (fabs(src[i]) < 1e-6) q8[i]=1` —
tức chỉ số ~0 tuyệt đối mới thành 0, trong khi `weight_quant` của dự án cho **34,6%**
weight thành 0.

Kiểm: bắt PyTorch dùng đúng công thức đó (`sign(w) * mean|w|`, không có số 0 nào),
bench 200 câu, chấm cùng phiên với bản GGUF:

| | TỔNG | ngắn | dài |
|---|---:|---:|---:|
| v6 PyTorch (gốc) | 80% | 83% | 77% |
| v6 GGUF i2_s | **68%** | 70% | 65% |
| **sign-only (bỏ số 0)** | **13%** | 25% | **1%** |

`sign-only vs GGUF`: 8 hơn / 117 kém, p=0,000. Nếu i2_s bỏ số 0 thì nó phải ~13%, không
phải 68%. ⇒ **i2_s CÓ giữ số 0 đúng cách.** Sai lầm của tôi: đoạn code đó **đang bị
comment** (`//`) mà tôi vẫn suy diễn từ nó. Nguồn: `eval/judge_zero/RESULT.json`.

Phụ phẩm có giá trị: **bỏ số 0 làm model sụp hoàn toàn ở câu dài (1%)** — xác nhận
34,6% zero KHÔNG phải dư thừa, nó là phần thiết yếu của năng lực model.

### Giả thuyết 2 CŨNG BỊ BÁC: lệch lượng tử hoá ACTIVATION

Lý lẽ: model train W1.58A8 (`activation_quant` = 8-bit per-token absmax mỗi BitLinear).
Nghi llama.cpp không tái tạo bước đó -> lệch train/infer.

Kiểm: patch `bitnet._ste_act` thành identity (tắt hẳn activation quant), bench 200 câu:

| | TỔNG | ngắn | dài |
|---|---:|---:|---:|
| PyTorch gốc (CÓ act quant) | 79% | 83% | 75% |
| **PyTorch TẮT act quant** | **80%** | 83% | 76% |
| i2_s | 62% | 68% | 57% |

**Tắt hoàn toàn cũng không đổi gì** (80 vs 79 = nhiễu) ⇒ activation quant KHÔNG phải
nguyên nhân. Nguồn: `eval/judge_act/RESULT.json`.

Phụ phẩm: **activation quant 8-bit gần như không tốn chất lượng** — bỏ được nếu cần tốc độ.

### Cũng đã loại: LƯỢNG TỬ HOÁ WEIGHT nói chung

Q8_0 (8 bit) = **54%**, i2_s (2 bit) = **54%**, Q4_0 = 52%, PyTorch = 76%
(cùng phiên `judge_quant`). Q8_0 lẽ ra gần như không mất gì. ⇒ Không phải do độ chính xác
weight. Và `convert_to_gguf.py:131` đã tự ternary hoá trước khi ghi nên số 0 được giữ đúng.

### ✅ ĐÃ TÁCH ĐƯỢC NHÁNH: lỗi ở **RUNTIME bitnet.cpp**, KHÔNG phải ở convert

Nạp trọng số **từ GGUF ngược lại PyTorch** (gán thẳng vào `_wq_frozen`, không re-quantize
— quan trọng, vì trọng số GGUF đã ternary `{-m,0,+m}` mà re-quantize bằng absmean sẽ đổi
biên độ m → 0,654m), rồi bench 200 câu cùng phiên:

| | TỔNG | ngắn | dài |
|---|---:|---:|---:|
| PyTorch, trọng số từ **checkpoint** | 80% | 85% | 75% |
| PyTorch, trọng số từ **GGUF** | **80%** | **85%** | 74% |
| **bitnet.cpp runtime (i2_s)** | **64%** | 68% | 60% |

| McNemar | kết quả |
|---|---|
| GGUF-weights vs checkpoint | 1 vs 2 câu → **p=1,000, KHÔNG khác nhau** |
| GGUF-weights vs bitnet.cpp | 40 vs 9 câu → **p=0,000** |

Kèm bằng chứng phụ: tỉ lệ số 0 trong trọng số GGUF = **34,59%**, khớp đúng 34,6% của
`weight_quant`. Mẫu ternary được bảo toàn chính xác.

⇒ **`convert_to_gguf.py` ĐÚNG, đừng sửa nó.** 16 điểm mất phát sinh khi bitnet.cpp *tính*,
không phải khi ta *ghi*. Nguồn: `eval/judge_bisect/RESULT.json`.

### Việc cần làm — đã khu trú vào code bên thứ ba

1. Đọc `include/bitnet-lut-kernels.h` + `src/ggml-bitnet-mad.cpp` — kernel LUT thật đang chạy
2. Thử build lại bitnet.cpp **bản mới hơn** (bản hiện tại `llama-cli` còn chưa có cờ `-no-cnv`,
   tức đã cũ) — có thể lỗi đã được sửa upstream
3. Thử tắt kernel LUT: build với `GGML_BITNET_X86_TL2=OFF` để dùng đường matmul thường,
   xem có về 80% không. Đây là phép thử rẻ nhất và tách được kernel khỏi phần còn lại
4. Chạy so logits từng lớp giữa hai runtime trên **cùng một câu** để tìm lớp phân kỳ đầu tiên

**Không cần** sửa gì ở phía dự án: checkpoint đúng, convert đúng, ternary đúng, activation
quant không liên quan.

**Phép bisect quyết định — script ĐÃ CÓ SẴN**: `scripts/gguf_pytorch_check.py` nạp trọng số
từ GGUF **ngược lại PyTorch** rồi chạy generate. Docstring ghi đúng mục đích:
*"phân định lỗi convert/runtime (llama.cpp) vs trọng số thật (train)"*.
- Nếu GGUF→PyTorch đạt ~79% ⇒ convert ĐÚNG, lỗi ở **runtime llama.cpp**
- Nếu chỉ ~62% ⇒ lỗi ở **bước convert**

Chạy: `python scripts/gguf_pytorch_check.py D:/Bit-Translate-data/dist/v6_avg_f16.gguf`

Sau đó tuỳ nhánh: nghi `attn_sub_norm`/`ffn_sub_norm` (norm phụ riêng của BitNet) bị đặt
sai chỗ, hoặc `output_norm`/`lm_head`, hoặc cách áp RoPE.

1. Thử `Q8_0` thay `I2_S` — nếu Q8_0 giữ được ~80% thì lỗi khu trú hẳn ở kernel i2_s
2. So logits PyTorch vs GGUF trên **cùng một câu, từng lớp**, tìm lớp đầu tiên phân kỳ
3. Đọc `include/bitnet-lut-kernels.h` và `src/ggml-bitnet-mad.cpp` — đó là kernel THẬT
   đang chạy, không phải đoạn reference bị comment trong `ggml-quants.c`
4. Kiểm lượng tử hoá **activation**: dự án dùng per-token absmax 8-bit
   (`activation_quant`), kernel i2_s có thể dùng công thức khác

### ⚠️ Thang judge TRÔI giữa các phiên — đọc mọi số ở trên có kèm điều này

Cùng bộ 200 câu, cùng judge `gemini-flash-latest`, ba phiên khác nhau:

| phiên | Google | v6 PyTorch | v6 GGUF |
|---|---:|---:|---:|
| judge_v6 | 82% | 72% | – |
| judge_final | 86% | 74% | 56% |
| judge_zero | – | **80%** | **68%** |

Biên độ trôi tới **8 điểm**. Nên **chỉ so các hệ TRONG CÙNG một phiên**, tuyệt đối không
so số giữa hai phiên. Khoảng cách PyTorch–GGUF thì bền: 18 điểm ở `judge_final`,
12 điểm ở `judge_zero`, **cả hai đều p=0,000**.

### File để tái lập

```
eval/bench_v6.jsonl            <- PyTorch avg7, 74%
eval/bench_v6avgdeploy.jsonl   <- GGUF i2_s avg7, 56%
eval/bench_v6deploy.jsonl      <- GGUF i2_s step16400, 54%
eval/judge_final/RESULT.json   <- diem judge 4 he cung phien
D:/Bit-Translate-data/dist/v6_avg_i2s.gguf        77,56 MB
D:/Bit-Translate-data/dist/v6_s16400_i2s.gguf     77,56 MB
```

---

## #5 — Cụm kính ngữ thư tín cố định + nghĩa phụ của động từ thường (MỞ)

**Phát hiện 2026-07-30, từ bench 200b chấm mù 4 hệ (`eval/judge_200b_claude/KETQUA.md`).**

### Số đo — evidence từng câu

Hai lớp lỗi, đều là **class B (nghĩa theo ngữ cảnh)** — không phải thiếu thuật ngữ
(khớp `loi-that-la-cau-truc-khong-phai-thuat-ngu`: thuật ngữ chỉ 3% lỗi):

**(a) Nghĩa phụ của động từ thường — CẢ 4 HỆ (v7a/Google/gate/v6) cùng fail:**

| id | câu | bẫy |
|---|---|---|
| **id67** | つい子どもに当たってしまう | 当たる = "trút giận lên con" — cả 4 dịch theo nghĩa đen "trúng/đánh" |
| **id169** | 情報が漏れかけた | 〜かけた = "SUÝT lộ" — cả 4 dịch thành "đã lộ", mất thể chưa-hoàn-thành |

**(b) Cụm kính ngữ thư tín thương mại cố định — giết nhiều hệ cùng lúc:**

| id | cụm | ai đúng |
|---|---|---|
| **id187** | 心ばかりの品ですがお納めください | **cả 4 hệ fail** (dịch thành "thanh toán/nộp/tặng cho tôi") |
| id182 | ご査収 | chỉ Google đúng |
| id184 | 取り急ぎご報告まで | chỉ v7a đúng |
| id189 | ご清栄 | chỉ Google tạm được |

Domain liên quan: `formal_letter` use chỉ 80% (v6 từng 90%), `art_ent` 80% (Google 95%).

### Vì sao đây là bẫy chung

3 câu (id67/169/187) cả 4 hệ — kể cả Google — cùng sai, tức không phải lỗi riêng của
model nhỏ: đây là những mục **tần suất thấp + mặt chữ đánh lừa** (động từ phổ thông mang
nghĩa phụ, quán ngữ thư tín chỉ xuất hiện trong business letter). Corpus KD hiện đào theo
khuôn mẫu/miền, chưa từng nhắm danh sách biểu thức thư tín hay cặp nghĩa đa nghĩa.

### Hướng fix (chưa làm)

1. **KD data nhắm cụm cố định**: lập danh sách biểu thức thư tín thương mại (ご査収/
   お納めください/ご清栄/取り急ぎ/ご笑納/ご自愛… — danh sách công khai, lớp từ ĐÓNG,
   vài trăm mục) → mine câu chứa trong CC-100 → KD Live API. Đúng công thức lexicon
   đã thắng ở chiến dịch katakana (PLAN_V7A §2).
2. **Cặp câu tối thiểu phân biệt nghĩa động từ đa nghĩa**: với mỗi động từ phổ thông đa
   nghĩa (当たる, 納める, 漏れる±かける, 押される…), sinh cặp câu chỉ khác ngữ cảnh —
   mỗi nghĩa một câu — để model học phân nhánh theo ngữ cảnh thay vì nghĩa đen tần suất cao.

### Cách kiểm lại

Dịch lại các id 67/169/182/184/187/189 trong `eval/bench_opus200b.jsonl` + bench mini
riêng cho danh sách biểu thức thư tín (chưa dựng); so trước/sau bằng chấm mù cùng phiên.
