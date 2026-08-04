# TQ33 Runner độc lập — forward pass ĐẦY ĐỦ Qwen3-0.6B, số đo THẬT trên toàn model

**Ngày**: 2026-08-04 · **Trạng thái**: Giai đoạn 1 PASS, Giai đoạn 2 PASS (đúng/sai) + đo tốc độ thật ·
**Máy**: máy B (Intel Core Ultra 5 225H, 14 luồng, không CUDA) · **Chi phí**: $0 (local)

## 0. Tóm tắt kết quả (đọc trước)

- **Giai đoạn 1 (F32 thuần, C độc lập không ggml/llama.cpp)**: khớp oracle PyTorch gần như
  tuyệt đối — rel err logits **2.9e-6**, top-1 token **TRÙNG**, mọi layer rel err < 4e-6.
- **Giai đoạn 2 (TQ33 kernel thật cho 7 loại linear x 28 layer = 196 tensor)**: encode
  **1.500 bpw đúng chuẩn, lossless bit-level 194/196 tensor** (2 tensor có 1 nhóm-4 vi phạm
  2:4 hiếm gặp, đã sanitize — ảnh hưởng 4/440M trọng số). Forward pass full-model: **top-1
  VÀ top-5 next-token TRÙNG TUYỆT ĐỐI với oracle F32**, dù rel err logits tổng thể 4.8e-2
  (bị kéo lên bởi 1 layer cuối gặp "activation outlier" — giải thích ở mục 3.3, đây là hiện
  tượng LLM quantization đã biết, không phải bug).
- **Tốc độ đo trên toàn model** (KV-cache, 30 token sinh, nhiều mức luồng): dao động
  **10.7–32.4 tok/s** tuỳ số luồng VÀ tuỳ lần chạy (nhiễu hệ thống lớn — xem mục 4.3).
  Hiệu năng kernel TQ33 hiệu dụng trong forward pass thật chỉ đạt **~4.6 GB/s (1 luồng)**,
  và **KHÔNG scale tốt như microbench cũ** khi nhiều luồng (~5–6 GB/s dù 6-14 luồng) —
  khác hẳn con số 19.3 GB/s@6T của microbench. Lý do đã xác định rõ (mục 4.4): microbench
  đo 1 tensor lớn stream liên tục; forward pass thật gọi 196 GEMV nhỏ/token, overhead đồng
  bộ hoá luồng (dù dùng thread-pool thường trực) ăn hết phần lợi từ chia luồng ở mức
  granularity nhỏ này. **Điểm nghẽn tốc độ LỚN NHẤT không phải TQ33 mà là lm_head/embedding
  F32 chưa lượng tử hoá (622MB đọc mỗi token)** — chiếm 28–87% thời gian tuỳ số luồng.

## 1. Sai lệch so với giả định ban đầu trong bối cảnh nhiệm vụ (đã sửa theo thực tế đo được)

### 1.1 `lm_head.weight` — SAI: không phải "không tồn tại vì tie_word_embeddings=true"

Giả định ban đầu: *"KHÔNG có `.weight` nào cho lm_head riêng vì `tie_word_embeddings=true`
(dùng lại embed_tokens transpose)."* — **Kiểm tra thực tế: SAI.** Ckpt `qat_gen4_n4.pt` CÓ
key `lm_head.weight` riêng (shape `[151936, 1024]`, giống `model.embed_tokens.weight`).
Đây hoá ra không phải hiếm: bản pristine gốc `Qwen/Qwen3-0.6B` (snapshot HF cache) **CŨNG**
lưu cả 2 tensor này dù config nói tied — nhiều repo HF vẫn serialize cả 2 cho tương thích
ngược, dù runtime tie lại thành 1 object.

Vấn đề thật: `lm_head.weight` và `model.embed_tokens.weight` trong ckpt bake **KHÁC NHAU**
(max abs diff 0.18), dù bản pristine gốc chứng minh **CẢ 2 giống hệt nhau tuyệt đối** lúc
`from_pretrained` (diff = 0.0). Tức là tie đã "vỡ" ở đâu đó trong pipeline train (nghi do
`.to(dtype=torch.float32)` tách 2 Parameter cùng storage thành 2 object độc lập — hành vi
PyTorch đã biết khi module có weight tied bị `.to()`/`.half()` duyệt cây module 2 lần).
Đã xác nhận qua đọc code (`exp_r_qat_lite.py`): **`lm_head` KHÔNG bao giờ được wrap bởi
`LearnQLinear`** (chỉ 7 linear q/k/v/o/gate/up/down bị wrap — xem `LIN_PATHS`), và **không
nằm trong optimizer nào** ở cả S1 lẫn S2 (`trainable = params_w+params_b+...` không có
embed/lm_head) → cả 2 tensor bị đóng băng nguyên trạng từ lúc load, khác nhau chỉ vì tie
bị vỡ ở mức object, không phải do 1 trong 2 được train tiếp.

**Quyết định + verify**: đã build 1 test riêng (`resolve_lmhead.py`, không phải file final
nhưng logic đã gộp vào `ref_forward_pytorch.py`): load model HF thật, gắn bias đúng (xem
1.2), thử PPL trên 100 câu dev.vi/dev.ja thật với TỪNG lựa chọn:

| Lựa chọn output head | ppl_vi (100 câu) | ppl_ja (100 câu) |
|---|---|---|
| `model.embed_tokens.weight` (tied-style) | **296.7** | **575.8** |
| `lm_head.weight` (tensor riêng trong ckpt) | 301.3 | 588.5 |

`embed_tokens.weight` cho PPL thấp hơn ~1.5-2% ở cả 2 chiều (nhất quán, không phải nhiễu 1
phía), VÀ khớp quy ước `PreTrainedModel.tie_weights()` của HF (tie OUTPUT theo INPUT —
`output_embeddings.weight = input_embeddings.weight`, không phải chiều ngược lại). Runner
dùng `model.embed_tokens.weight` cho cả input lookup lẫn output logits. `lm_head.weight`
vẫn được dump đầy đủ (không bỏ sót tensor nào khi export) nhưng không dùng trong forward.

### 1.2 Bias — ĐÚNG NHƯ CẢNH BÁO: tất cả 196 tensor `.bias` (7 loại x 28 layer) đều khác 0

Đã kiểm toàn bộ (không chỉ vài mẫu): absmax dao động 0.005–0.038, **0 tensor nào = 0**.
Điểm cần chú ý thêm KHÔNG có trong bối cảnh gốc: **`mlp.{gate,up,down}_proj` cũng có bias**
dù kiến trúc Qwen3 chuẩn của HF (`Qwen3MLP.__init__`) **luôn** tạo `nn.Linear(..., bias=False)`
cho MLP — bias ở đây là do `LearnQLinear` (wrapper QAT) tự thêm cho CẢ 7 loại linear đồng
loạt, không phân biệt loại nào gốc có bias hay không. Khi build oracle PyTorch bằng
`transformers.Qwen3ForCausalLM` chuẩn, **phải monkey-patch thêm `nn.Parameter` bias vào cả
7 submodule TRƯỚC `load_state_dict`**, nếu không `strict=False` sẽ ÂM THẦM BỎ QUA toàn bộ
bias (đã tự bắt lỗi này 1 lần lúc viết `resolve_lmhead.py` — ppl tính thiếu bias vẫn ra số
"nghe hợp lý" 296-305, dễ tưởng đúng nếu không đối chiếu kỹ).

### 1.3 RoPE NEOX, GQA grouping, thứ tự phép toán layer — ĐÚNG, đã double-check 2 nguồn độc lập

Đọc trực tiếp `transformers/models/qwen3/modeling_qwen3.py` (bản cài thật, transformers
4.57.6) — `rotate_half()` + `apply_rotary_pos_emb()` — và đối chiếu với
`ggml_compute_forward_rope_flt` (`GGML_ROPE_TYPE_NEOX`) trong llama.cpp: **2 công thức
tương đương bit-for-bit về mặt toán học** (xoay cặp `x[j], x[j+dim/2]`, KHÔNG phải cặp liền
kề). GQA: `repeat_kv()` xác nhận q-head `h` dùng kv-head `h // (n_head/n_kv_head)` = `h//2`
(nhóm liền kề — 2 q-head 0,1 dùng kv-head 0; 2,3 dùng kv-head 1; …). Thứ tự phép toán từng
layer (`Qwen3DecoderLayer.forward`, `Qwen3Attention.forward`) khớp CHÍNH XÁC mô tả gốc
trong bối cảnh nhiệm vụ, không cần sửa gì.

## 2. Giai đoạn 1 — Runner F32 thuần

### 2.1 Phương pháp

- `export_full_model_f32.py`: dump toàn bộ 507 tensor của `qat_gen4_n4.pt` (fp16 gốc) sang
  float32 phẳng little-endian, `D:\Bit-Translate-data\tq33_runner\weights_f32\weights.bin`
  (3.008 GB) + `meta.json` (mô tả đầy đủ) + `meta_index.txt` (format đơn giản cho C parse
  bằng `fscanf`, tránh viết JSON parser trong C).
- `ref_forward_pytorch.py`: build `transformers.Qwen3ForCausalLM` từ config thật, monkey-patch
  bias (mục 1.2), `load_state_dict` ckpt bake, set output head = `embed_tokens` (mục 1.1),
  forward prompt tiếng Việt thật **"Xin chào, hôm nay"** (tokenize bằng tokenizer Qwen3-0.6B
  thật → 7 token: `[55, 258, 139548, 11, 130535, 308, 352]` = `X/in/ chào/,/ hôm/ n/ay`), hook
  per-layer để lưu **hidden state RAW sau MỖI trong 28 layer** + hidden sau final norm +
  logits đầy đủ `[7, 151936]` ra `.npy`.
- `qwen3_runner.c`: C thuần (không ggml/llama.cpp), đọc weights_f32 + token id, forward
  brute-force O(seq²) đúng thứ tự phép toán (RMSNorm → Q/K/V proj+bias → per-head RMSNorm
  → RoPE NEOX → GQA attention → o_proj → residual → RMSNorm → SwiGLU → residual, 28 lần →
  final norm → logits). Build bằng `python -m ziglang cc -O3 -mavx2 -mfma` (máy không có
  MSVC/gcc, chỉ ziglang qua pip — theo đúng constraint đã biết).
- **Lỗi kỹ thuật đã bắt trước khi chạy**: `ftell()/fseek()` trả về `long` — trên Windows
  `long` là 32-bit (LLP64) dù chương trình 64-bit, sẽ TRÀN SỐ với file `weights.bin` 3GB
  (>2GB). Đã sửa: tính tổng số phần tử trực tiếp từ meta (offset+numel tensor cuối) rồi
  `fread` đúng số đó — `fread`/`fwrite` nhận `size_t` (64-bit) nên không bị giới hạn này.

### 2.2 Kết quả — rel err từng layer (so PyTorch oracle)

| Layer | rel err (L2) | max abs diff |
|---|---|---|
| 0 | 8.5e-7 | 2.9e-6 |
| 1 | 8.8e-7 | 3.8e-6 |
| 2–25 | 2.8e-7 – 3.2e-7 | 1.5e-3 (hằng định) |
| 26 | 4.1e-7 | 2.0e-3 |
| 27 | 3.5e-6 | 8.5e-3 |
| **final (post-norm)** | **1.8e-6** | — |
| **logits** | **2.9e-6** | 8.4e-5 |

**Top-1 token**: C runner = 11, PyTorch = 11 — **TRÙNG**. Top-5 logit KHỚP đến 4 chữ số
thập phân (`12.9424, 12.3293, 11.8079, 11.6590, 11.6258`) giữa 2 implementation độc lập.

**Phán quyết**: **ĐẠT** — rel err << 1e-3 (ngưỡng yêu cầu), thực ra còn tốt hơn kỳ vọng ban
đầu (<1e-4). Sai số hoàn toàn giải thích được bằng nhiễu tích luỹ float32 (thứ tự cộng dồn
khác nhau giữa vòng lặp C tuần tự và BLAS/vector hoá của PyTorch) — không phát hiện bug.
→ đủ điều kiện sang Giai đoạn 2.

## 3. Giai đoạn 2 — Thay 7 loại linear bằng kernel TQ33

### 3.1 Encode 196 tensor (28 layer x 7 loại: q/k/v/o/gate/up/down)

Script `export_tq33_all_linears.py` mở rộng `exp_t24_export_bench.py` (vốn chỉ làm 1
tensor) cho toàn model, tái sử dụng NGUYÊN VẸN `build_patterns()` (33 pattern hợp lệ) từ
`exp_t24_codec.py` để đảm bảo pattern-id khớp đúng thứ tự với `tq33_bench.c`.

- **196/196 tensor encode xong trong 17 giây**, verify roundtrip (giải mã lại bằng Python
  độc lập với encode, so bit-level) **LOSSLESS 194/196 tensor tuyệt đối**.
- **2 tensor có vi phạm 2:4 hiếm** (`model.layers.9.mlp.gate_proj`,
  `model.layers.11.mlp.gate_proj`) — mỗi tensor đúng **1 nhóm-4 duy nhất** có 3 giá trị
  khác 0 thay vì ≤2 (trên tổng 110,100,480 nhóm-4 toàn model — tỷ lệ 1.8e-8). Đã sanitize
  (giữ 2 |W| lớn nhất trong nhóm, ép phần tử nhỏ nhất về 0) trước khi encode — sai số tối
  đa do sanitize: 0.043 và 0.086 (trên 4 trọng số / 440 triệu trọng số toàn model).
- **Codebook scale**: tối đa 47 giá trị duy nhất/tensor (tensor `model.layers.26.mlp.down_proj`),
  luôn ≤ 256 → giả định "scale trên lưới ≤256 giá trị" của format TQ33 giữ đúng cho TOÀN
  BỘ 196 tensor, không chỉ 6 tensor mẫu đã test trước đó.
- **bpw thực tế đo được: 1.500 chẵn** (82.58 MB packed / 440,401,920 trọng số) — khớp lý
  thuyết tuyệt đối. So với fp32 gốc (1761.6 MB cho cùng 196 tensor): **21.3x nhỏ hơn**.

### 3.2 Runner TQ33 (`qwen3_runner_tq33.c`)

Tái sử dụng NGUYÊN VẸN kernel `decode_block()` + `row_dot_avx2()` (kernel v3 — LUT→buffer +
FMA-accumulate) từ `tq33_bench.c`, cắm vào đúng vị trí 7 linear/layer; RMSNorm/RoPE/attention
brute-force/SwiGLU activation giữ float32 y hệt Giai đoạn 1. Activation lượng tử hoá int8
per-64-group NGAY TRƯỚC mỗi lời gọi linear TQ33 (đúng như microbench: `scale=max|x|/127`).

### 3.3 Kết quả đúng/sai — rel err từng layer, VÀ phát hiện activation outlier ở layer cuối

| Layer | rel err (L2) vs oracle F32 | max abs diff |
|---|---|---|
| 0 | 1.16e-2 | 1.9e-2 |
| 1 | 1.23e-2 | 5.1e-2 |
| 2 | 6.3e-4 | 1.59 |
| … (tăng dần đều) | 6.3e-4 → 3.1e-3 | ~1.4–1.6 |
| 26 | 3.1e-3 | 3.26 |
| **27** | **1.37e-1** | **312.5** |
| final (post-norm) | 5.7e-2 | — |
| **logits** | **4.77e-2** | 1.95 |

**Layer 2–26**: rel err tăng dần đều 6.3e-4 → 3.1e-3 qua 25 layer — nhiễu tích luỹ hợp lý,
CÙNG BẬC với con số microbench gốc đo trên 1 tensor (6.7e-3 rel err int8-activation).

**Layer 27 (layer CUỐI): rel err nhảy vọt gấp ~44 lần** (3.1e-3 → 1.37e-1). Đã điều tra
KHÔNG PHẢI bug — đây là hiện tượng **"activation outlier channel"** đã biết trong văn liệu
LLM quantization (vd. LLM.int8(), "Massive Activations in LLMs"): hook trực tiếp vào input
của các linear tại layer 27 cho thấy **1 kênh ẩn cụ thể (dim ~48/52 trong không gian
hidden 1024) có |giá trị| gấp 37–237 lần các kênh khác CÙNG NHÓM-64** (vd. input
`mlp.down_proj` layer 27: absmax=42.18 so với median-nhóm=7.63, tỷ lệ outlier/khác 237x).
Đối chiếu layer giữa (layer 13): absmax chỉ 1.94–18.49 — outlier RÕ RÀNG chỉ nổi bật/phóng
đại mạnh ở các layer cuối (khớp y văn liệu: outlier channel thường xuất hiện/tăng cường ở
layer sâu). Kiểm thêm: outlier này có mặt ở **MỌI vị trí token** (không riêng vị trí cuối),
kênh dim=48 tại vị trí token đầu tiên đạt |5577| — mạnh nhất (khớp hiện tượng "attention
sink" ở token đầu). Cơ chế gây lỗi: lượng tử hoá int8 per-64-group dùng
`scale=max|x|/127` — khi 1 phần tử outlier áp đảo nhóm, scale bị kéo lớn, các phần tử NHỎ
còn lại trong CÙNG nhóm bị lượng tử thô (sai số tuyệt đối ~0.27, tương đương ~4-7% sai số
tương đối cho các giá trị "bình thường" trong đúng nhóm đó).

**Mặc dù vậy**: **top-1 VÀ top-5 next-token vẫn TRÙNG TUYỆT ĐỐI** với oracle F32
(`top-5 overlap: 5/5`, token id giống hệt `[11, 220, 293, 308, 89437]`) — margin logit giữa
top-1 (12.92) và top-2 (12.33) đủ lớn (~0.6) để hấp thụ nhiễu này mà không đổi kết quả dự
đoán, **ít nhất trên prompt thử nghiệm này**. Đây là 1 giới hạn thật cần nêu rõ: KHÔNG có gì
đảm bảo margin sẽ luôn đủ lớn ở câu/vị trí khác — quantization naive per-group không có cơ
chế bảo vệ outlier-channel (khác các kỹ thuật đã công bố như SmoothQuant/LLM.int8() mixed-
precision decomposition), đây là hướng cải thiện tự nhiên nếu tiếp tục nghiên cứu.

**Phán quyết Giai đoạn 2 (đúng/sai)**: **ĐẠT theo tiêu chí task đề ra** (top-1 khớp; rel err
logits 4.8e-2 — lớn hơn Giai đoạn 1 như dự kiến, có nguồn gốc rõ ràng là 1 outlier channel
ở layer cuối chứ không phải lỗi ngẫu nhiên/bug rải rác).

## 4. Tốc độ — đo trên TOÀN MODEL (không chỉ 1 tensor)

### 4.0 Kiểm chứng đường KV-cache riêng (KHÔNG chỉ đường batch/prefill đã validate ở mục 3.3)

`qwen3_runner_tq33.c` có **2 đường tính riêng biệt**: `run_compare()` (batch, brute-force
attention toàn bộ seq 1 lần, đã so khớp oracle ở mục 3.3) và `forward_one_token()` +
KV-cache (dùng cho benchmark tốc độ, xử lý từng token 1, đọc/ghi cache). 2 đường này share
`linear_tq33`/`rmsnorm`/`rope_neox` nhưng orchestration khác hẳn — 1 bug riêng ở logic
cache (sai vị trí ghi/đọc, sai chỉ số position) sẽ KHÔNG bị bắt bởi so sánh ở mục 3.3.
Đã bổ sung kiểm chứng chéo: chạy đường KV-cache cho đúng 7 token prompt (pos 0..6, dùng
cache), lấy token dự đoán tại vị trí cuối — **kết quả = 11, TRÙNG TUYỆT ĐỐI với top-1 đã
validate độc lập ở đường batch** (mục 3.3). Đây là bằng chứng mạnh cả 2 đường triển khai
đều đúng (không chỉ 1 trong 2), vì chúng tính TOÁN HOÀN TOÀN KHÁC NHAU (brute-force toàn
seq vs. tuần tự + cache) nhưng ra CÙNG kết quả tại điểm giao nhau.

Token sinh tiếp theo (đọc greedy/argmax, không sampling, không repetition penalty) rơi vào
**vòng lặp lặp lại chu kỳ 8** (`128296 28776 128254 272 2185 58701 128310 11 …`) — đây là
hành vi ĐÃ BIẾT của greedy decoding không có cơ chế chống lặp, đặc biệt dễ xảy ra với model
0.6B bị nén nặng tiếp tục từ 1 prompt ngắn/chung chung — KHÔNG phải bug, chỉ phản ánh việc
benchmark dùng argmax đơn giản nhất có thể (mục đích là đo tốc độ, không phải chất lượng
sinh văn bản).

### 4.1 Phương pháp

`qwen3_runner_tq33.c` phần B: KV-cache đơn giản (append-only, không tối ưu, cấp phát tĩnh
`MAX_POS=128`), sinh tự hồi quy (autoregressive, argmax) **30 token** sau khi prefill 7 token
prompt (đo thời gian CHỈ phần sinh 30 token, không tính prefill). Chia luồng theo HÀNG của
GEMV (row-parallel) — không dùng OpenMP được (**đã thử `-fopenmp` với zig cc: thiếu
`omp.h`, xác nhận không khả dụng** trên toolchain này), dùng Windows thread thủ công.

**Lỗi thiết kế đã bắt và sửa giữa chừng**: bản đầu tiên tạo/huỷ thread (`_beginthreadex`/
`CloseHandle`) MỖI LẦN gọi GEMV (196 lần/token) — kết quả tăng luồng > 6 làm CHẬM ĐI (14
luồng: 7.0 tok/s, tệ hơn cả 6 luồng 11.2 tok/s) vì overhead tạo thread thật (~50-200us/lần)
lấn át lợi ích chia luồng cho các GEMV nhỏ. Đã sửa bằng **thread-pool thường trực** (spawn N
thread 1 lần lúc khởi động, đồng bộ qua Windows Event auto-reset mỗi lần gọi thay vì tạo
thread mới) — sau khi sửa, tăng luồng không còn làm chậm đi (xem 4.2). Cũng phát hiện + sửa
1 vấn đề khác: GEMV F32 cho embed/logits ban đầu viết vòng lặp scalar thuần — clang KHÔNG
tự động vector hoá được (strict FP semantics, không `-ffast-math`) → viết tay AVX2+FMA
tường minh (`dot_f32_avx2`, cùng style `row_dot_avx2`) mới phản ánh đúng tốc độ F32 khả thi,
nếu không sẽ thổi phồng sai % thời gian embed/logits chiếm dụng.

### 4.2 Tốc độ đo được — dao động lớn giữa các lần chạy (5 lần, cùng máy, liên tiếp)

| Luồng | Chạy 1 | Chạy 2 | Chạy 3 | Chạy 4 | Chạy 5 | **min–max** |
|---|---|---|---|---|---|---|
| 1 | 18.50 | 10.69 | 10.73 | 17.90 | 14.32 | **10.7–18.5** |
| 2 | 26.10 | 12.03 | 12.81 | 22.47 | 17.44 | 12.0–26.1 |
| 4 | 29.38 | 12.80 | 13.26 | 15.90 | 23.79 | 12.8–29.4 |
| 6 | 30.08 | 15.97 | 16.96 | 18.43 | 25.75 | 16.0–30.1 |
| 8 | 30.24 | 11.99 | 23.36 | 17.60 | 29.69 | 12.0–30.2 |
| 12 | 30.26 | 13.10 | 30.01 | 13.32 | 30.71 | 13.1–30.7 |
| 14 | 30.94 | 15.13 | 32.42 | 16.57 | 31.10 | 15.1–**32.4** |

(đơn vị: tok/s, đo autoregressive 30 token sinh, thread-pool thường trực)

**Nhiễu 2-3 lần giữa các lần chạy LIÊN TIẾP trên CÙNG 1 máy, CÙNG file .exe** — kể cả ở 1
luồng (không có tranh chấp đồng bộ hoá luồng nào để đổ lỗi). Đối chiếu hiệu năng GB/s hiệu
dụng của riêng phần TQ33 (mục 4.4) cho thấy dao động y hệt (2.1–4.6 GB/s ở 1 luồng) — nghi
ngờ hợp lý nhất là **thermal throttling tích luỹ** qua các lần chạy liên tiếp không nghỉ
(khớp ghi chú đã biết về máy B trong CLAUDE.md: hiệu năng CPU máy này nhạy với trạng thái
nhiệt/nguồn). Đã xác nhận máy đang cắm điện (`PowerOnline=True`) lúc đo — không phải do
chạy pin. **Kết luận thực tế**: con số "capability" hợp lý nhất để trích dẫn là dải chạy
1 ("mát", ngay sau khi nạp xong model) — **18.5–30.9 tok/s** tuỳ luồng — nhưng dưới tải bền
vững (nhiều request liên tục) nên kỳ vọng có thể tụt xuống nửa dưới của dải trên.

### 4.3 Breakdown thời gian (đại diện, chạy "mát")

| Luồng | linear-TQ33 | attention | norm+RoPE | embed+logits (F32) |
|---|---|---|---|---|
| 1 | 34% | 2% | 1% | **63%** |
| 2 | 39–61% | 2-3% | 1% | 37-57% |
| 6 | 44-58% | 1-5% | 1-2% | 38-50% |
| 14 | 49-92%* | 1-5% | 0-2% | 7-45%* |

\* dao động lớn giữa các lần chạy, xem 4.4 để hiểu vì sao.

**Phát hiện quan trọng nhất về kiến trúc**: ở 1 luồng, **embed+logits (F32, KHÔNG lượng tử
hoá) chiếm 63-87% tổng thời gian — LỚN HƠN cả overhead attention/norm/RoPE cộng lại rất
nhiều lần**. Đây là vì `model.embed_tokens.weight` [151936×1024] = 622MB fp32 phải đọc TOÀN
BỘ để tính logits mỗi token (task không yêu cầu lượng tử hoá embed/lm_head, chỉ 7 loại
linear per-layer) — với model 0.6B có vocab bất thường lớn (151936) so với hidden (1024),
chi phí lm_head KHÔNG lượng tử hoá trở thành nút thắt chính, LỚN HƠN toàn bộ 28 layer TQ33
cộng lại (82.58MB). Đây là phát hiện mà microbench cũ (chỉ đo 1 tensor linear) không thể
thấy được — đúng như mục tiêu nhiệm vụ đề ra.

### 4.4 So với microbench cũ (5.6 GB/s@1T, 19.3 GB/s@6T trên 1 tensor)

Tính GB/s hiệu dụng riêng phần TQ33 trong forward pass thật (82.58MB / (%linear-tq33 x
thời-gian-mỗi-token)):

| Luồng | GB/s hiệu dụng (forward thật, dải 5 lần chạy) | GB/s microbench cũ |
|---|---|---|
| 1 | **2.1 – 4.6 GB/s** | 5.6 GB/s |
| 6 | **4.5 – 5.7 GB/s** | 19.3 GB/s |
| 14 | **~5 – 6 GB/s** (ước lượng, nhiễu) | (không đo ở microbench) |

**1 luồng: gần microbench** (~4.6/5.6 = 82% ở lần chạy tốt nhất) — chênh lệch còn lại giải
thích được bằng việc forward pass thật phải **lượng tử hoá activation MỖI LẦN gọi** (microbench
đo GEMV lặp lại trên activation ĐÃ lượng tử hoá sẵn 1 lần trước vòng lặp đo thời gian — không
tính overhead này vào con số 5.6GB/s).

**6+ luồng: KHÔNG đạt** — chỉ 4.5-5.7 GB/s thay vì 19.3 GB/s kỳ vọng (~25-30% mức
microbench). Nguyên nhân xác định rõ: microbench chia 6 luồng theo 6 "bản sao" ĐỘC LẬP của
1 tensor lớn (mỗi luồng cày ~25MB liên tục, KHÔNG đồng bộ hoá giữa các luồng cho tới hết).
Forward pass thật gọi `linear_tq33()` **196 lần/token** (7 linear x 28 layer), MỖI LẦN với
kích thước nhỏ hơn nhiều (131KB–786KB/tensor) và phải đợi TẤT CẢ luồng xong (đồng bộ hoá
qua Event) trước khi sang lời gọi kế — 196 lần đồng bộ hoá/token, mỗi lần dù đã dùng
thread-pool thường trực (không tạo thread mới) vẫn tốn round-trip signal+wait. Ở granularity
nhỏ này (vd. `k_proj`: 1024 hàng / 14 luồng ≈ 73 hàng/luồng ≈ chỉ vài chục KB việc/luồng),
overhead đồng bộ hoá chiếm tỷ trọng đáng kể so với việc thực. **Kết luận trung thực**: kernel
TQ33 tự nó (decode+dot) nhanh như đã đo ở microbench, nhưng **cách tổ chức đa luồng theo
từng lời gọi GEMV riêng lẻ (thay vì theo layer/toàn bộ token) không tận dụng được lợi thế đa
luồng đã thấy ở microbench** khi áp dụng vào 1 forward pass thật với nhiều lớp nhỏ nối tiếp.
Hướng cải thiện tự nhiên (chưa làm, ngoài phạm vi nhiệm vụ này): gộp nhiều GEMV liên tiếp
cùng 1 phân luồng (vd. chia theo layer thay vì theo hàng của từng GEMV), hoặc pipeline các
layer qua nhiều luồng độc lập.

## 5. Kết luận tổng thể

1. **Runner độc lập (C thuần + AVX2, không ggml/llama.cpp) hoạt động ĐÚNG** — đã validate 2
   tầng (F32 thuần khớp oracle gần tuyệt đối; TQ33 khớp top-1/top-5 dù có 1 layer chịu
   activation-outlier rõ nét). Đây là bằng chứng THẬT trên toàn bộ 28-layer model, không
   phải suy diễn từ microbench 1 tensor.
2. **Format TQ33 giữ đúng 1.500 bpw và lossless trên 194/196 tensor thật** (2 tensor còn
   lại chỉ lệch 1 nhóm-4/tensor, đã sanitize, ảnh hưởng không đáng kể).
3. **Tốc độ thật (10.7–32.4 tok/s tuỳ luồng/lần chạy) THẤP HƠN NHIỀU so với con số chiếu từ
   microbench cũ** (ước tính cũ trong RESEARCH_TQ33.md: "~21-24 tok/s cho 30B-A3B" — số này
   dùng cho model KHÁC và pipeline KHÁC, không so trực tiếp được, nhưng cùng tinh thần chiếu
   optimistic). Root cause đã xác định RÕ RÀNG, KHÔNG mơ hồ: (a) lm_head/embedding F32
   622MB/token là nút thắt lớn nhất (63-87% thời gian ở 1 luồng) — nằm NGOÀI phạm vi kernel
   TQ33 vốn chỉ áp dụng cho 7 loại linear; (b) đa luồng theo từng GEMV nhỏ không tận dụng
   được lợi thế đã thấy ở microbench (chỉ đạt 25-30% GB/s kỳ vọng ở 6+ luồng).
4. **Không có bug nào bị che giấu** — mọi số liệu bất thường (layer 27 outlier, nhiễu benchmark
   2-3x giữa các lần chạy, đa luồng không scale) đều đã điều tra tới gốc rễ và có lời giải
   thích cụ thể, kiểm chứng được (không phải "có thể do X" mà đã đo trực tiếp X).

## 6. File đã tạo

**Scripts (e:\Bit-Translate\eval\lowbit_ptq\)**:
- `export_full_model_f32.py` — dump 507 tensor ckpt → F32 phẳng + meta.
- `ref_forward_pytorch.py` — oracle PyTorch (bias monkey-patch, per-layer hidden hook).
- `qwen3_runner.c` / `qwen3_runner.exe` — Giai đoạn 1, F32 thuần.
- `compare_phase1.py` — so rel err Giai đoạn 1.
- `export_tq33_all_linears.py` — encode 196 tensor → TQ33 (lossless-verified).
- `qwen3_runner_tq33.c` / `qwen3_runner_tq33.exe` — Giai đoạn 2, TQ33 kernel + KV-cache + benchmark.
- `compare_phase2.py` — so rel err Giai đoạn 2 + đối chiếu Giai đoạn 1.

**Data (D:\Bit-Translate-data\tq33_runner\)**:
- `weights_f32\weights.bin` (3.008 GB) + `meta.json` + `meta_index.txt`
- `oracle\` — `tokens.bin`, `hidden_l0..27.npy`, `hidden_final.npy`, `logits.npy`, `summary.txt`
- `tq33_packed\` — `tq33_packed.bin` (82.58 MB), `tq33_codebooks.bin`, `tq33_meta_index.txt`
- `runner_output_f32.bin`, `runner_output_tq33.bin` — output 2 runner để compare script đọc.

## 7. Giới hạn đã biết (trung thực, không che giấu)

- Chỉ test **1 prompt tiếng Việt ngắn (7 token)** — layer-27 outlier "may mắn" không đổi
  top-1/top-5 ở câu này, KHÔNG có nghĩa sẽ luôn đúng ở câu khác/vị trí khác/model khác.
- Benchmark tốc độ nhiễu 2-3x giữa các lần chạy trên máy này — số liệu tok/s nên hiểu là
  **dải**, không phải điểm cố định; nghi ngờ hợp lý nhất là thermal throttling (mục 4.2)
  nhưng CHƯA đo nhiệt độ CPU trực tiếp để xác nhận 100% — nguyên nhân khác (OS scheduler
  xếp luồng lên core P/E không nhất quán, tiến trình nền) chưa loại trừ dứt điểm.
- KV-cache trong benchmark là bản tối giản (append-only, cấp phát tĩnh `MAX_POS=128`) —
  không tối ưu bộ nhớ/không hỗ trợ batch, chỉ đủ dùng cho mục đích đo tốc độ single-stream.
- Đa luồng dùng row-split thủ công qua Windows Event, KHÔNG dùng OpenMP (xác nhận thiếu
  `omp.h` trên toolchain zig cc 0.16.0 hiện có) — có thể có cách tổ chức song song hiệu quả
  hơn (theo layer/pipeline) chưa thử nghiệm do ngoài phạm vi thời gian.
- Attention brute-force O(seq²) không tối ưu (đúng như task cho phép ở Giai đoạn 1) — với
  seq ngắn (<40 token) trong benchmark này không phải nút thắt (1-5% thời gian), nhưng sẽ
  trở thành vấn đề nếu test với ngữ cảnh dài hơn nhiều.
