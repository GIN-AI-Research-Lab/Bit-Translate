# TQ33 SPEED-100 — nén int8 embed/lm_head + tối ưu kernel: Qwen3-0.6B đạt đỉnh 114,7 tok/s, vùng bền vững ~90–105 tok/s (8 luồng, máy đã "ramp")

**Ngày**: 2026-08-04 · **Máy**: máy B (Intel Core Ultra 5 225H, 4P+8E+2LPE, DDR5-5600,
DRAM thật 41,0 GB/s đỉnh @4T theo `bw_bench.c`) · **Chi phí**: $0 (100% local) ·
**Runner**: `qwen3_runner_tq33_fast.c` (BẢN MỚI — `qwen3_runner_tq33.c`/`qwen3_runner.c` gốc
GIỮ NGUYÊN không đụng tới)

## 0. Tóm tắt kết quả (đọc trước)

- **Mục tiêu 100–200 tok/s: ĐẠT Ở BIÊN DƯỚI, có điều kiện.** Cấu hình tốt nhất (8 luồng,
  chia việc tĩnh, AVX-VNNI-INT8): **đỉnh 114,7 tok/s**, phiên chạy bền vững 12 lần liên tiếp
  cho **median ~100–103 tok/s** (dải 66–115). Điều kiện: máy phải ở trạng thái "ramp"
  (đang tải liên tục) — chạy từ trạng thái vừa idle xong có thể chậm 3–7 lần trong ~10–30
  giây đầu (mục 5.3, đây là phát hiện mới về máy B, KHÔNG phải thermal throttling như giả
  định trước đây trong RESEARCH_TQ33_RUNNER.md 4.2).
- **Băng thông hiệu dụng đỉnh 28,6 GB/s = 69,8% trần DRAM 41 GB/s** — với ~250MB đọc/token
  (82,58MB TQ33 + 155,58MB int8-head + ~10MB KV/norm/bias), trần lý thuyết tuyệt đối của
  máy là ~164 tok/s; đạt 60–70% trần là mức thực tế tốt cho forward pass thật có 141 điểm
  đồng bộ luồng/token.
- **So với runner cũ** (`qwen3_runner_tq33.exe`, RESEARCH_TQ33_RUNNER.md): 10,7–32,4 tok/s
  → 66–115 tok/s: **nhanh hơn ~3,5x ở đỉnh, ~4–6x ở median**, đến từ 2 nguồn chính đúng
  như phân tích cũ chỉ ra: (1) lm_head/embed F32 622MB/token (chiếm 63–87% thời gian cũ)
  nén int8 per-row còn 155,58MB; (2) đồng bộ luồng Event (~5–20µs/lần × 196 lần/token)
  thay bằng spin-barrier có padding cache-line (~1–2µs) + gộp GEMV còn 141 dispatch/token.
- **Đúng/sai GIỮ NGUYÊN chuẩn đã validate**: so oracle PyTorch trên **3 prompt thật (7+21+14
  = 42 vị trí)**: top-1 tại vị trí cuối **KHỚP CẢ 3/3 prompt**, top-5 overlap 5/5 tại vị trí
  cuối cả 3, top-1 khớp 36/42 vị trí giữa chuỗi (6 vị trí lệch đều có margin oracle mỏng
  0,02–0,04 — cùng bậc với nhiễu TQ33 CŨ vốn đã lệch 1 vị trí như vậy, xem 4.2). Kernel mới
  kiểm chứng runtime **bit-identical** với kernel gốc trên 300 hàng thật (mục 4.1).
- **AVX-512: máy này KHÔNG có** — đo trực tiếp cpuid: `avx512f=0` (Arrow Lake-H, Intel bỏ
  AVX-512 trên client hybrid). Khẳng định "máy CÓ AVX512" trong đề bài là SAI; wheel
  llama-cpp-python từng crash nhiều khả năng vì wheel build AVX-512 chạy trên máy không có
  lệnh đó (illegal instruction). Thay thế đúng trên máy này: **AVX-VNNI-INT8** (`VPDPBSSD`,
  cpuid bit thật `avxvnniint8=1`) — dot-product s8×s8 1 lệnh, đo được +6,7% kernel-level.

## 1. Điểm xuất phát và kế hoạch (từ phân tích cũ)

RESEARCH_TQ33_RUNNER.md mục 4.3–4.4 đã đo: 196 tensor linear TQ33 chỉ 82,58MB/token nhưng
embed/lm_head F32 622MB/token chiếm 63–87% thời gian; đa luồng theo GEMV nhỏ chỉ đạt 25–30%
GB/s so với microbench vì 196 lần đồng bộ Event/token. Phép tính mục tiêu: cần tổng
byte/token ~150–250MB để có cơ hội 100+ tok/s trên trần 41GB/s. Kế hoạch (đã thực hiện đủ):

1. int8 per-row cho `model.embed_tokens.weight` (dùng cho CẢ input lookup lẫn output head —
   quyết định dùng embed_tokens thay lm_head.weight đã chốt từ trước, không làm lại): 622MB
   → 155,58MB (+0,61MB scale). Tổng byte/token ~250MB (kèm KV ~9MB ở cửa sổ sinh 64 token).
2. Kernel int8×int8 mới cho head + giữ nguyên semantics kernel TQ33.
3. Sửa tổ chức đa luồng (nút thắt #2) + thử AVX-512 (kết quả: không tồn tại) / AVX-VNNI.

## 2. Việc 1 — nén int8 embed_tokens/lm_head (`export_embed_int8.py`)

- **Phương pháp**: symmetric per-row, `scale[v] = max|W[v,:]|/127`, clip ±127 (tránh ±128
  để kernel maddubs không saturate: 127×127×2 = 32258 < 32767). Đọc thẳng từ dump
  `weights_f32/weights.bin` (giá trị giống hệt đọc lại ckpt `qat_gen4_n4.pt` vì dump là
  f32 của fp16 gốc).
- **Sai số tái tạo** (per-row rel L2): mean 8,25e-3, median 7,84e-3, max 2,16e-2.
- **Kiểm trước bằng numpy trên oracle thật** (hidden_final + logits của prompt đã validate):

| Cấu hình head | rel err logits | top-1 vị trí cuối | top-5 overlap cuối |
|---|---|---|---|
| w-int8, activation f32 | 7,82e-3 | KHỚP (id 11) | 5/5 |
| w-int8 + act-int8/g64 (đúng như kernel C) | **1,35e-2** | KHỚP (id 11) | 5/5 |

  Đối chiếu: logits của TQ33 runner cũ (head F32) đã có rel err **4,8e-2** so oracle — sai
  số int8-head thêm vào (1,35e-2) NHỎ HƠN nhiễu TQ33 sẵn có, không phải yếu tố chi phối.
- **Phân tích vị trí lệch top-1** (quan trọng để hiểu bản chất): int8-head lệch đúng 1/7 vị
  trí (pos 5, margin oracle top1-top2 chỉ **0,039**); trong khi TQ33 head-F32 CŨ (đã được
  chấp nhận) cũng lệch 1/7 vị trí (pos 0, margin **0,022**). Kết luận: cả 2 loại nhiễu chỉ
  lật được argmax ở những vị trí margin mỏng ~0,02–0,04; mọi vị trí margin bình thường
  (0,4–1,6) đều giữ nguyên.
- **Input lookup cũng dùng int8** (dequant 1 hàng ×scale, ~1KB/token): rel err hàng embedding
  ~7e-3 — được RMSNorm layer 0 hấp thụ, xem sai số end-to-end mục 4.2. Nghĩa là file model
  thật sự chỉ cần **82,58MB TQ33 + 155,58MB int8 + norm/bias ~1,6MB + codebook 0,2MB ≈ 240MB**
  (không cần giữ bảng F32 622MB nào).

## 3. Việc 2 — kernel + tổ chức đa luồng (`qwen3_runner_tq33_fast.c`)

### 3.1 AVX-512 vs AVX-VNNI-INT8 — đo cpuid thật, không đoán

Probe trực tiếp (leaf 7): `avx512f=0, avx512bw=0, avx512vl=0, avx512vnni=0` — **Arrow
Lake-H không có AVX-512**, không phải lỗi toolchain (zig cc build và chạy probe bình
thường). Nhưng leaf 7 subleaf 1: `avxvnni256=1`, `avxvnniint8=1` → dùng **`VPDPBSSD`**
(`_mm256_dpbssd_epi32`, s8×s8 dot 4-phần-tử → i32, 1 lệnh thay chuỗi 4 lệnh
abs/sign/maddubs/madd). Tích int8 là số học CHÍNH XÁC và cùng cách nhóm 4 phần tử → tổng
i32 từng lane **giống hệt từng bit** đường maddubs — kiểm chứng runtime lúc khởi động
(`[kernel-check]`... KHỚP, exe tự dừng nếu lệch). Kernel chọn qua function pointer theo
cpuid, có cờ `novnni` để ép đường maddubs (dự phòng đúng yêu cầu "nếu lỗi quay lại AVX2").

### 3.2 Kernel int8-head mới (`row_dot_i8_avx2` / `row_dot_i8_vnni`)

GEMV [151936×1024]: hàng int8 × activation int8 per-64-group (tái dùng `quantize_x_int8`
y hệt các linear TQ33), tích lũy i32 per-group → FMA với `xs[b]`, nhân `scale[v]` cuối hàng.
Argmax tính NGAY trong worker (per-slot best, merge cuối) — không cần quét lại 608KB logits.

### 3.3 Decode TQ33 trong thanh ghi (fix store-to-load-forwarding)

Kernel gốc decode 12B block → buffer `t64[64]` bằng 8 store 8B rồi load lại 2×32B — pattern
này bị **STLF fail** (store 8B → load 32B không forward được, stall mỗi load) ngay giữa vòng
lặp nóng nhất. Bản mới (`DECODE_REG`): 8 mục LUT load 8B + ghép thẳng vào 2 thanh ghi YMM
(`_mm256_set_epi64x`), không round-trip bộ nhớ. Kiểm chứng bit-identical với `decode_block`
gốc trên 300 hàng thật của 3 tensor rải 3 layer (chạy mỗi lần khởi động).

**Microbench 1 luồng XEN KẼ trong cùng tiến trình** (8 lượt × 8 layer × 7 tensor = 23,6MB/lượt
— xen kẽ để trung hòa trôi nhiệt/clock, phương pháp bắt buộc trên máy này):

| Kernel | GB/s (1T) | vs gốc |
|---|---|---|
| decode_block + maddubs (kernel gốc) | 4,89 | — |
| decode-reg + maddubs | 5,05 | +3,3% |
| **decode-reg + VNNI** | **5,22** | **+6,7%** |

Khiêm tốn hơn kỳ vọng (kernel vốn nghẽn ở phần decode LUT tuần tự chứ không chỉ STLF),
nhưng miễn phí về rủi ro vì bit-identical.

### 3.4 Thread-pool spin-barrier + padding cache-line (nút thắt #2 của runner cũ)

- Worker spin trên generation counter (`_mm_pause`), master TỰ LÀM khúc đầu rồi spin đợi
  done-counter — không còn Event/kernel round-trip (~5–20µs × 196 lần/token của runner cũ).
- **Bài học đắt giá đã đo được**: bản đầu để `g_gen/g_done/next/job-desc` chung cache-line
  → fetch-add của worker ping-pong đúng line mà mọi luồng khác đang đọc/spin. Chỉ riêng
  việc tách mỗi biến nóng ra 1 line 64B đã nâng 4T từ 57–60 lên **71–80 tok/s**.
- **Gộp GEMV giảm số điểm đồng bộ**: Q+K+V chung 1 dispatch (chung input đã quantize 1 lần
  — 4096 hàng ảo), gate+up+**silu** chung 1 dispatch (worker tính xong 2 dot của hàng i thì
  áp silu×up ngay — silu "miễn phí" trong worker thay vì 3072 expf tuần tự trên master),
  attention song song theo head, o/down riêng → **5 dispatch/layer × 28 + head = 141/token**
  (cũ: 196 + head, và mỗi linear lại quantize riêng).
- RoPE dùng bảng cos/sin tính trước theo đúng công thức tuần tự cũ (giá trị giống hệt từng
  bit); quantize activation AVX2 (chia thật + `cvtps` round-nearest-even = `lrintf` scalar).

### 3.5 Chia việc tĩnh vs work-stealing vs affinity — 3 thí nghiệm, 2 thất bại có số liệu

| Biến thể | Kết quả đo | Phán quyết |
|---|---|---|
| Work-stealing v1 (grain 32 hàng, KHÔNG padding) | 8T: 26–53 tok/s (tệ hơn tĩnh 82–90); 12T sập 0,66 tok/s | **BỎ** — false sharing biến grab-loop thành bão coherence; khúc nhỏ rời rạc phá stream tuần tự |
| Pin affinity core 0..nt-1 | 4T: 41–56 (tĩnh không pin: 71–80) | **BỎ** — giả định "core 0–3 = P-core" sai/cản Thread Director; Windows tự xếp tốt hơn |
| Work-stealing v2 (padding line riêng, grain 64–2048 liên tục) | 6T: **thắng** tĩnh (94–96 vs 72–90); 8T: **thua** tĩnh (66–96 vs 71–115) | **GIỮ cả 2** — cờ `steal`; mặc định tĩnh; đo A/B xen kẽ bằng cờ `absteal` |
| 12–14 luồng (mọi biến thể) | sập 2–42 tok/s, tệ nhất 0,66 | **KHÔNG dùng >8T** — spin-pool + oversubscription/LP-E pathology; đã ghi nhận, không che |

## 4. Đúng/sai — validate TRƯỚC/SAU đầy đủ

### 4.1 Kernel-level (chạy mỗi lần khởi động, exe tự dừng nếu lệch)

- `row_dot_i8` maddubs vs VNNI: KHỚP tuyệt đối.
- `row_dot_tq33` decode-reg (avx2 + vnni) vs `row_dot_tq33_ref` (decode_block gốc): KHỚP
  tuyệt đối trên 300 hàng thật.

### 4.2 Model-level vs oracle PyTorch — 3 prompt thật (`compare_phase3.py`)

Oracle sinh bằng đúng pipeline đã validate (`ref_forward_pytorch.py` logic, thêm 2 prompt
mới bằng `make_oracle_multi.py`): P1 = "Xin chào, hôm nay" (7 tok), P2 = "Chiều nay trời mưa
to, tôi phải mang ô đi làm và đường phố rất đông người." (21 tok), P3 =
"今日は天気がいいので、公園へ散歩に行きましょう。" (14 tok).

| Prompt | rel err logits | top-1 vị trí CUỐI | top-5 cuối | top-1 mọi vị trí | top-5 TB |
|---|---|---|---|---|---|
| P1 (7 tok) | 5,04e-2 | **KHỚP** (11) | 5/5 | 6/7 | 4,29/5 |
| P2 (21 tok) | 3,58e-2 | **KHỚP** (220) | 5/5 | 17/21 | 4,71/5 |
| P3 (14 tok) | 3,06e-2 | **KHỚP** (220) | 5/5 | 13/14 | 4,86/5 |

- So sánh nền: TQ33 runner CŨ (head F32) trên P1 có rel err logits 4,77e-2 và cũng lệch
  top-1 ở 1/7 vị trí giữa chuỗi — bản mới 5,04e-2 (chỉ +0,027e-1 do int8 embed+head).
  Layer-27 activation-outlier giữ nguyên hành vi cũ (rel err ~1,0–1,4e-1 tại layer 27 cả 3
  prompt) — không phải lỗi mới.
- Kiểm chéo 2 đường tính (batch compare không cache vs KV-cache từng token): token đầu tiên
  sinh bởi đường KV-cache = top-1 đường batch (= 11 với P1) — pass ở mọi phiên chạy.
- **Giới hạn trung thực**: 6/42 vị trí giữa chuỗi lệch top-1 — TẤT CẢ ở margin mỏng; với
  sinh văn bản thực tế (chỉ dùng logits vị trí cuối + sampling) ảnh hưởng nhỏ, nhưng nếu
  dùng cho PPL/teacher-forcing thì sai số 3–5e-2 logits là có thật, cùng bậc với nhiễu TQ33
  gốc. Chưa đo PPL trọn bộ dev set (cần chạy C runner theo teacher-forcing — ngoài phạm vi).

## 5. Tốc độ — số đo thật, kèm phát hiện mới về máy

### 5.1 Phương pháp

KV-cache autoregressive, sinh 32–64 token sau prefill 7 token, chỉ tính giờ phần sinh; đo
nhiều mức luồng × nhiều lần lặp; **A/B tĩnh-vs-steal xen kẽ trong cùng tiến trình** (cờ
`absteal`) để trung hòa trôi trạng thái máy. Máy cắm điện (BatteryStatus=2). Byte/token in
kèm để quy GB/s hiệu dụng: 82,58 (TQ33) + 156,19 (head int8+scale) + 1,63 (norm/bias) +
~5,6–9,3 (KV đọc+ghi, tùy cửa sổ) ≈ **246–250MB/token**.

### 5.2 Kết quả chính (8 luồng, phiên bền vững 12 lần liên tiếp, 64 tok/lần)

| Lần (xen kẽ) | static (tok/s) | steal (tok/s) |
|---|---|---|
| 1–2 | 71,5 | 94,1 |
| 3–4 | **106,6** | 72,1 |
| 5–6 | **114,7** | 77,6 |
| 7–8 | **103,4** | 96,2 |
| 9–10 | 93,1 | 66,5 |
| 11–12 | **101,8** | 69,5 |

- **static @8T: median ~102,6, max 114,7 (28,6 GB/s hiệu dụng = 69,8% trần 41GB/s)** —
  4/6 lần vượt 100.
- steal @8T: median ~74,9 (nhưng ở 6T steal lại THẮNG tĩnh: 84–96 vs 72–90 — xem 3.5).
- Các phiên khác cùng cấu hình static@8T: 100,5 / 98,7 / 98,0 tok/s — mốc ~100 lặp lại
  được qua nhiều phiên, không phải 1 lần may mắn.

Breakdown ở lần chạy tốt nhất (114,7 tok/s = 8,72ms/token): TQ33 52% (~4,5ms ≈ 18,2GB/s),
head-i8 39% (~3,4ms ≈ **45,9GB/s**) — cao hơn số 41GB/s của `bw_bench` là hợp lý: bw_bench
đo STREAM-**triad** (2 đọc + 1 ghi), còn head là stream THUẦN ĐỌC tuần tự (đọc-only thường
cao hơn triad trên DDR5), cộng thêm ranh giới timer. Attn 7%, norm 3%. Đúng như thiết kế:
head giờ chạy sát trần DRAM, TQ33 là phần còn dư địa (kernel decode compute-bound
~5,2GB/s/luồng).

### 5.3 Phát hiện mới về nhiễu máy B: KHÔNG phải (chỉ) thermal — là power-state ramp

Thí nghiệm chủ đích: nghỉ 4 phút → chạy ngay: 6 lần đầu chỉ 13–43 tok/s, tăng dần, cuối
phiên đạt 98,7–101,7; chạy tiếp NGAY khi "máy nóng": 54,8–98,0. Tức **máy vừa idle xong thì
CHẬM** (clock/EPP/memory-controller ở trạng thái tiết kiệm, mất ~10–30s tải liên tục để
ramp), ngược hẳn giả thuyết thermal-throttling cũ. Hệ quả thực dụng: (a) benchmark trên máy
này bắt buộc xen kẽ biến thể trong cùng phiên; (b) số "10,7–18,5 tok/s lần chạy 2–3" của
runner cũ trong RESEARCH_TQ33_RUNNER.md nhiều khả năng cũng dính hiệu ứng này chứ không
phải chỉ nhiệt; (c) phục vụ request thưa (cold) sẽ chậm hơn đáng kể con số benchmark.

### 5.4 Đối chiếu trước/sau (cùng máy, cùng model, cùng phương pháp KV-cache)

| | Runner cũ (`qwen3_runner_tq33.c`) | Runner mới (`qwen3_runner_tq33_fast.c`) |
|---|---|---|
| Đỉnh | 32,4 tok/s (14T) | **114,7 tok/s (8T static)** |
| Dải điển hình | 10,7–30 | 66–115 (median ~100 @8T, phiên ramp) |
| Byte/token | ~705MB (622 F32 head + 82,6) | ~250MB (156,2 int8 head + 82,6 + KV) |
| GB/s hiệu dụng đỉnh | ~7 (toàn model) | 28,6 (69,8% trần STREAM) |
| Đồng bộ luồng | Event, 196+1 lần/token | spin-barrier padded, 141 lần/token |

## 6. Phán quyết cuối

1. **Mốc 100–200 tok/s: CHẠM biên dưới một cách lặp lại được** — 100,5 / 98,7 / 101,7 /
   103,4 / 106,6 / 114,7 tok/s ở 8 luồng qua 3 phiên khác nhau, nhưng CHỈ khi máy đã ramp;
   median toàn cục qua mọi phiên/trạng thái là ~85–95. Nói "đạt 100–200 tok/s vô điều kiện"
   là KHÔNG trung thực; nói "đạt ~100–115 tok/s có điều kiện, bền vững ~90–105 khi tải liên
   tục" là đúng với số đo.
2. Con đường byte đã đi đúng: giảm 705→250MB/token cho đúng tỷ lệ tăng tốc dự đoán; phần
   head chạy sát trần DRAM; dư địa còn lại nằm ở kernel TQ33 (compute-bound decode) và
   ~10–12% overhead attention/norm/sync.
3. Đúng/sai không bị hy sinh: top-1 vị trí cuối khớp oracle cả 3 prompt, top-5 5/5, kernel
   mới bit-identical kernel cũ; sai số thêm của int8 embed/head nhỏ hơn nhiễu TQ33 sẵn có.

## 7. Hướng còn mở (chưa làm, ước lượng có căn cứ)

- **int4 per-row cho head** (155,6→77,8MB): tổng ~172MB/token → trần lý thuyết ~238 tok/s,
  thực tế có thể ~130–150. Rủi ro: sai số embedding int4 lớn hơn nhiều, phải lặp lại đúng
  quy trình validate mục 4 (không được đoán).
- Kernel TQ33 2-hàng/lượt (chia sẻ load xq, tăng ILP) — dự kiến +10–20% phần TQ33.
- Gộp norm+rope K/V vào dispatch QKV (bớt ~0,2–0,3ms serial/token).
- Scheduler-aware: đọc EfficiencyClass qua `GetLogicalProcessorInformationEx` để pin đúng
  P-core thay vì đoán chỉ số core (thí nghiệm affinity mù đã fail, mục 3.5).
- PPL teacher-forcing trọn dev set trên C runner để đóng nốt khoảng trống validate 4.2.

## 8. File tạo mới trong lần này

**Code (e:\Bit-Translate\eval\lowbit_ptq\)**:
- `export_embed_int8.py` — quantize int8 per-row + verify numpy (recon, logits, input-lookup).
- `make_oracle_multi.py` — oracle PyTorch cho 2 prompt bổ sung (P2 VI dài, P3 JA).
- `compare_phase3.py` — compare_phase2 tổng quát hóa (argv oracle_dir + runner_bin).
- `qwen3_runner_tq33_fast.c/.exe` — runner mới; build
  `python -m ziglang cc -O3 -mavx2 -mfma -o qwen3_runner_tq33_fast.exe qwen3_runner_tq33_fast.c -lm`;
  cờ runtime: `novnni | steal | absteal | affin | headcapN | nocompare | nobench | kbench`.

**Data (D:\Bit-Translate-data\tq33_runner\)**:
- `embed_int8\embed_int8.bin` (155,58MB) + `embed_scales.bin` + `embed_int8_meta.txt`
- `oracle_p2\`, `oracle_p3\` — oracle 2 prompt mới (tokens/hidden/logits/summary)
- `runner_output_tq33_fast.bin`, `runner_output_fast_p2.bin`, `runner_output_fast_p3.bin`

**Không đụng tới**: `qwen3_runner_tq33.c`, `qwen3_runner.c`, `tq33_bench.c` và mọi file data
cũ — đường cũ vẫn build/chạy được nguyên trạng để so sánh/rollback.
