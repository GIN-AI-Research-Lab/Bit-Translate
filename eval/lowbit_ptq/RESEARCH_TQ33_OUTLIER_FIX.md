# TQ33 outlier-channel isolation — thí nghiệm thăm dò trên Qwen3-30B-A3B

**Ngày**: 2026-08-04 · **Trạng thái**: Giai đoạn 1 xong (đo thật), Giai đoạn 2 xong (cài đặt +
validate rel-err), Giai đoạn 3 xong (sinh văn bản thật, so sánh) · **Máy**: máy B (Intel Core
Ultra 5 225H, 14 luồng, không CUDA) · **Chi phí**: $0 (100% local, không Modal/GPU/train lại)

## 0. Tóm tắt kết quả (đọc trước) — KẾT QUẢ ÂM TÍNH, TRUNG THỰC

**Kỹ thuật outlier-channel isolation (kiểu LLM.int8() decomposition) được cài đặt ĐÚNG và VERIFY
được bằng số đo, nhưng KHÔNG cải thiện chất lượng sinh văn bản trong thí nghiệm này.** Cụ thể:

1. **Giai đoạn 1 (đo thật, không đoán)**: phát hiện kênh `cur` (input router+expert gate/up,
   không gian HIDDEN=2048) có **kênh 0 là "massive activation" cực mạnh và CỐ ĐỊNH** — kênh 0
   là giá trị **LỚN NHẤT TUYỆT ĐỐI trong toàn bộ 2048 kênh** (không chỉ trong nhóm-64 chứa nó)
   ở **45/48 layer liên tục (layer 2→46)**, biến mất hoàn toàn ở layer 0, 1 và đặc biệt là layer
   47 cuối cùng (rank rơi xuống 1876/2048). Đây là chỉ số kênh **HOÀN TOÀN KHÁC** báo cáo 0.6B
   (dim ~48/52 trong hidden=1024) — xác nhận đúng cảnh báo trong nhiệm vụ là phải đo lại, không
   suy diễn. Ngược lại, kênh outlier cho `h` (input down_proj, MOE_FFN=768) có tín hiệu **yếu
   và KHÔNG nhất quán** — không có kênh cố định chiếm ưu thế (giả thuyết hợp lý: `h` được tính
   lại từ đầu bởi trọng số RIÊNG của expert được chọn — 128 expert khác nhau — trong khi `cur`
   gắn liền với residual stream bền vững xuyên suốt mạng).

2. **Giai đoạn 2 (cài đặt + validate rel-err layer-by-layer so oracle)**: fix hoạt động ĐÚNG như
   thiết kế toán học — **layer 2–23 rel err giảm trung bình 4.8 lần** (0.0180 → 0.00376), nhưng
   **layer 33–47 rel err lại TĂNG trung bình 1.64 lần**, đặc biệt **layer cuối (47) TĂNG từ
   7.83e-2 lên 1.78e-1 (tệ hơn 2.27 lần)** — đúng layer mà nhiệm vụ kỳ vọng cải thiện nhất. Điều
   tra sâu (không dừng ở con số thô): đây **KHÔNG PHẢI bug** — so với oracle THẬT, routing (chọn
   expert) sau khi fix thực ra **khớp oracle NHIỀU HƠN** (85/336 vị trí lệch so với 104/336
   trước fix — cải thiện 18%), nhưng một số ít lần "lật" chọn expert xảy ra đúng ở các layer sâu
   (33+), và vì hidden state cộng dồn qua residual, 1 lần chọn SAI expert ở layer sâu đủ để
   khuếch đại rel-err cho mọi layer sau đó — đây là đặc tính VỐN CÓ của MoE routing nhạy với
   nhiễu số học ở các expert gần-hòa (đã ghi nhận ngay trong báo cáo gốc `RESEARCH_TQ33_RUNNER_
   30B.md`), không phải lỗi do kỹ thuật fix gây ra.

3. **Giai đoạn 3 (sinh văn bản thật, sampling temp=0.7/top_p=0.9, cùng seed=12345)**: cả 3 prompt
   (chat/toán/code) **KHÔNG mạch lạc hơn** sau fix — không trả lời được câu hỏi, không tính đúng
   29.000đ, không sinh code Python `def is_prime`, giống hệt trước fix ở mức "0/3 đạt tiêu chí"
   cả hai điều kiện. Đo khách quan (tỷ lệ loại token): sau fix, tỷ lệ token THUẦN SỐ tăng vọt
   (chat 2,2%→58,8%; suy luận 4,7%→47,7%; code 2,1%→13,1%) trong khi tỷ lệ token CHỮ giảm mạnh
   (chat 91,1%→0,0%; suy luận 88,2%→6,8%) — cho thấy fix có thể đã đổi kiểu suy thoái (từ "câu
   có cấu trúc ngữ pháp nhưng vô nghĩa" sang "chuỗi số lặp lại") chứ không sửa được sự thiếu
   mạch lạc gốc rễ.

**Giả thuyết cho kết quả âm tính**: outlier-channel trong `cur` không phải nguyên nhân CHÍNH
của sự thiếu mạch lạc — nó là 1 nguồn sai số lượng tử hóa THẬT (đã verify giảm rõ rệt ở layer
2–23), nhưng (a) chỉ là 1 phần nhỏ trong TỔNG sai số tích lũy qua 48 layer × 7 loại linear/layer
× 8 expert/layer (Q/K/V/O attention KHÔNG được fix trong thí nghiệm này), và (b) MoE routing tự
nó vốn cực nhạy với nhiễu số học bất kể nguồn nhiễu nào — nên việc giảm 1 nguồn nhiễu cụ thể
không đảm bảo cải thiện đầu ra cuối cùng khi routing có thể "lật" theo hướng khác ở layer sâu.
Giả thuyết phù hợp nhất với các phát hiện trước đó (`RESEARCH_TQ33_RUNNER_30B.md`,
`lowbit-lab-hoc-tap` trong bộ nhớ dự án): **nguyên nhân chính của sự thiếu mạch lạc nhiều khả
năng là model chưa qua KD/fine-tune SAU KHI nén** (S1-only), không phải riêng activation
quantization — fix ở đây chỉ chạm được 1 phần rất nhỏ của tổng sai số, không đủ để đổi chiều
kết quả sinh văn bản.

## 1. Bối cảnh & phạm vi thí nghiệm

Runner C độc lập (không ggml/llama.cpp) chạy Qwen3-30B-A3B nén ternary 1.5bpw (TQ33), đã validate
kiến trúc đúng 48/48 layer (`RESEARCH_TQ33_RUNNER_30B.md`) nhưng sinh văn bản thật không mạch
lạc. Báo cáo 0.6B (`RESEARCH_TQ33_RUNNER.md` mục 3.3) và 30B (`RESEARCH_TQ33_RUNNER_30B.md` mục
3.4) đều ghi nhận hiện tượng "activation outlier channel" (massive activations, văn liệu đã
biết) ở layer sâu/cuối — giả thuyết thăm dò: liệu cô lập kênh outlier ra khỏi lượng tử hóa int8
(kiểu LLM.int8() decomposition) có cải thiện chất lượng sinh văn bản không.

**Ràng buộc đã tuân thủ tuyệt đối**:
- Không dùng Modal/GPU/train lại — 100% local, chỉ sửa cách RUNNER xử lý lúc suy luận.
- Không đụng ckpt gốc, dữ liệu TQ33 đã encode (`manifest.json`, `linears/*.tq33`, `extras.bin`).
- Không sửa/xóa `run_benchmark()`/chế độ "bench" đã validate — **verify bằng SHA256**: dump
  hiden-state từ `run_benchmark()` sau khi thêm toàn bộ code mới **giống hệt bit-for-bit**
  (`370f11eb...`) dump gốc trước khi sửa; chế độ "gen" (sampling) cũng cho **SHA256 giống hệt**
  (`52952ee7...`) so với `results.json`/`chat_out.bin` gốc khi tắt outlier-fix. Cả 2 xác nhận
  toàn bộ code mới hoàn toàn "trong suốt" (no-op) khi không kích hoạt.

## 2. Giai đoạn 1 — đo outlier-channel THẬT trên 30B (không đoán từ báo cáo 0.6B)

### 2.1 Phương pháp

Thêm instrumentation vào `qwen3moe_runner_tq33.c` (mặc định TẮT, `g_stats_enabled=0`, không ảnh
hưởng bench/gen):

- `g_chan_absmax_cur[N_LAYER][HIDDEN]` — tích lũy `max(|giá trị|)` theo TỪNG KÊNH của `cur` (sau
  `rmsnorm(cur, ffn_inp, g_ln_post[l], HIDDEN)`, input CHUNG cho router VÀ expert gate/up), gọi
  ở MỌI layer MỌI vị trí token qua `stats_update_cur()`.
- `g_chan_absmax_h[N_LAYER][MOE_FFN]` — tích lũy tương tự cho `h = silu(gate)*up` trong
  `moe_expert_job()` (input cho down_proj), qua `stats_update_h()`.
- Chế độ CLI mới `stats <runner_dir> <extras_bin> <prompt_tokens_bin> <out_stats_bin>`: chạy
  `forward_one_token()` qua 1 chuỗi prompt THẬT (không sinh, không sampling), ép `g_nthreads=1`
  (tránh race khi nhiều expert-job ghi đồng thời vào cùng mảng thống kê), ghi 2 mảng ra file
  nhị phân.
- Prompt hiệu chuẩn: **20 câu đầu của `D:\Bit-Translate-data\iq_compare\val100_vi.txt`** (câu
  tiếng Việt thật, có sẵn trong repo), tokenize bằng `AutoTokenizer.from_pretrained(".../smoke06b")`
  → **470 token** (đủ "vài trăm token" theo yêu cầu, còn margin dưới `MAX_POS=768`).
  Script: `prepare_stats_prompt.py`.
- Phân tích: `analyze_outlier_stats.py` — với mỗi layer, tìm kênh có absmax lớn nhất, tính tỷ lệ
  so với MEDIAN của 63 kênh còn lại TRONG CÙNG 1 nhóm-64 (đúng nhóm mà `quantize_x_int8()` dùng
  để tính scale — đo trực tiếp mức độ "kéo méo" mà outlier gây ra), và kiểm tra giả thuyết
  "kênh cố định xuyên layer" bằng tần suất xuất hiện, **không giả sử trước**.

Chạy: `qwen3moe_runner_tq33.exe stats <runner> <extras> stats_prompt.bin outlier_stats.bin` —
470 token, 1 luồng, **63.5 giây**.

### 2.2 Kết quả — `cur` (HIDDEN=2048, input router + expert gate/up)

**Phát hiện chính: kênh 0 là massive-activation CỰC MẠNH và GẦN NHƯ CỐ ĐỊNH.**

Rank của kênh 0 (0 = giá trị lớn nhất trong TOÀN BỘ 2048 kênh của layer đó, không chỉ trong
nhóm-64) qua từng layer:

| Layer | 0 | 1 | 2–46 (liên tục) | 47 |
|---|---|---|---|---|
| Rank của kênh 0 (0=lớn nhất/2048) | 88 | 19 | **0 (mọi layer)** | 1876 |

Tức là kênh 0 **hoàn toàn KHÔNG phải outlier ở layer 0/1**, trở thành **giá trị lớn nhất tuyệt
đối trong cả vector 2048 chiều ở TẤT CẢ 45 layer liên tiếp từ layer 2 đến layer 46**, rồi **biến
mất hoàn toàn ở layer 47** (layer cuối cùng — đúng layer mà báo cáo gốc thấy rel-err nhảy vọt,
nhưng ở layer NÀY kênh 0 lại KHÔNG phải outlier — outlier ở layer 47 chuyển sang kênh khác, xem
bảng layer 47 dưới).

Tỷ lệ (absmax kênh top-1 / median 63 kênh còn lại cùng nhóm-64) ở 1 số layer tiêu biểu:

| Layer | Kênh top-1 | absmax | median-nhóm | Tỷ lệ |
|---|---|---|---|---|
| 0 | 468 | 3.65 | 0.536 | 6.8x (chưa phải outlier mạnh) |
| 1 | 40 | 5.15 | 0.825 | 6.2x |
| 2 | **0** | 148.8 | 0.547 | **272x** |
| 3 | **0** | 110.2 | 0.895 | **123x** |
| 24 | **0** | 62.2 | 1.392 | 44.7x |
| 46 | **0** | 120.1 | 1.946 | 61.7x |
| 47 | 364 | 46.4 | 1.836 | 25.3x (kênh 0 KHÔNG còn là outlier) |

Tần suất kênh là "top-1" (absmax lớn nhất) qua 48 layer: **kênh 0: 45/48 layer (93,8%)** — vượt
trội hoàn toàn so với kênh xếp thứ 2 (kênh 468, chỉ 1/48). Các kênh xuất hiện lặp lại trong
top-5 (không chỉ top-1) với tần suất giảm dần: kênh 8 (30/48), kênh 40 (22/48), kênh 24 (19/48),
kênh 28 (15/48), kênh 32 (12/48), kênh 236 (9/48), kênh 4 (8/48). **Tất cả 6 kênh [0,8,24,28,32,
40] đều nằm trong CÙNG 1 nhóm-64 đầu tiên (kênh 0–63)** — thuận lợi cho việc cài fix (chỉ cần
giải mã 1 block/hàng).

### 2.3 Kết quả — `h` (MOE_FFN=768, input down_proj)

**Tín hiệu YẾU HƠN VÀ ÍT NHẤT QUÁN HƠN HẲN so với `cur`** — không có kênh cố định chiếm ưu thế:

- Tần suất "top-1": kênh mạnh nhất (kênh 0) chỉ là top-1 ở **3/48 layer (6,2%)**; các kênh
  20/80/76/88 chỉ 2/48 mỗi kênh — phân tán, không có kênh áp đảo như `cur`.
- Tần suất trong top-5: kênh 8 (10/48), kênh 0 (9/48), kênh 16 (6/48), kênh 20 (5/48), rồi
  nhiều kênh đồng hạng ở 4/48 (44, 28, 24, 84, 112, 76).
- Kiểm tra rank toàn cục (lọt top-50/768): kênh 0 lọt **31/48 layer**, kênh 8 lọt **26/48**,
  yếu hơn nhiều so với mức "gần như luôn luôn" của kênh 0 trong `cur`.

**Giả thuyết hợp lý (không khẳng định chắc chắn)**: `h` được TÍNH LẠI mỗi lần từ trọng số RIÊNG
của expert được chọn (128 expert khác nhau, thay đổi theo routing mỗi layer/vị trí), khác hẳn
`cur` vốn gắn liền với residual/hidden stream BỀN VỮNG xuyên suốt độ sâu mạng (nơi hiện tượng
"massive activation cố định" trong văn liệu thường được quan sát). Do đó không có 1 kênh
"massive activation" CỐ ĐỊNH cho `h` giống như `cur`.

### 2.4 Quyết định danh sách kênh bảo vệ (dựa trên số đo, K=6 mỗi phía)

```
PROTECT_CUR (không gian HIDDEN=2048) = { 0, 8, 24, 28, 32, 40 }
PROTECT_H   (không gian MOE_FFN=768) = { 0, 8, 16, 20, 24, 28 }
```

Cả 2 danh sách đều lấy từ tần suất đo được (không đoán trước), đều nằm gọn trong nhóm-64 đầu
tiên của không gian tương ứng. Lưu ý quan trọng về mặt toán học: bảo vệ 1 kênh KHÔNG PHẢI outlier
trong 1 nhóm cụ thể là **NO-OP tuyệt đối** (loại 1 phần tử không phải max ra khỏi phép tính max()
không đổi kết quả max còn lại) — nên dù tín hiệu cho `h` yếu hơn, việc thử áp dụng vẫn AN TOÀN
về mặt số học (không thể làm THÊM sai lệch do lựa chọn kênh sai, chỉ có thể "không giúp gì" ở
layer mà kênh đó không thực sự là outlier).

## 3. Giai đoạn 2 — cài đặt outlier-channel isolation + validate rel-err

### 3.1 Cài đặt trong `qwen3moe_runner_tq33.c`

Nguyên tắc: **giữ nguyên `quantize_x_int8()`, `linear_tq33()`, `linear_tq33_rows_from_q()`,
`run_benchmark()` gốc** — mọi hàm mới đều THÊM (không sửa/xóa), kích hoạt qua cờ toàn cục
`g_use_outlier_fix` (mặc định 0):

- `quantize_x_int8_protected()`: giống `quantize_x_int8()` nhưng loại các kênh trong danh sách
  protected ra khỏi CẢ (a) phép tính `max()` để ra scale nhóm-64, VÀ (b) mảng int8 (đặt 0 để
  không đóng góp sai vào tổng int8).
- `linear_tq33_add_protected()`: cộng THÊM đóng góp CHÍNH XÁC (FP32) của các kênh protected vào
  `out[]` — với mỗi hàng đầu ra, giải mã ĐÚNG block-64 chứa kênh protected bằng `decode_block()`
  (hàm gốc, không sửa), lấy giá trị ternary CHÍNH XÁC tại vị trí kênh, nhân với giá trị FP32
  THẬT (chưa lượng tử hóa) + scale codebook của đúng block đó. Vì số kênh protected rất ít
  (≤8) và chỉ nằm trong 1 block, chi phí thêm không đáng kể.
- 2 điểm tích hợp CHÍNH XÁC theo phạm vi nhiệm vụ: (1) quantize hóa `cur` trước khi đưa vào
  gate/up của 8 expert (dùng `PROTECT_CUR`); (2) quantize hóa `h` trước khi đưa vào down_proj
  trong `moe_expert_job()` (dùng `PROTECT_H`). Q/K/V/O attention **KHÔNG bị đụng tới** (ngoài
  phạm vi nhiệm vụ) — vẫn dùng `quantize_x_int8()`/`linear_tq33()` gốc dù `g_use_outlier_fix=1`.
- CLI mới: `genfix` (như `gen` nhưng bật fix — dùng lại NGUYÊN VĂN `run_generate()`), `dumpfix`
  (như chế độ mặc định nhưng bật fix + ghi dump ra `runner_dump_layers_fixed.bin` — dùng lại
  NGUYÊN VĂN `run_benchmark()`, không đụng dump gốc `runner_dump_layers.bin`).

**Regression check (bằng chứng, không phải khẳng định suông)**:

| Kiểm tra | Kết quả |
|---|---|
| Build `-O3 -mavx2 -mfma -Wall -Wextra` | 0 warning |
| SHA256 dump `run_benchmark()` (mặc định, fix tắt) trước/sau khi thêm code mới | `370f11eb...` = `370f11eb...` **GIỐNG HỆT** |
| SHA256 output `gen` (sampling, fix tắt) trước/sau khi thêm code mới, cùng seed/prompt | `52952ee7...` = `52952ee7...` **GIỐNG HỆT** |
| Token đầu tiên sinh ra (greedy, benchmark) | 128296 — khớp `RESEARCH_TQ33_RUNNER_30B.md` |

### 3.2 Kết quả rel-err layer-by-layer (so oracle F32 độc lập, `validate_30b_layers_compare.py`)

Oracle đọc TRỰC TIẾP ckpt gốc 61GB qua `safe_ckpt_reader` (giống hệt `validate_30b_layers.py` đã
có, KHÔNG sửa file gốc — copy logic sang script mới để tính oracle 1 lần, so cả 2 dump). Cùng
prompt 7 token `"Xin chào, hôm nay"` như báo cáo gốc.

| Layer | rel_err TRƯỚC | rel_err SAU | Tỷ lệ (sau/trước) | Ghi chú |
|---|---|---|---|---|
| 0 | 3.9296e-03 | 3.7601e-03 | 0.957x | ~không đổi |
| 1 | 3.3692e-03 | 2.9389e-03 | 0.872x | giảm |
| 2 | 2.4346e-02 | 4.2400e-03 | 0.174x | **GIẢM MẠNH** |
| 3 | 1.7836e-02 | 2.5717e-03 | 0.144x | **GIẢM MẠNH** |
| 4–9 | 1.78–1.79e-02 | 2.69–2.75e-03 | ~0.15x | **GIẢM MẠNH** (đều) |
| 10–20 | 1.756–1.777e-02 | 2.82–3.18e-03 | 0.16–0.18x | **GIẢM MẠNH** (đều) |
| 21 | 1.7553e-02 | 7.3193e-03 | 0.417x | giảm mạnh |
| 22–23 | 1.756–1.760e-02 | 9.36–9.95e-03 | 0.53–0.57x | giảm |
| 24 | 1.8139e-02 | 1.9241e-02 | 1.061x | **tăng** (routing bắt đầu lệch, xem 3.3) |
| 25–32 | 1.808–1.815e-02 | 1.299–1.500e-02 | 0.72–0.83x | giảm |
| 33 | 1.9077e-02 | 4.0928e-02 | 2.145x | **TĂNG MẠNH** |
| 34–36 | 1.889–1.909e-02 | 3.63–3.99e-02 | 1.92–2.09x | **TĂNG MẠNH** |
| 37–46 | 1.847–2.312e-02 | 2.15–3.36e-02 | 1.16–1.45x | tăng |
| **47 (cuối)** | **7.8344e-02** | **1.7796e-01** | **2.272x** | **TĂNG MẠNH** |

**Thống kê tổng hợp**:

| Phạm vi | Trung bình rel_err TRƯỚC | Trung bình rel_err SAU | Tỷ lệ |
|---|---|---|---|
| Layer 2–23 (outlier `cur` hoạt động mạnh) | 0.0180 | 0.00376 | **0.209x (giảm 4.8 lần)** |
| Layer 33–47 (routing lệch tăng dần) | 0.0235 | 0.0385 | **1.64x (tăng 1.64 lần)** |
| Toàn bộ 48 layer — trung bình | 0.0192 | 0.0166 | 0.87x |
| Toàn bộ 48 layer — trung vị | 0.0181 | 0.0130 | 0.72x |
| Số layer giảm rel-err ≥5% | 31/48 | | |
| Số layer tăng rel-err ≥5% | 16/48 | | |

Theo **trung vị** và **số layer cải thiện** (31/48), fix có vẻ là 1 cải thiện ròng ở mức từng
layer — nhưng layer 47 (layer CUỐI, chính là layer nhiệm vụ kỳ vọng cải thiện nhất) lại **TỆ HƠN
2.27 lần**, và các layer sâu 33-47 nói chung tệ hơn khi tính trung bình đơn giản (bị kéo lên bởi
layer 47 có giá trị tuyệt đối lớn nhất).

### 3.3 Điều tra nguyên nhân: routing (chọn expert) — KHÔNG PHẢI bug

**Câu hỏi**: tại sao rel-err TĂNG ở layer sâu dù kỹ thuật fix đã verify đúng toán học và giảm
rel-err rõ rệt ở layer 2–23?

**Bước 1 — kiểm tra routing có đổi giữa 2 lần chạy (trước/sau fix) không**: so `sel_idx` (tập 8
expert được chọn) giữa dump trước/sau ở từng layer/vị trí (7 vị trí × 48 layer = 336 tổ hợp).
Kết quả: routing **BẮT ĐẦU LỆCH TỪ LAYER 2** (1/7 vị trí), tăng dần theo độ sâu (3-5/7 vị trí ở
layer 24-36) — xác nhận: hidden state bị thay đổi bởi fix TỪ layer 2 trở đi (khớp với việc `cur`
outlier hoạt động mạnh nhất từ layer 2), và vì router nhận input là `cur` (đã bị fix ảnh hưởng),
quyết định routing có thể thay đổi ngay khi hidden state đổi đủ nhiều.

**Bước 2 — câu hỏi quan trọng hơn: routing SAU fix có khớp ORACLE THẬT nhiều hơn hay ít hơn
routing TRƯỚC fix?** (không chỉ so 2 runner với nhau, mà so CẢ HAI với oracle độc lập —
`check_routing_vs_oracle.py`, tính lại oracle routing tại mỗi layer bằng float32 đầy đủ):

| | Số vị trí (layer, pos) lệch tập-expert so với oracle |
|---|---|
| Runner TRƯỚC fix | **104 / 336** (31.0%) |
| Runner SAU fix | **85 / 336** (25.3%) |

**Routing SAU fix khớp oracle NHIỀU HƠN — giảm 18% số vị trí lệch** (104→85). Đi vào chi tiết
từng layer: SAU fix lệch ÍT HƠN oracle ở nhiều layer riêng lẻ (2,3,4,5,6,7,8,...,44,45) nhưng
lệch NHIỀU HƠN ở 1 số layer khác (11,18,19,22,23,28,30,33,37,39,41,42,46) — bao gồm layer 33
(nơi rel-err bắt đầu tăng vọt: "SAU LỆCH NHIỀU HƠN" 3 vs 2 vị trí).

**Kết luận cơ chế (có bằng chứng, không phải suy đoán)**: MoE routing với 128 expert có rất
nhiều lựa chọn Ở NGƯỠNG GẦN HÒA (đã ghi nhận ngay trong `RESEARCH_TQ33_RUNNER_30B.md` mục 3.4:
"96,4% lệch nằm ở 3 hạng THẤP NHẤT trong top-8" — đúng nơi trọng số renormalize gần bằng nhau
nhất). Bất kỳ thay đổi số học nào (kể cả thay đổi CÓ LỢI như fix này) đều có thể làm "lật" 1 vài
lựa chọn routing ở các vị trí gần-hòa đó — và MẶC DÙ về TỔNG THỂ fix làm routing khớp oracle
NHIỀU HƠN (85 vs 104), một số ít lần "lật" xảy ra đúng ở layer sâu (33+, 46). Vì hidden state
cộng dồn qua residual connection, 1 lần chọn SAI 1 expert ở layer sâu sẽ dùng TOÀN BỘ trọng số
KHÁC (không phải "nhiễu nhỏ" mà là "expert hoàn toàn khác") cho phần MoE của layer đó — sai số
này KHÔNG PHẢI cùng loại với nhiễu lượng tử hóa mượt mà fix nhắm tới, và lớn hơn nhiều — rồi lan
truyền/khuếch đại qua các layer còn lại (residual stream giữ nguyên sai số này tới cuối mạng).
Đây là đặc tính VỐN CÓ của kiến trúc MoE-128-expert khi routing dựa trên ngưỡng gần-hòa, **không
phải lỗi cài đặt của kỹ thuật outlier-isolation**.

## 4. Giai đoạn 3 — sinh văn bản thật, so sánh trực tiếp

### 4.1 Phương pháp

`gen_quality_check_fixed.py` (bản sao `gen_quality_check.py` gốc — KHÔNG sửa file gốc), gọi chế
độ CLI `genfix` thay vì `gen`. Cùng 3 prompt, cùng `temperature=0.7, top_p=0.9, seed=12345,
threads=8, n_gen=100` — MỌI điều kiện GIỐNG HỆT lần chạy gốc (`results.json`), chỉ khác runner
có bật outlier-fix hay không.

### 4.2 Kết quả — cả 3 prompt, đầy đủ, TRƯỚC/SAU đặt cạnh nhau

**Prompt "chat"**: *"Xin chào! Bạn có thể giới thiệu ngắn gọn về bản thân không?"*

| | Văn bản sinh ra (100 token) |
|---|---|
| **TRƯỚC fix** | " 1. Cảm giá là một số phần trong các trường có thể được tìm ra. Tần của các chương mới có thể được sử dụng như một số phương pháp mà có thể được sử dụng các lớp ảnh, hoặc các ứng dụng như các gói không có thể được sử dụng các lớp đã được tạo ra. Văng 1000, có thể có thể đi đúng và có thể dùng để kích thường và có thể điều chỉnh các phần nào. Dên và có" |
| **SAU fix** | " 2000, 2018, 2018, 10000. \n\n 2.15\n 2017-08 10.28\n 1000 2010, 1922, 1250, 1000, 1831, 105 100, 1000. 10" |

**Prompt "reasoning"** (toán đố táo/cam, đáp án đúng = 5000×3+7000×2 = **29.000 đồng**):
*"Một quả táo giá 5000 đồng, một quả cam giá 7000 đồng. Tôi mua 3 quả táo và 2 quả cam. Hãy tính
tổng số tiền tôi phải trả, giải thích từng bước."*

| | Văn bản sinh ra (100 token) |
|---|---|
| **TRƯỚC fix** | " 1. Cảm giác và phêp trân và điều khiển các hàm số lớn, có thể có thể được kiểm tra các sự thật của bạn. Một điều này có thể được sử dụng 200 hoặc 1 trong mỗi trân và có thể có thể được kiểm tra các loại khác. Các loại nào có thể được sử dụng các phần cần đóng giá, và có thể được kiểm tra các phần được đưa ra, và các phần trong 10" |
| **SAU fix** | "\n 200, 200, 1900, 198, 193, 100. Hãy đọc 1000, 100. 100, 1200, 100, 200, 100. 10.0.1\n\n Mật 100, 100.0, 100.0," |

**Prompt "coding"**: *"Write a Python function `is_prime(n)` that returns True if n is a prime
number and False otherwise. Include a short docstring."*

| | Văn bản sinh ra (100 token) |
|---|---|
| **TRƯỚC fix** | " 1. The main purpose is that the same thing is a \"m\" (the \"H\" is a special type of data that is used to help with the current user. The purpose is that the number of the original code is a \"c\" or \"c\" in the same way.  It is not a typical problem. But the process is not only a very easy task. For example, if you want to have a single sentence, the \"l' and 1." |
| **SAU fix** | " 1. The main purpose is that the same thing is a \"m\" (the \"H\" and \" 200\" for the \"sp, 100\" and 19.0) 100.0\n\n 2. The user can use the code to do the same kind of\n 1. The code is very similar to the actual programming in the same context. So I have tried to get the \"1.0\" for the same" |

**Đánh giá theo tiêu chí ĐO ĐƯỢC (không chủ quan)**:

| Tiêu chí cụ thể | TRƯỚC fix | SAU fix |
|---|---|---|
| Trả lời được câu hỏi tự giới thiệu? | Không | Không |
| Có tính đúng/gần đúng 29.000đ hoặc nhắc con số 5000/7000/3/2 một cách có ý nghĩa? | Không | Không |
| Có sinh cú pháp Python hợp lệ (`def`, `return`, `%`, vòng lặp...)? | Không | Không |
| Có nhắc từ khóa liên quan ("prime", "số nguyên tố", `is_prime`)? | Không | Không |

**Cả 2 điều kiện đều 0/4 trên mọi tiêu chí cụ thể** — không có cải thiện đo được nào về mặt nội
dung/tính đúng đắn.

### 4.3 Đo khách quan bổ sung — thành phần loại token (không đánh giá chủ quan)

Phân loại từng token đầu ra thành CHỮ (alpha), SỐ (digit), hoặc DẤU CÂU/KÝ TỰ KHÁC (punct):

| Domain | Điều kiện | Số token | % số (digit) | % chữ (alpha) | % dấu câu | % token duy nhất |
|---|---|---|---|---|---|---|
| chat | TRƯỚC | 90 | 2.2% | 91.1% | 6.7% | 53.3% |
| chat | SAU | 34 | **58.8%** | **0.0%** | 41.2% | 55.9% |
| reasoning | TRƯỚC | 85 | 4.7% | 88.2% | 7.1% | 49.4% |
| reasoning | SAU | 44 | **47.7%** | **6.8%** | 45.5% | 34.1% |
| coding | TRƯỚC | 97 | 2.1% | 78.4% | 19.6% | 55.7% |
| coding | SAU | 84 | 13.1% | 63.1% | 23.8% | 52.4% |

**Quan sát khách quan**: sau khi bật fix, tỷ lệ token THUẦN SỐ tăng mạnh ở cả 3 domain (rõ nhất
ở chat và reasoning), trong khi tỷ lệ token CHỮ (từ ngữ thật) giảm mạnh tương ứng — đặc biệt domain
"chat" chuyển từ 91,1% chữ xuống **0,0% chữ** (toàn bộ output sau fix chỉ còn số và dấu câu, không
còn 1 từ tiếng Việt/tiếng Anh nào). Đây là 1 kiểu suy thoái KHÁC (degenerate vào lặp số) so với
kiểu suy thoái trước fix (câu có cấu trúc ngữ pháp — chủ ngữ/vị ngữ/liên từ — nhưng vô nghĩa về
nội dung). Không thể khẳng định chắc chắn kiểu suy thoái nào "tệ hơn" một cách tuyệt đối, nhưng
theo tiêu chí "giống văn bản tự nhiên" thì kết quả SAU fix rõ ràng KÉM tự nhiên hơn ở 2/3 domain.

## 5. Kết luận & giả thuyết

1. **Kỹ thuật outlier-channel isolation được cài đặt ĐÚNG** — verify bằng 3 lớp bằng chứng độc
   lập: (a) toán học (bảo vệ kênh không phải outlier là no-op, đã chứng minh), (b) rel-err giảm
   RÕ RỆT 4.8 lần ở layer 2-23 nơi outlier `cur` hoạt động mạnh nhất theo đúng số đo Giai đoạn 1,
   (c) routing tổng thể khớp oracle THẬT nhiều hơn (85/336 so 104/336, cải thiện 18%).
2. **Nhưng KHÔNG cải thiện chất lượng sinh văn bản cuối cùng** — vì (a) MoE routing với 128
   expert cực nhạy với BẤT KỲ thay đổi số học nào ở các lựa chọn gần-hòa (đặc tính vốn có của
   kiến trúc, không phải lỗi của fix), khiến 1 số ít lần "lật" routing ở layer sâu đủ để làm
   rel-err layer cuối (47) TỆ HƠN 2.27 lần — đúng layer quyết định logits cuối cùng dùng để
   sample token; (b) fix chỉ chạm 2 điểm cụ thể (cur→gate/up, h→down_proj) trong khi Q/K/V/O
   attention và hàng trăm nguồn nhiễu lượng tử hóa khác trên 48 layer × 8 expert/layer vẫn y
   nguyên — 1 phần nhỏ của tổng sai số được sửa không đủ để đổi chiều kết quả.
3. **Giả thuyết phù hợp nhất cho sự thiếu mạch lạc gốc rễ (không phải do outlier-channel)**:
   khớp với hiểu biết đã có trong bộ nhớ dự án (`lowbit-lab-hoc-tap`, `ternary-lab-verdict-map`)
   — PTQ ternary (post-training quantization, không train lại) có giới hạn rate-distortion cơ
   bản; con đường thật để có mạch lạc là **BitNet Distillation** (train/KD lại sau khi nén),
   không phải các kỹ thuật vá lỗi lượng tử hóa activation lúc suy luận như outlier-isolation.
   Kỹ thuật này có thể vẫn HỮU ÍCH như 1 thành phần trong pipeline có KD đầy đủ (nơi routing đã
   ổn định hơn nhờ train lại), nhưng ĐƠN LẺ trên PTQ thuần thì không đủ.
4. **Đây là kết quả HỢP LỆ của thí nghiệm thăm dò** — không phải thất bại trong việc cài đặt/đo
   đạc (cả 2 đều verify cẩn thận, nhiều lớp bằng chứng độc lập), mà là bằng chứng THỰC NGHIỆM
   cho thấy outlier-channel isolation không phải đòn bẩy chính cho vấn đề mạch lạc trên setup
   PTQ-only này.

## 6. Giới hạn đã biết (trung thực, không che giấu)

- Chỉ test **1 seed (12345), 1 bộ 3 prompt cố định** cho Giai đoạn 3 — theo đúng yêu cầu nhiệm vụ
  ("so sánh công bằng" với kết quả cũ), nhưng KHÔNG loại trừ khả năng seed/prompt khác cho kết
  quả khác (routing "lật" ở layer sâu có thể phụ thuộc nhiều vào token cụ thể).
- Chỉ fix 2 điểm (`cur`→gate/up, `h`→down_proj) theo đúng phạm vi nhiệm vụ — KHÔNG fix Q/K/V/O
  attention (`attn_concat`→o_proj cũng gọi `quantize_x_int8()` nhưng KHÔNG đo/fix trong thí
  nghiệm này) — chưa biết liệu outlier có tồn tại ở đó hay không (task không yêu cầu đo).
- Giai đoạn 1 chỉ hiệu chuẩn trên 470 token tiếng Việt (1 domain) — kênh outlier phát hiện được
  (đặc biệt kênh 0 của `cur`) có khả năng cao là hiện tượng cấu trúc/kiến trúc (khớp văn liệu
  "massive activations" xuất hiện với BẤT KỲ input nào), nhưng chưa verify chéo bằng input tiếng
  Anh/code để loại trừ khả năng phụ thuộc domain.
- Danh sách kênh protected (K=6 mỗi phía) chọn theo tần suất đo được, KHÔNG phải kết quả của 1
  phép tối ưu hóa (vd. thử nhiều giá trị K, nhiều tiêu chí chọn kênh khác nhau và so sánh) — có
  thể có lựa chọn K/kênh khác cho kết quả khác, ngoài phạm vi thời gian thí nghiệm thăm dò này.
- 41 expert + 1 layer attention (layer 24) vẫn bị zero-hóa vĩnh viễn do lỗi tải ckpt gốc (đã ghi
  nhận trong `RESEARCH_TQ33_RUNNER_30B.md` mục 1.1) — ảnh hưởng CẢ 2 điều kiện trước/sau fix như
  nhau, không phải biến số của thí nghiệm này nhưng góp phần vào baseline "không mạch lạc" chung.
- Không đo tốc độ sinh văn bản chi tiết trước/sau fix (thêm `decode_block()` cho protected
  channels mỗi hàng đầu ra) — quan sát sơ bộ không thấy chậm đi rõ rệt trong log chạy thực tế,
  nhưng không có số liệu tok/s chính xác để trích dẫn (ngoài phạm vi nhiệm vụ, vốn tập trung vào
  chất lượng chứ không phải tốc độ).

## 7. File đã tạo/sửa + lệnh tái lập

**Sửa (KHÔNG đụng `run_benchmark()`/chế độ "bench", verify SHA256 — xem mục 3.1)**:
- `e:\Bit-Translate\eval\lowbit_ptq\qwen3moe_runner_tq33.c` — thêm instrumentation Giai đoạn 1
  (`g_chan_absmax_cur/h`, `stats_update_cur/h`, chế độ CLI `stats`), outlier-fix Giai đoạn 2
  (`quantize_x_int8_protected`, `linear_tq33_add_protected`, `setup_outlier_protect`, cờ
  `g_use_outlier_fix`, chế độ CLI `genfix`/`dumpfix`). Tổng ~180 dòng thêm mới, 0 dòng xóa.
- `e:\Bit-Translate\eval\lowbit_ptq\qwen3moe_runner_tq33.exe` — build lại (0 warning
  `-Wall -Wextra`).

**Tạo mới (scripts)**:
- `prepare_stats_prompt.py` — chuẩn bị prompt hiệu chuẩn 470 token từ `val100_vi.txt`.
- `analyze_outlier_stats.py` — phân tích `outlier_stats.bin`, in bảng tần suất/tỷ lệ outlier.
- `validate_30b_layers_compare.py` — bản sao `validate_30b_layers.py` (không sửa file gốc), so
  rel-err TRƯỚC/SAU cùng 1 oracle.
- `check_routing_vs_oracle.py` — so routing TRƯỚC/SAU với oracle THẬT (không chỉ so nhau).
- `gen_quality_check_fixed.py` — bản sao `gen_quality_check.py` (không sửa file gốc), gọi
  `genfix` thay vì `gen`.

**Data mới (`D:\Bit-Translate-data\tq33_30b\genqa\`)**:
- `stats_prompt.bin`, `outlier_stats.bin` — Giai đoạn 1.
- `runner_dump_layers_fixed.bin` (trong `runner\oracle\`) — dump sau fix, Giai đoạn 2.
- `rel_err_compare.json` — số liệu đầy đủ 48 layer trước/sau.
- `*_prompt_fix.bin`, `*_out_fix.bin`, `results_fixed.json` — Giai đoạn 3.

**Lệnh tái lập**:
```
# build (0 warning)
python -m ziglang cc -O3 -mavx2 -mfma -Wall -Wextra -o qwen3moe_runner_tq33.exe qwen3moe_runner_tq33.c -lm

# Giai đoạn 1 — đo outlier channel
python prepare_stats_prompt.py
./qwen3moe_runner_tq33.exe stats D:\Bit-Translate-data\tq33_30b\runner D:\Bit-Translate-data\tq33_30b\extras.bin D:\Bit-Translate-data\tq33_30b\genqa\stats_prompt.bin D:\Bit-Translate-data\tq33_30b\genqa\outlier_stats.bin
python analyze_outlier_stats.py

# Giai đoạn 2 — dump sau fix + so rel-err
./qwen3moe_runner_tq33.exe dumpfix D:\Bit-Translate-data\tq33_30b\runner D:\Bit-Translate-data\tq33_30b\extras.bin D:\Bit-Translate-data\tq33_30b\runner\oracle D:\Bit-Translate-data\tq33_30b\runner\oracle 20
python validate_30b_layers_compare.py
python check_routing_vs_oracle.py

# Giai đoạn 3 — sinh văn bản thật, so sánh
python gen_quality_check_fixed.py
```
