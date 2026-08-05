# Nén model có sẵn theo ĐỘ NHẠY: down=ternary + gate_up=int3 (điểm ngọt PTQ thuần)

**Ngày**: 2026-08-05 · **Model**: OLMoE-1B-7B (MoE) + Qwen3-0.6B (dense) · **Chi phí**: $0
(CPU máy A) · **Ràng buộc**: PTQ thuần (naive quantize, không reconstruction/train) trừ mục 4.

> Hướng dự án (chốt 2026-08-05): **nén + tăng tốc model CÓ SẴN**, chứng minh trên 0.6B rồi
> scale lên model to. KHÔNG còn nhắm model dịch VI-JA cũ. Tài liệu này là nền tảng phương pháp.

## 0. Tóm tắt (đọc trước)

Chuỗi thí nghiệm exp_ak→bc trả lời dứt điểm 3 câu hỏi, tất cả bằng PPL/battery THẬT trên model
đầy đủ (không phải sai-số-tái-tạo tensor cô lập — bài học đắt của exp_ad→aj, xem
`RESEARCH_AQLM_CODEBOOK_PTQ.md` §5c):

1. **Nén đầy đủ NAIVE thì HỎNG, và thủ phạm là gate_up_proj** (exp_az): down-only ×1,35 →
   thêm gate_up ×1700 → thêm attention ×5715. gate_up nhạy vì nó là ĐẦU VÀO của phi tuyến
   SiLU + phép nhân gate×up (sai số bị khuếch đại phi tuyến/nhân), còn down là đầu ra tuyến
   tính được residual bảo vệ.
2. **Sequential reconstruction CỨU được** collapse đó (exp_bb): naive ×98757 → sequential
   ×476 trên Qwen3-0.6B, nhờ 3 ingredient bắt buộc (scale HỌC ĐƯỢC + 2 pass + norm co-tune).
   Nhưng "cứu" = sống-lại-nhưng-suy-giảm (còn ×8,6 kể cả recipe đủ của lab = PPL 594), KHÔNG
   phải khôi phục chất lượng.
3. **KHÔNG CẦN sequential — chỉ cần PHÂN BỔ BIT THEO ĐỘ NHẠY** (exp_bc): đây là kết quả thực
   dụng mạnh nhất. Xem mục 2.

## 1. exp_az — ablation nén tăng dần (OLMoE, PPL baseline Q4_K_M = 8,616)

| Stage | PPL | ×base | Battery |
|---|---:|---:|---:|
| baseline | 8,62 | ×1 | 16/17 |
| +down_proj ternary | 11,60 | ×1,35 | 15/17 |
| +gate_up_proj ternary (toàn expert) | 14.649 | **×1700** | 0/17 |
| +attention ternary | 49.239 | ×5715 | 0/17 |

Thủ phạm cô lập chính xác: **gate_up_proj** (nén nó là sụp; nén down thì không).

## 2. exp_bc — PHÂN BỔ BIT THEO ĐỘ NHẠY (kết quả chính, PTQ thuần, deploy được ngay)

down=ternary 2:4 CỐ ĐỊNH (đã chứng minh bền), quét gate_up qua các mức bit:

| gate_up | PPL | ×base | expert bpw |
|---|---:|---:|---:|
| fp16 | 11,60 | ×1,35 | 11,19 |
| int8 | 11,58 | ×1,34 | 5,94 |
| int4 | 11,79 | ×1,37 | 3,27 |
| **int3** | **11,85** | **×1,38** | **2,60 ← ĐIỂM NGỌT** |
| int2 | 229,8 | ×26,67 | 1,94 |
| ternary | 14.649 | ×1700 | 1,56 |

**Vách đá cực rõ giữa int3 và int2**: từ fp16→int3 chất lượng gần phẳng (×1,34→1,38) trong khi
bpw giảm 4,3×; rớt vực ở int2 (×27) và ternary (×1700).

**Kết luận thực dụng**: `down=ternary + gate_up=int3` → khối expert **2,60bpw, chất lượng
×1,38, KHÔNG train/reconstruction**. Đánh bại nén đồng đều: uniform-ternary 1,56bpw sụp ×1700;
phân-bổ-theo-độ-nhạy 2,60bpw gần như nguyên vẹn — chênh 1bpw đổi lấy ~1230× chất lượng. Đây là
config PTQ-thuần deploy được ngay, và là nền cho hướng "nén model có sẵn".

## 2b. exp_bd — quét attention trên nền down=ternary+gate_up=int3 (hoàn thiện phân bổ)

Trên nền điểm ngọt (down=ternary, gate_up=int3), quét attention q/k/v/o + đo BATTERY:

| attn | PPL | ×base | battery | attn bpw |
|---|---:|---:|---:|---:|
| fp16 | 11,9 | ×1,38 | 9/12 | 16,0 |
| int8 | 11,8 | ×1,37 | 10/12 | 8,12 |
| **int4** | **12,1** | **×1,40** | **9/12** | **4,12 ← điểm ngọt attn** |
| int3 | 15,4 | ×1,79 | 8/12 | 3,12 |
| int2 | 82.918 | ×9623 | 0/12 | 2,12 |
| ternary | 7.520 | ×873 | 0/12 | 1,56 |

**Attention nhạy hơn gate_up ĐÚNG MỘT BẬC**: gate_up êm tới int3, attention chỉ êm tới int4
(int3 đã tụt ×1,79). Xác nhận trực giác lab "attention nhạy nhất".

### BẢNG XẾP HẠNG ĐỘ NHẠY HOÀN CHỈNH (đo thật, xây dần cả chuỗi)
| Thành phần | Mức bit thấp nhất còn ổn | Cơ chế |
|---|---|---|
| **down_proj** (bền nhất) | ternary ~1,56bpw | đầu ra tuyến tính, residual bảo vệ |
| **gate_up_proj** (nhạy vừa) | int3 ~3,1bpw | đầu vào SiLU + phép nhân |
| **attention q/k/v/o** (nhạy nhất) | int4 ~4,1bpw | softmax + quyết định "nhìn đâu", lỗi rời rạc |

### CONFIG PTQ-THUẦN ĐẦY ĐỦ (deploy được ngay, KHÔNG train)
`down=ternary + gate_up=int3 + attention=int4` → **PPL ×1,40, battery 9/12, overall ~3,06bpw**.
Xác nhận bằng CẢ PPL lẫn battery (không chỉ 1 thước). Embed/lm_head/router/norm giữ fp16.

**Phát hiện phụ (lever tiếp theo)**: embed+lm_head giữ fp16 = 3% tham số nhưng ~16% tổng bit.
Nén chúng (int8, biết chịu được từ bản 0.6B TQ33 speed) → overall ~2,8bpw. Việc kế §6.

## 2c. exp_be — quét embed+lm_head (hoàn thiện nén toàn model)

Trên nền down=ternary+gate_up=int3+attn=int4, quét embed_tokens+lm_head + battery + tổng GB:

| emb+head | PPL | ×base | battery | TỔNG GB | overall bpw |
|---|---:|---:|---:|---:|---:|
| fp16 | 12,1 | ×1,40 | 9/12 | 2,65 | 3,06 |
| int8 | 12,1 | ×1,41 | 10/12 | 2,45 | 2,83 |
| **int4** | **13,2** | **×1,53** | **11/12** | **2,34** | **2,71 ← điểm ngọt** |
| int3 | 14,2 | ×1,65 | 11/12 | 2,32 | 2,68 |
| int2 | 398 | ×46 | 2/12 | 2,29 | 2,65 |
| ternary | 40,7 | ×4,72 | 6/12 | 2,28 | 2,63 |

- **Lợi nhuận giảm dần rõ rệt**: dưới int4, GB tiết kiệm không đáng kể (2,34→2,28, chênh
  0,06GB) mà rủi ro sụp (int2 ×46). KHÔNG đi dưới int4 cho embed/head. int8 = an toàn.
- **int2 tệ hơn ternary** (398 vs 40,7) — vùng chết, ternary có mức 0, int2 không (bài học
  nhất quán). Đừng nội suy trong vùng chết.

### 🏁 CONFIG NÉN TOÀN MODEL ĐẦY ĐỦ (PTQ THUẦN, KHÔNG TRAIN) — kết quả cuối
`down=ternary + gate_up=int3 + attn=int4 + embed/head=int4`
= **2,34GB (~2,71bpw), PPL ×1,53, battery 11/12** (baseline 12/12).
**13,84GB (fp16 gốc) → 2,34GB = 5,9× nhỏ hơn**, giữ 11/12 năng lực. Bản an toàn (embed/head
=int8): 2,45GB = 5,6×, ×1,41, 10/12. Tất cả bằng phân-bổ-bit-theo-độ-nhạy, đo bằng CẢ PPL
lẫn battery, KHÔNG reconstruction/train.

## 2d. exp_bf — KIỂM CHỨNG CHUYỂN GIAO (Qwen3-0.6B dense vs OLMoE MoE)

Câu hỏi: thứ hạng + config tìm trên OLMoE có chuyển giao sang kiến trúc KHÁC HẲN không?
Test trên Qwen3-0.6B (dense, GQA 16q/8kv, tied embedding — khác OLMoE cả 3 mặt).

**Ranking probe (mỗi thành phần ternary riêng lẻ, ×baseline PPL):**
| | Qwen3-0.6B | OLMoE |
|---|---:|---|
| down | ×51 (nhẹ nhất) | ×1,35 (nhẹ nhất) |
| gate_up | ×335 | (giữa) |
| attention | ×23.003 (nặng nhất) | (nặng nhất) |

**→ THỨ HẠNG CHUYỂN GIAO HOÀN TOÀN** (down < gate_up < attention), trên kiến trúc khác hẳn →
bắt nguồn từ VAI TRÒ thành phần, không phải model cụ thể. Đây là kiến thức mang đi được.

**→ NGƯỠNG KHÔNG chuyển giao**: config thắng OLMoE (×1,53) áp thẳng Qwen3-0.6B → **×492 (vỡ)**.

**Giải thích + hệ quả cho lộ trình (quan trọng):**
- OLMoE nén tốt (down=ternary chỉ ×1,35) vì là **MoE 7B tổng** — dư thừa lớn (7B params +
  expert dư thừa lẫn nhau) dù 1B active/token. Qwen3-0.6B dense chỉ 0.6B → mỗi trọng số quan
  trọng hơn → nhạy hơn cả trăm-nghìn lần. Qwen3-0.6B là **ca KHÓ NHẤT** (nhỏ nhất + dense).
- **Hệ quả THUẬN cho scale-up**: model TO hơn / MoE nhiều dư thừa hơn sẽ nén CÒN TỐT HƠN
  OLMoE, không tệ hơn. Lộ trình "0.6B/1B → model to" đi đúng chiều: OLMoE (nén tốt) là bằng
  chứng, Qwen3-0.6B (nén kém) chỉ là sàn dưới của định luật kích thước.
- **Quy trình chuẩn cho model mới**: KHÔNG copy config. Bắt đầu từ thứ hạng đã biết
  (down<gate_up<attention<lm_head), chạy sweep (exp_bc/bd/be, script sẵn) đo NGƯỠNG riêng.

## 2e. ⚠️ ĐO THẬT bằng llama.cpp + SO CÔNG BẰNG vs Q2_K — PHÁN QUYẾT QUAN TRỌNG (exp_bg/bh)

Dùng llama.cpp (kernel CPU thật, có sẵn ở `F:/Project Ai/teams-caption-translator/bin/`) đo
tok/s + RAM + PPL thật cho OLMoE ở các quant chuẩn, so với config sensitivity tự chế của ta:

| | bpw | tok/s (thật) | RAM/file | PPL ratio vs Q4 (cùng text) |
|---|---:|---:|---:|---:|
| Q4_K_M | ~4,5 | 39,9 | 3,92 GiB | ×1,00 (mốc) |
| Q3_K_M | ~3,3 | 48,0 | 3,11 GiB | ×1,015 |
| **Q2_K (off-the-shelf)** | **2,6** | **56,6** | **2,39 GiB** | **×1,12** |
| **Config tự chế của ta** | 2,71 | (cần tự viết kernel) | 2,35GB (file) | **×1,57** |

(Ratio đo công bằng: cùng text ppl_big, tỷ lệ vs Q4 TRONG CÙNG framework — llama.cpp cho quant
chuẩn, PyTorch cho config ta. exp_bh.)

**XÁC NHẬN CÙNG FRAMEWORK (exp_bi, bỏ nhiễu framework)**: dequant Q2_K ra fp16 rồi đo PPL PyTorch CÙNG text/baseline với config ta → Q2_K **×1,20** vs config ta **×1,57** (baseline Q4-dequant 5,01; Q2_K 6,03; ta 7,89). Q2_K tốt hơn ~1,3× ở gần cùng bpw — kết luận KHÔNG phải artifact framework, giữ nguyên.

**PHÁN QUYẾT: Q2_K THẮNG config tự chế trên MỌI trục** — ít bit hơn (2,6 vs 2,71), chất lượng
tốt hơn nhiều (×1,12 vs ×1,57), có kernel nhanh sẵn (56 tok/s). **→ KHÔNG đáng tự viết kernel
cho sơ đồ naive của ta trên OLMoE. Muốn deploy OLMoE nén: DÙNG THẲNG Q2_K của llama.cpp.**

**Vì sao Q2_K thắng (bài học thật, quan trọng cho cả hướng "nén model có sẵn"):**
- Q2_K KHÔNG phải int2 naive — nó là K-quant tinh vi: super-block, scale + MIN bất đối xứng,
  imatrix-tunable, VÀ tự phân bổ bit theo vai trò tensor. Ý tưởng "phân bổ theo độ nhạy" của
  ta ĐÚNG nhưng llama.cpp ĐÃ làm rồi ở dạng chín muồi hơn.
- Bộ lượng tử hóa từng thành phần của ta THÔ (per-group symmetric, không imatrix, không min
  bất đối xứng) → thua máy K-quant.
- **Giá trị nghiên cứu của cả chuỗi KHÔNG phải "đánh bại llama.cpp"** — mà là HIỂU CƠ CHẾ
  (thứ hạng độ nhạy, vì sao gate_up vỡ, sequential cứu ra sao, chuyển giao thế nào). Là
  artifact deploy thì off-the-shelf Q2_K thắng.
- **Muốn THẬT SỰ thắng Q2_K**: (a) ghép phân-bổ-theo-độ-nhạy + bộ lượng tử hóa CHÍN (imatrix
  + asymmetric, tức nâng cấp máy quant của ta lên ngang K-quant) rồi mới thêm allocation; HOẶC
  (b) xuống dưới 2,6bpw nơi mọi quant llama.cpp sụp — nhưng ta đã chứng minh sub-2bit sụp.
  Cả hai đều là việc lớn; Q2_K đã rất tốt nên bar cao.

## 3. Vì sao gate_up nhạy hơn down (cơ chế)

`FFN(x) = down( SiLU(gate(x)) × up(x) )`.
- **down = đầu ra**: sai số cộng thẳng vào residual (tuyến tính, hiền, residual pha loãng).
- **gate/up = đầu vào phi tuyến + nhân**: sai số phải qua SiLU (gần 0 = cổng bật/tắt, dễ lật
  trạng thái rời rạc) + phép nhân gate×up (sai số × sai số, nhân chứ không cộng). → khuếch đại.
- Cộng dồn qua độ sâu là CẤP SỐ NHÂN: (1+δ)^L. Nén 1/3 (down) δ nhỏ, dưới ngưỡng → ×1,35;
  nén thêm gate_up δ vượt ngưỡng tới hạn → nổ ×1700. Giải thích vì sao KHÔNG tuyến tính.

## 4. exp_bb — sequential reconstruction (bằng chứng cơ chế, Qwen3-0.6B)

Naive ternary TẤT CẢ ma trận → ×98757 (sụp). Sequential (block-wise, mỗi block bù sai số
block trước) với recipe đúng → ×476 (sống lại). 3 ingredient bắt buộc (khớp lab Bài 4/10):
scale HỌC ĐƯỢC (softplus, grad qua t×s), 2 PASS, norm co-tune. MSE mỗi block pass 2 giảm
~16-45× so pass 1 (block 7: 162→3,6). Bản exp_ba thiếu 3 cái này → tệ hơn cả naive (đúng
failure "e1>e0" lab cảnh báo). Lab recipe ĐỦ đạt 594 @1,94bpw (còn thiếu gauge/fix-pack ở đây).

## 5. exp_aw/ax/ay — contextual sparsity (âm tính có ích, Qwen3-0.6B)

Cấu trúc thưa TỒN TẠI (23% neuron đủ 95% năng lượng, contextual mạnh) nhưng trần khai thác
YẾU: oracle top-k giữ 50% neuron → PPL ×1,16; 30% → ×1,62; 23% → ×2,06. Importance "đúng"
(|x|×‖down_col‖) KHÔNG cứu (tệ hơn). Kém xa quantization về byte/chất lượng. Đúng "định luật
kích thước": contextual sparsity mạnh ở 7B+, yếu ở 0.6B.

## 6. Việc kế (hướng nén model có sẵn)

1. ~~Xác nhận config điểm ngọt bằng battery~~ ✅ XONG (exp_bd: 9/12 battery cho config đầy đủ).
2. ~~Quét attention~~ ✅ XONG (exp_bd: attention êm tới int4, nhạy hơn gate_up 1 bậc).
3. ~~Nén embed+lm_head~~ ✅ XONG (exp_be: int4 là điểm ngọt, dưới int4 vô ích + sụp). Config
   đầy đủ chốt: 2,34GB (5,9× fp16), ×1,53, 11/12 battery.
4. **Đóng gói + kernel** để đo tok/s THẬT (mọi thứ tới giờ là chất lượng; tốc độ chưa đổi vì
   chưa packed — xem `RESEARCH_MOE_SPEED_PTQ.md`). Config đích: down=TQ33(ternary) +
   gate_up=int3-packed + attn=int4-packed + embed/head=int4/int8. Đây là việc lớn kế tiếp.
5. **Scale lên model to** (theo lộ trình): chạy LẠI sweep (exp_bc/bd/be — script sẵn) trên
   model to; kỳ vọng ngưỡng bit còn dễ thở hơn (định luật kích thước — model to chịu nén tốt
   hơn). CHỈ cần đo lại NGƯỠNG; THỨ HẠNG độ nhạy (down<gate_up<attention<lm_head) chuyển giao.

## 2f. ⚠️ SCALE-UP THẬT: Qwen3-30B-A3B — mix-theo-độ-nhạy THUA off-the-shelf Q2_K (exp llama.cpp)

Áp ranking độ nhạy lên 30B-A3B qua `llama-quantize --tensor-type` (requantize từ Q4_K_M vì
mix ≤Q4; PyTorch sweep bất khả vì 30B fp16 ~60GB > RAM 34GB). Mix: down=Q2, gate_up=Q3,
attn=Q4, lm_head=Q6, embed=Q3. Đo THẬT bằng llama.cpp trên máy desktop này (6 luồng, cùng
text ppl_big):

| Bản | Size | bpw | tok/s | PPL | ×Q4 |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 17,28 GiB | ~4,5 | 7,0 | 2,124 | ×1,00 |
| **SENS (mix của ta)** | 11,57 GiB | 3,26 | 18,4 | 2,282 | ×1,074 |
| **Q2_K (off-the-shelf)** | 10,48 GiB | 2,6 | 21,3 | **2,169** | **×1,02** |

**PHÁN QUYẾT: Q2_K Pareto-dominate mix của ta — nhỏ hơn + nhanh hơn + PPL tốt hơn**, dù ta
dùng nhiều bit hơn (3,26 vs 2,6). Ba nguyên nhân:
1. **30B MoE nén cực tốt off-the-shelf**: Q2_K ×1,02 so Q4 = gần lossless → gần hết dư địa.
2. **Ta ép down=Q2 nhiều khả năng SAI**: Q2_K/Q4_K_M đều BẢO VỆ down (Q3/Q6). "down bền" (đo
   ternary-riêng trên OLMoE) CÓ THỂ không chuyển giao sang Qwen-MoE — llama.cpp bảo vệ down
   xuyên model là có lý do.
3. **Requantize từ Q4 (không f16)**: down Q4→Q2 double-quant, tệ hơn f16→Q2 của Q2_K.

**Định luật kích thước theo chiều BẤT LỢI cho custom scheme**: model càng to (OLMoE Q2_K ×1,12
→ 30B Q2_K ×1,02), off-the-shelf Q2_K càng gần hoàn hảo → custom scheme càng ít chỗ thêm giá
trị. **Kết luận thực dụng cuối: nén model to có sẵn để deploy → dùng thẳng Q2_K/K-quant
llama.cpp. Custom PTQ tự chế KHÔNG thắng, càng lên model to càng thua rõ.** (30B chạy thật
trên máy desktop 34GB RAM: Q2_K 10,5GB, 21 tok/s — dùng được.)

## 2g. CỰC HẠN 30B: ép experts xuống sàn Q1_0 (~1bit) — SỤP THẢM, xác nhận tường rate-distortion

Thử ép phần nặng (experts = 93% params) xuống SÀN llama.cpp (Q1_0 1.125bpw), attn/lm_head bảo
vệ. ("0.3bit" bất khả: llama.cpp sàn 1.125bpw; ternary-1:32 tự chế cần PyTorch load 60GB > RAM.)

| Bản | Size | bpw | tok/s | PPL | ×Q4 |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 17,28 GiB | ~4,5 | 7,0 | 2,124 | ×1,00 |
| SENS | 11,57 GiB | 3,26 | 18,4 | 2,282 | ×1,07 |
| Q2_K | 10,48 GiB | 2,6 | 21,3 | 2,169 | ×1,02 |
| **FLOOR (experts Q1_0)** | 4,72 GiB | 1,33 | 20,8 | **11.340** | **×5.340 💀** |

**KẾT LUẬN**: NGAY CẢ trên 30B siêu-dư-thừa (Q2_K ×1,02 gần lossless), ép experts xuống ~1bit
→ PPL sụp ×5.340. **Sàn dùng được ≈ 2,6bpw (Q2_K); dưới đó chết, không vùng xám.** VÀ FLOOR
4,72GB KHÔNG nhanh hơn Q2_K 10,48GB (20,8 vs 21,3 tok/s) — nén dưới Q2_K MẤT chất lượng mà
KHÔNG được tốc độ (Q1_0 decode nặng compute + vẫn memory-bound). Q2_K là điểm ngọt/sàn thực
dụng dứt điểm. Tường rate-distortion xác nhận toàn dải 0.6B→30B.
