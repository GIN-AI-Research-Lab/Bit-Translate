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
3. **Nén embed+lm_head xuống int8** (lever rẻ tiếp theo): 3% param nhưng 16% tổng bit; int8
   đã chứng minh chịu được ở bản 0.6B (RESEARCH_TQ33_SPEED_100). Kỳ vọng overall 3,06→~2,8bpw
   gần như không mất chất lượng. → config gần-tối-ưu đầy đủ.
4. **Đóng gói + kernel** để đo tok/s THẬT (mọi thứ tới giờ là chất lượng; tốc độ chưa đổi vì
   chưa packed — xem `RESEARCH_MOE_SPEED_PTQ.md`). Config đích: down=TQ33(ternary) +
   gate_up=int3-packed + attn=int4-packed + embed/head=int8.
5. **Scale lên model to** (theo lộ trình): kỳ vọng ngưỡng bit còn dễ thở hơn (định luật
   kích thước — model to chịu nén tốt hơn; contextual sparsity/extreme-quant đều mạnh hơn ở 7B+).
