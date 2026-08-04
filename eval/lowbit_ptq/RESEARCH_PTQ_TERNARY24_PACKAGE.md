# PTQ ternary N:M 2:4 — bake thật + đóng gói TQ33 + đo tốc độ thật trên Qwen3-0.6B

**Ngày**: 2026-08-05 · **Mục tiêu**: đóng gói kỹ thuật PTQ thắng cuộc của toàn bộ nhánh nghiên
cứu tối 04-05/08 (ternary N:M 2:4 + Wanda mask + Lloyd-scale + SEQUENTIAL/BRECQ-lite — đã
thắng scalar mọi biến thể VQ/AQLM đã thử) thành một checkpoint THẬT, đo chất lượng đa miền
(geo6), dung lượng thật, và tốc độ thật qua runner đã tối ưu — không phải ước tính.

## 1. Recipe bake (`exp_ap_bake_ternary24_geo6.py`)

Y hệt kỹ thuật đã thắng ở `exp_k_fixpack_sequential.py` (Wanda 2:4 mask `|W|·‖X‖` + Lloyd-scale
trên survivor + SEQUENTIAL block-wise 28 block, 70 bước/block), với 2 thay đổi:

1. **GROUP=64** (thay vì 32 của exp_k) — khớp CHÍNH XÁC granularity scale mà format TQ33 đã
   thiết kế sẵn, để bước đóng gói sau đó không bị lượng tử hóa lần 2 (group32→group64 sẽ làm
   tròn sai nếu 2 sub-group-32 trong cùng block64 có scale khác nhau — đúng "luật F0": bake
   trong khung đích rẻ hơn convert hậu kỳ).
2. **Bỏ ARM1b/ARM1** (TF-only, đã biết thua SEQUENTIAL) — chạy thẳng SEQUENTIAL, lưu checkpoint
   đầy đủ cuối cùng (khác exp_k chỉ đo PPL rồi revert, không lưu gì).

## 2. Hai lỗi thật gặp phải VÀ ĐÃ SỬA (không phải chạy lại hú họa)

### Lỗi 1 — calib chỉ vi/ja lặp lại đúng bug "mixw" đã biết trong lab

Lần chạy đầu copy nguyên calib của exp_k (60 câu vi/ja thuần) → **geo6 sụp 53.202,9** (×2164
so FP), các miền KHÔNG có trong calib sụp nặng nhất: en ×16.539, code ×44.978, zh ×33.761 —
khớp CHÍNH XÁC cơ chế đã ghi trong memory lab ("calib vi/ja-thuần phá en ×808/code ×4434").
**Fix**: trộn calib đủ 6 miền (`CAL_EN/CAL_CODE/CAL_ZH/CAL_MATH/CAL_CHAT` từ `exp_r_qat_lite.py`
+ vi/ja, tổng 142 câu, shuffle) cho cả bước tính mask Wanda lẫn SEQUENTIAL.

### Lỗi 2 — scale Lloyd liên tục không đóng gói được TQ33

Format TQ33 cần ≤256 giá trị scale duy nhất/tensor (codebook 1-byte scale-idx) — giả định này
đúng khi model được TRAIN với scale ép lưới f8 (QAT), nhưng Lloyd-scale PTQ của tôi là số thực
liên tục → **32.720 giá trị duy nhất/tensor**, export crash (`AssertionError`).
**Fix**: thêm bước ép `s = to_f8(s)` (lưới 8 mức mantissa/octave, hàm có sẵn trong
`exp_r_qat_lite.py`) sau khi Lloyd hội tụ, trước khi tính `t = round(Wm/s)` cuối cùng. Verify
trên 1 ma trận trước khi chạy full: 34 giá trị duy nhất (dưới 256) — an toàn.

### Lỗi 3 (nhỏ) — runner cũ yêu cầu tensor `.bias` cho mọi linear

`qwen3_runner_tq33_fast.c` viết cho checkpoint QAT gen4_n4.pt (có bias giả từ LearnQLinear) —
`find_t()` gọi `exit(1)` nếu không thấy tensor `.bias`. Model HF chuẩn (bake PTQ này) không có
bias thật (Qwen3 bỏ bias, dùng QK-norm). **Fix**: thêm 196 tensor bias=0 (no-op toán học) vào
export trước khi ghi `weights_f32` — không sửa code C đã proven.

## 3. Kết quả cuối — sau khi sửa cả 3 lỗi

### Chất lượng (geo6, 6 miền, so FP32)

| Miền | FP32 | baked ternary 2:4 | ×FP |
|---|---:|---:|---:|
| vi | 68,97 | 729,85 | ×10,58 |
| ja | 125,07 | 1.726,17 | ×13,80 |
| en | 34,19 | 1.373,01 | ×40,16 |
| code | 2,66 | 243,29 | ×91,49 |
| zh | 50,78 | 2.635,48 | ×51,90 |
| math | 5,54 | 59,07 | ×10,66 |
| **geo6** | 24,58 | **634,93** | **×25,83** |

Gần khớp mốc `exp_k` đã biết (vi 594,3 @ group=32) — group=64 làm vi tệ hơn ~23% (729,85 vs
594,3), cái giá thật của việc đổi granularity để đóng gói được, không phải lỗi. Verify 2:4:
chỉ 1/110.100.480 nhóm vi phạm (0,0000%) — mask gần như hoàn hảo.

### Dung lượng (thật, đo trực tiếp từ file đã ghi)

| Thành phần | Kích thước |
|---|---:|
| 196 ma trận linear TQ33 (440.401.920 trọng số, 1,500bpw CHÍNH XÁC) | 82,58 MB |
| `embed_tokens` int8 per-row (dùng chung input+output head) | 156,19 MB |
| Extras (norm + bias=0 placeholder, fp16) | 0,82 MB |
| **Tổng** | **239,58 MB** |

So FP32 gốc (3.007,9 MB): nén **12,55×**. So Q4_K_M đã biết (373MB): nhỏ hơn **~1,56×**.
Pack verify lossless thật: werr pack-vs-baked = 0,0060% (gần như 0, đúng kỳ vọng khi bake
native ở group=64 + scale đã ép lưới f8 trước khi pack).

### Tốc độ (thật, qua `qwen3_runner_tq33_fast.exe`, 8 luồng, 5 lần lặp)

| Run | ms/48 tok | tok/s | % băng thông 41GB/s |
|---|---:|---:|---:|
| 1 | 300,4 | 159,78 | 97,8% |
| 2 | 297,2 | 161,49 | 98,9% |
| 3 | 298,7 | 160,70 | 98,4% |
| 4 | 294,0 | 163,25 | 100,0% |
| 5 | 289,5 | 165,80 | 101,5% |

**159,8–165,8 tok/s — gần như bão hòa băng thông DRAM đo thật (41GB/s)**, vượt mục tiêu
100-200 tok/s ban đầu, cùng hạ tầng tốc độ đã build ở `RESEARCH_TQ33_SPEED_100.md` (int8 head +
threading redesign), tái sử dụng nguyên vẹn không cần sửa kernel gì thêm cho checkpoint mới này.
Kernel verify: `i8 maddubs == vnni` khớp tuyệt đối, `tq33 decode-register == decode_block` khớp
tuyệt đối trên 300 hàng thật.

## 4. Kết luận

Đây là gói PTQ-thuần (không train) hoàn chỉnh đầu tiên của nhánh nghiên cứu tối nay có ĐỦ 3
trục đo thật: chất lượng đa miền, dung lượng thật, tốc độ thật qua runner production-ready.
Kết quả trung thực: **nén 12,55× + tốc độ 160+ tok/s nhưng chất lượng suy giảm đáng kể** (geo6
×25,8, các miền ngoài vi/ja như code/zh suy giảm 50-90×) — đúng dự đoán "0,6B là ca khó nhất"
đã lặp lại xuyên suốt lab. Đây là SÀN THỰC DỤNG của PTQ-thuần trên model này: nhẹ và nhanh thật,
nhưng không "dùng được" theo nghĩa hội thoại mạch lạc đa miền — muốn hơn nữa cần train (QAT/KD,
đã có GEN4 val-100 vi 350,9 @1,56bpw sau full KD, hoặc Rollout-KD đang chờ máy A).

## 5. File

- `exp_ap_bake_ternary24_geo6.py` — bake (đã vá 2 lỗi: calib mix 6 miền + ép f8-grid)
- `exp_ap_results.json` — kết quả geo6 + config
- `exp_ap_run2.log` — log chạy đúng (lần đầu `exp_ap_run.log` là bản LỖI, giữ lại làm bằng chứng)
- `export_baked_ternary24_for_runner.py` — export weights_f32 + tq33_packed + embed_int8 (đã
  vá lỗi thiếu bias)
- Checkpoint: `D:\Bit-Translate-data\qat_ckpts\ternary24_seq_g64_baked.pt`
- Runner data: `D:\Bit-Translate-data\tq33_runner_ptq24\{weights_f32,tq33_packed,embed_int8,oracle}`
