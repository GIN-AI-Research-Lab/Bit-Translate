# TỔNG KẾT NGHIỆM THU V8 — soup / averaging / liều-lượng / bench 200b i2s

> 2026-08-09→10, máy B. Tiếp nối `TONGKET_V8.md` (train + nghiệm thu vòng đầu).
> Mọi số dưới đây là chấm mù A/B cùng-phiên (randomize + key riêng, giải mã độc lập,
> noise-check bằng cặp câu giống hệt). Artifact: `eval/v8_grade/nghiemthu/`.
> Thước so sánh liên phiên DUY NHẤT: cột **bad** (thang judge trôi tới 8-9 điểm giữa phiên,
> v7a err64 good=9 phiên gốc vs 0 phiên sau — chỉ so DELTA trong cùng phiên).

## 1. Ba đường TONGKET_V8 §4 — kết quả đo

### (soup) v7a_avg ⊕ v8 — CHẾT: cú sửa KD giòn, nội suy mức nào cũng phá

| cùng-phiên vs v8 | err64 bad (soup vs v8) | rand150 bad |
|---|---|---|
| α=0,3 | 22 vs 12 (**+10**) | 1 vs 6 (−5) |
| α=0,7 | 26 vs 17 (+9) | 5 vs 7 (−2) |

- α=0,7 gần v8 hơn nhiều mà mất cú sửa Y NHƯ α=0,3 ⇒ cú sửa chỉ tồn tại trọn ở α=1.
- Vùng giữa sinh câu VỠ MỚI không có ở cả hai đầu (十分→"một phút", 若年層の流出→"rò rỉ dữ liệu").
- α=0,5 nhiễu loạn cả hai trục; α=0,7 bị α=0,3 đè. **Đừng quét thêm alpha.**
- α=0,3 giữ làm phương án zero-regression (rand bad 1 < v7a 2) nếu cần an toàn tuyệt đối.

### (v8_avg) checkpoint-averaging 4 mốc 14000-16750 — ứng viên tốt nhất nhánh ×8

- vs v8-raw cùng phiên: err64 bad **17<19**, rand150 bad **4<7** — thắng cả hai trục.
- vs v7a trực tiếp (rand150): 30–38, bad 6 vs 3 → thuế net **+3** ≈ v8-raw. Averaging
  khử NHIỄU cuối train, KHÔNG khử DRIFT (drift nằm trong cả 4 mốc).

### (retrain ×3) hạ liều KD ×8→×3 — BÁC GIẢ THUYẾT LIỀU (thí nghiệm một-biến, ~$8)

Cách làm sạch: cắt đuôi bin ×8 thành ×3 ngay trên volume (`cloud/modal_rebin_v8x3.py`,
kiểm 13.100 nhóm ×8 nguyên vẹn từng byte) → cùng xuất phát v7a_avg, cùng 8000 step, cùng LR.

| liều | err64 bad | thuế phổ thông net vs v7a |
|---|---:|---:|
| ×0 (v7a) | 37-40 | — |
| ×3 (v8x3_avg) | **27** | **+2** |
| ×8 (v8_avg) | **17-19** | **+2..+3** |

**Cú sửa tỉ lệ thuận liều; thuế KHÔNG theo liều** (trùng cả danh sách câu: 気力/文化予算/
お義母さん). Thuế đến từ SỰ CÓ MẶT của KD + LR-restart 8000 step. **×8 đè ×3 — đừng thử ×5.**
Checkpoint ×3: Modal `thaovyh2t:vija-v8-vol/checkpoints_v8x3/` (giữ làm bằng chứng).

## 2. Bench 200b bằng bản i2s THẬT (deploy path, cùng phiên, noise=0)

Pipeline i2s sạch tuyệt đối: convert (converter git-HEAD) + `llama-quantize I2_S 1` +
llama-server + token ids → **sanity 7/7 trùng PyTorch từng ký tự** (cả v8_avg lẫn v7a
tái tạo). 77,56MB, 200 câu/19s.

| v8avg-i2s vs v7a-i2s | đối đầu | v8avg bad | v7a bad |
|---|---|---:|---:|
| Toàn bộ 200 | 31 – 44 (hoà 125) | 24 | 21 |
| Câu ngắn (100) | 9 – 10 | **6** | 8 |
| **Câu dài (100)** | **22 – 34** | **18** | 13 |
| Câu khó (68) | 14 – 20 | 12 | 10 |

Chênh chưa đạt ý nghĩa thống kê (31-44: p≈0,13; dài: p≈0,11) nhưng chiều nhất quán:
**cú sửa KD không lan ra ngoài tập nhắm; thuế boundary-shift thì lan, cắn mạnh nhất câu dài**
(13.100 câu KD toàn câu ngắn, không có replay dài đối trọng).

## 3. Quyết định (user chốt 2026-08-10)

1. **v7a vẫn là bản deploy chung.** Giữ CẢ HAI bản i2s trên release `v8-avg-step16750`
   (v8_avg_i2s + v7a_avg_i2s rollback) — đổi qua lại chỉ là đổi đường dẫn model.
2. **Bỏ vòng lặp "bench random → thấy lỗi → KD điểm"** — không hội tụ: taxonomy 74% lỗi
   là lớp năng lực (cấu trúc/vai), mỗi vòng chỉ tái phát hiện cùng họ lỗi + trả thuế ~2-3 câu.
3. **Suite hồi quy TÍCH LŨY** thay bench random làm thước chính: `eval/regression_suite_v1.jsonl`
   (114 câu, 79 point / 35 structure, 58 câu bad ≥2 hệ; gộp err64 + thuế rand150 + lõi 200b
   — 5/10 câu lõi chính là ISSUES #5 CHƯA TỪNG được nhắm). Mọi ứng viên deploy chấm trên
   suite + 200b + rand150; pass-rate suite phải đơn điệu tăng. Family gán bằng heuristic —
   REVIEW TAY trước khi dùng làm target KD.
4. **V9 = một vòng 18L cuối, công thức chống-thuế**: KD nhắm toàn suite (point-family) +
   **anchor-rehearsal** (thầy dịch câu-hàng-xóm model đang đúng, ghim ranh giới) + **replay
   câu dài** + **L2-SP về v7a_avg** + LR-restart hạ 5e-5→2e-5. Thắng → công thức xác lập.
   Vẫn <2 điểm trên 200b → **hard-cap luật grow kích hoạt đúng luật** (PLAN_V8, user duyệt
   30/7: 2 vòng 18L liên tiếp <2 điểm) → grow 24L + ctx384.
5. Đã bác, đừng lặp: soup mọi alpha, chỉnh liều, hậu kỳ; beam/rerank; vá tầng app (30/7).

## 4. Artifact

| gì | đâu |
|---|---|
| Release: v8_avg.pt + 2 bản i2s | `v8-avg-step16750` (GitHub) |
| Verdict/key/decoded 8 phiên chấm | `eval/v8_grade/nghiemthu/` |
| Bản dịch: soup×3, x3avg, v8avg (1199), v8avg-i2s (200b) | `eval/v8_grade/nghiemthu/` |
| Suite | `eval/regression_suite_v1.jsonl` (+ `build_suite.py`) |
| Script rebin/avg trên volume | `cloud/modal_rebin_v8x3.py`, `cloud/modal_avg_v8x3.py` |
| Checkpoint ×3 (bằng chứng âm tính) | Modal `thaovyh2t:vija-v8-vol/checkpoints_v8x3/` |
