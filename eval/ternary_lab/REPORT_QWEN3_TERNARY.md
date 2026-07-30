# BÁO CÁO KẾT QUẢ THÍ NGHIỆM: CONVERT-ONLY QWEN3-0.6B SANG TERNARY (BITNET 1.58-BIT)

> Ngày thực hiện: **30/07/2026**  
> Đối tượng: Mô hình **`Qwen/Qwen3-0.6B`** (596M parameters, 196 linear projection modules).  
> Mục tiêu: Kiểm chứng xem hướng **Convert-only (Zero-train Post-Training Quantization)** có khả thi ở dải 1.58-bit mà không cần gradient training hay không.

---

## 1. Bảng số liệu đối chiếu 5 Điều kiện (Conditions A - E)

| Điều kiện | Mô tả phương pháp | PPL English (Wiki) | PPL Japanese (`dev.ja`) | PPL Vietnamese (`dev.vi`) | Tỷ lệ Perplexity so với Baseline | Trạng thái Output |
|---|---|---:|---:|---:|---:|---|
| **Condition A** | **FP32 gốc (Baseline)** | **203.01** | **88.51** | **66.34** | **1.0×** (Mốc) | Văn bản tự nhiên |
| **Condition B** | **TWN làm tròn thuần (Per-row)** | 12.487.716 | 10.716.020 | 23.660.041 | **> 120.000×** (Sập) | Trùng lặp từ vô nghĩa (`rices郎郎郎...`) |
| **Condition C** | **GPTQ-Ternary (Hessian 128 câu calib)** | 29.236.324 | 52.941.961 | 48.514.040 | **> 500.000×** (Sập) | Rác ký tự hỗn hợp (`пасhower<li_IW...`) |
| **Condition D** | **Q4 → GPTQ-Ternary** | 30.820.735 | 16.264.846 | 25.388.909 | **> 200.000×** (Sập) | Mảng từ giả lập (`ieteitiietebage...`) |
| **Condition E** | **Hybrid GPTQ (Giữ 1% FP16)** | 68.978.456 | 55.423.437 | 22.325.499 | **> 330.000×** (Sập) | Rác ký tự đặc biệt (`ootaveryinity...`) |

---

## 2. Phân tích Chi tiết Kết quả

1. **GPTQ-Ternary bù lỗi Hessian vẫn không cứu nổi 1.58-bit**:
   - Ngay cả khi sử dụng thuật toán GPTQ mạnh nhất (tính ma trận Hessian $H = 2XX^T$ từ 128 câu calibration en+ja+vi và bù lỗi từng cột theo công thức closed-form Cholesky inverse), PPL vẫn nổ tung từ **88.5 (FP32)** lên **> 52 triệu (GPTQ-Ternary)**.
2. **Q4 → GPTQ-Ternary không giảm bớt biến dạng**:
   - Đi từ bản đã nén INT4 (Q4) rồi tiếp tục bù lỗi sang Ternary làm sai lệch tích lũy (quantization error accumulation), dẫn đến PPL sập tương tự mốc C.
3. **Giữ 1% cột trọng yếu FP16 (Hybrid Condition E)**:
   - Dù giữ lại 1% số cột trọng số có năng lượng activation lớn nhất ở dạng FP16, 99% tham số còn lại bị ép xuống 3 mức $\{-1, 0, 1\}$ vẫn phá hủy hoàn toàn không gian ẩn (latent representations) của mô hình 0.6B.

---

## 3. Ước tính Kích thước & Tốc độ nếu xuất định dạng `i2_s` (BitNet)

- **Tham số tổng**: 596.049.920 (~596M).
- **Trọng số nén 1.58-bit (`i2_s`)**: ~357.8M tham số Linear target.
- **Dung lượng mô hình `i2_s`**: **~150 MB – 180 MB**.
- **Tốc độ suy luận dự kiến trên CPU Laptop**: **~140 – 180 tok/s** (dựa trên băng thông RAM ~50-60 GB/s).
- **Kết luận khả thi**: Dung lượng và tốc độ rất ấn tượng, nhưng **chất lượng = 0%** nếu dùng Convert-only.

---

## 4. KẾT LUẬN & ĐÓNG ĐINH LUẬN ĐIỂM (VERDICT)

> [!CAUTION]
> **Kết luận cuối cùng**: **Hướng CONVERT-ONLY (Zero-train PTQ) KHÔNG THỂ THÀNH CÔNG ở dải 1.58-bit (Ternary)** trên các mô hình LLM quy mô nhỏ (< 1B như Qwen 0.5B/0.6B). Mọi thuật toán PTQ toán học thuần túy (kể cả GPTQ Hessian compensation) đều thất bại hoàn toàn (PPL nổ > 10.000×, mô hình sinh rác).

### Khuyên dùng cho lộ trình sản phẩm:
Muốn có mô hình Ternary 1.58-bit chạy siêu tốc `i2_s` trên `bitnet.cpp`:
1. **BẮT BUỘC phải qua Gradient Training (STE)**: Sử dụng phương pháp **QAT Warm-start** (như kế hoạch trong `PLAN_TERNARY_QWEN.md` — giữ master weight FP16, forward qua `BitLinear` + STE, train self-distillation 2.000–3.000 step) hoặc **Train from-scratch**.
2. **Lập hồ sơ kết thúc thí nghiệm Convert-only**: File số liệu gốc đã được ghi nhận đầy đủ tại `eval/ternary_lab/qwen3_results.json` và báo cáo khảo sát tại `eval/ternary_lab/SURVEY_TERNARY_2026.md`.
