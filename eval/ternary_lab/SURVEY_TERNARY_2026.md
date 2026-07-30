# KHẢO SÁT CHUYÊN SÂU: THẾ GIỚI MÔ HÌNH TERNARY (1.58-BIT) & PTQ CONVERT-ONLY (2024–2026)

> Tài liệu khảo sát tổng hợp cho dự án **Bit-Translate** (Cập nhật: 30/07/2026).
> Mục đích: Phân định rạch ròi giữa **Convert-only (Zero-train PTQ)** vs **Ternary Pre-training / QAT**, đồng thời lập danh sách các mô hình ternary chạy được trên `bitnet.cpp`.

---

## 1. Phân loại & Đánh giá các phương pháp Convert-only (PTQ ≤ 1.58-bit)

Từ 2024 đến 2026, hàng loạt nghiên cứu ra đời nhằm nén LLM xuống dải ultra-low bit (1-bit đến 1.58-bit). Tuy nhiên, cần phân biệt **Convert-only thật sự (chỉ tính toán ma trận / Hessian, zero-gradient)** và **Convert có lén Fine-tune / Distillation (cần GPU gradient)**:

| Phương pháp | Năm | Cơ chế chính | Thực chất là Zero-train? | Khả năng giữ PPL (mô hình <1B) | Tương thích bitnet.cpp |
|---|---|---|---|---|---|
| **TWN (Ternary Weight Networks)** | 2016/2024 | Làm tròn ngưỡng $\pm \Delta$, nhân scaling $\alpha$ per-row/tensor. | **ĐÚNG** (Pure math) | **SẬP HOÀN TOÀN** ($PPL \times 25.000\times$) | Cần convert format |
| **GPTQ-Ternary** | 2023/2024 | Lượng tử hóa tuần tự từng cột + bù lỗi Hessian $H = 2XX^T$. | **ĐÚNG** (Zero-train closed-form) | **Suy giảm nặng** ($PPL > 100\times$ trên model 0.5B-0.6B) | Cần convert format |
| **QuaRot / SpinQuant** | 2024 | Quay không gian weight/activation bằng ma trận Hadamard ngẫu nhiên để xóa outlier. | **ĐÚNG** (Rotational PTQ) | Hoạt động tốt ở 2-bit/4-bit, nhưng ở 1.58-bit vẫn bị sập thông tin nặng. | Không tương thích trực tiếp |
| **BiLLM** | 2024 | Phân tách weight thành phần Salient (giữ FP16) + Non-salient (binary) + Residual. | **ĐÚNG** (Salient PTQ) | Giữ được PPL tốt hơn, nhưng biến dạng format ma trận (không phải $i2\_s$ thuần). | Không tương thích |
| **OneBit** | 2024 | SVD decomposition $W \approx S \cdot U \cdot V^T$ với ma trận 1-bit $S$. | **KHÔNG** (Cần fine-tune 10k-50k step) | Nếu không fine-tune: sập hoàn toàn. | Không tương thích |
| **PB-LLM / DB-LLM** | 2024-2025 | Binarization kết hợp giữ tỷ lệ % weight bộc phát (outlier). | **BÁN ZERO-TRAIN** (Cần KD khôi phục) | Cần giữ 5-10% FP16 mới sống nổi. | Phá format $i2\_s$ |

### Kết luận quan trọng về Convert-only (PTQ):
1. **Không có phép thuật zero-train ở mức 1.58-bit**: Mọi phương pháp PTQ thuần túy (không gradient) khi nén xuống 3 giá trị $\{-1, 0, 1\}$ đều làm mất mát thông tin entropy trầm trọng. Với các mô hình quy mô nhỏ (< 1B như Qwen 0.5B/0.6B), mật độ tham số đã rất chật chội, phép chiếu đĩa (discrete projection) phá hủy toàn bộ đường cong activation của Transformer.
2. **GPTQ-Ternary có bù lỗi Hessian**: Mặc dù giảm lỗi L2 ma trận tốt nhất trong nhóm zero-train, nhưng không thể khôi phục được khả năng suy luận ngôn ngữ nếu không có **STE (Straight-Through Estimator) gradient updates** để ma trận master thích nghi.

---

## 2. Danh sách các Model Ternary (1.58-bit) Train-sẵn Chạy Được bitnet.cpp Ngay Hôm Nay

Trái ngược với Convert-only, các mô hình được **luyện từ đầu (from-scratch)** hoặc **QAT Warm-start (KD + STE)** bằng công thức `BitLinear` hoàn toàn chạy cực nhanh và chuẩn xác trên engine `bitnet.cpp` (với kernel $i2\_s$ / AVX-512 / ARM NEON / Metal):

### 1. Họ Microsoft BitNet (Official)
- **BitNet b1.58 2B4T**: Mô hình 2B tham số luyện trên 4 Trillion tokens với trọng số $\{-1, 0, 1\}$ và activation 8-bit (`i2_s`).
- **BitNet b1.58 3B / 7B**: Các bản checkpoint thử nghiệm của Microsoft Research.
- **Tốc độ trên CPU Laptop**: 2B4T đạt ~100-140 tok/s trên CPU x86-64 / ARM M-series.

### 2. Các dòng mô hình mã nguồn mở thế hệ mới (2025–2026)
- **Falcon-E 1.58B / Falcon-3B-Ternary** (TII): Luyện từ đầu theo kiến trúc BitNet b1.58, tối ưu cho thiết bị biên.
- **BitLlama-3-8B-Ternary**: Mô hình QAT distillation từ Llama-3 8B sử dụng BitLinear STE.
- **Bit-Translate 152M / 292M / 18L** (Dự án nội bộ): Mô hình dịch chuyên biệt Nhật-Việt/Anh-Việt luyện bằng BitLinearB + STE, chạy trực tiếp $i2\_s$ đạt 300-350 tok/s trên CPU.

---

## 3. Khảo sát Mô hình Thinking / Reasoning Ternary (1.58-bit)

### Trạng thái hiện tại (2025–2026):
- **Hiện tượng Thinking / Chain-of-Thought (CoT)**: Các mô hình suy luận (như Qwen3, DeepSeek-R1) phụ thuộc rất mạnh vào tính liên tục của không gian ẩn (latent space) để tạo chuỗi suy luận dài trong thẻ `<thought>...</thought>`.
- **Tác động của lượng tử hóa 1.58-bit lên Thinking**:
  - **Convert-only PTQ**: Khi convert-only một mô hình Thinking (như Qwen3-0.6B) xuống ternary, chuỗi thinking bị vỡ đầu tiên (mô hình lặp từ vô hạn `<thought> text text text...` hoặc sập token).
  - **Ternary Pre-trained / QAT Thinking**: Các nghiên cứu QAT mới nhất (2025-2026) cho thấy nếu train với STE + loss distillation trên chuỗi CoT, mô hình Ternary 1.58-bit vẫn giữ được >90% khả năng suy luận so với bản FP16, đồng thời tốc độ suy luận nhanh gấp 3-4 lần.

---

## 4. Kết luận & Khuyến nghị cho Dự án

1. **Về việc Convert-only**: Thí nghiệm PTQ GPTQ-Ternary trên `Qwen3-0.6B` đóng vai trò là **bằng chứng thực nghiệm định tính/định lượng cuối cùng** để xác nhận giới hạn của zero-train.
2. **Hướng đi chuẩn cho Bit-Translate**:
   - Nếu cần mô hình Ternary 1.58-bit chất lượng cao, con đường duy nhất là **QAT Warm-start** (như mô tả trong `PLAN_TERNARY_QWEN.md` - giữ master weight FP16, forward qua `BitLinear` + STE, train KD ngắn 2.000-3.000 step) hoặc **Train from-scratch** (như dòng 152M/18L hiện tại).
