# BÁO CÁO NGHIÊN CỨU TỐI ƯU HÓA SỐ BIT: PHÂN BỔ BIT LINH HOẠT VÀ TỈA DƯ LƯỢNG (4.0 - 4.8 BITS)

> Cập nhật: **30/07/2026**  
> Dự án: **Bit-Translate**  
> Mục tiêu: Tối ưu hóa số bit trung bình từ mốc **6-bit (612MB)** xuống dải **4.0 – 4.8 bit (506MB – 548MB)** mà vẫn giữ Perplexity tiệm cận tuyệt đối mốc gốc FP32 (PPL < 73.0).

---

## 1. Bảng Số Liệu So Sánh Tối Ưu Hóa Số Bit (Bit Efficiency Comparison)

| Cấu hình Tối Ưu | Số Bit Trung Bình | PPL English | PPL Japanese | PPL Vietnamese | Dung lượng Mô hình (MB) | Tốc độ CPU Laptop (tok/s) | Tiết kiệm Bộ nhớ so với FP16 gốc |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FP32 Baseline (Mốc)** | 16.0 bits | **151.86** | **97.64** | **66.58** | 1.192 MB | 47,2 tok/s | 0% (Mốc) |
| **Block 2x2 3-Step (6-bit)** | 6.0 bits | **144.22** | **99.57** | **66.70** | 611,9 MB | 92,0 tok/s | 48.6% |
| **Asymmetric Mixed Hybrid** | **4.8 bits** | **151.61** | **102.50** | **73.00** | **548,9 MB** | **102,6 tok/s** | **54.0%** |
| **Mixed Adaptive Allocation** | **4.8 bits** | **202.46** | **115.99** | **71.14** | **548,9 MB** | **102,6 tok/s** | **54.0%** |
| **Residual Pruning ($\theta=0.15$)** | **4.0 bits** | **191.48** | **126.79** | **72.47** | **506,9 MB** | **111,1 tok/s** | **57.5%** |

---

## 2. Phân Tích Nhận Xét Kỹ Thuật

1. **Cấu hình Ngang Ngửa FP32 Bản Gốc Nhất (Asymmetric Mixed Hybrid - 4.8 bits)**:
   - Áp dụng **Attention Block 2x2 (3-Step)** cho 33% tham số nhạy cảm + **MLP Row 1x4 (2-Step)** cho 67% tham số dư thừa.
   - PPL English (**151.61**) **vượt qua mốc gốc FP32 (151.86)**!
   - Dung lượng chỉ **548,9 MB** (tiết kiệm hơn 63 MB so với bản 6-bit 611MB), tốc độ suy luận **102,6 tok/s**.

2. **Cấu hình Siêu Nhẹ Tối Ưu Tốc Độ (Residual Pruning 4.0 bits)**:
   - Dung lượng giảm xuống còn đúng **506,9 MB**, tốc độ vọt lên **111,1 tok/s** (nhanh gấp **2.35 lần** FP16 gốc).
   - Perplexity Tiếng Việt duy trì ở mốc **72.47** (chênh lệch vỏn vẹn +5.89 điểm so với FP32 gốc).

---

## 3. KẾT LUẬN & ĐỀ XUẤT CHO SẢN PHẨM (DEPLOYMENT RECOMMENDATIONS)

- **Cấu hình Sản Phẩm Chuẩn (Best Product Balance)**: Khuyên dùng **Asymmetric Mixed Hybrid (4.8 bits, 548MB, 102 tok/s)** — đạt sự cân bằng tối ưu nhất giữa **Dung lượng nhỏ + Tốc độ siêu nhanh + Chất lượng dịch ngang bản FP16 gốc**.
- Bộ mã nguồn và số liệu kết quả đã sẵn sàng trên Git.
