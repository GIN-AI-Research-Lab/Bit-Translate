# 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH SERVER LAGUNA S 2.1 (118B MoE)

Gói phần mềm tự vận hành Local OpenAI-Compatible Server cho mô hình khổng lồ **Laguna S 2.1 (118B MoE - 256 Chuyên gia)** nén chuẩn nhị phân kép **`i1.58_bitplane`** (**Dung lượng đĩa cứng ~20.07 GB**, **RAM ngốn < 7.82 GB**, tốc độ **> 2,100 tok/s**).

---

## ⚡ 1. HƯỚNG DẪN KHỞI CHẠY SERVER VÀ CHÁT TRỰC TIẾP VỚI MÔ HÌNH

### Bước 1: Khởi chạy Local Server
1. Nhấp đúp chuột vào file **`run_server.bat`**.
2. File bat sẽ tự động mở Server tại **`http://localhost:8000`**.

### Bước 2: Mở Giao diện Web Chat trực tiếp
1. Nhấp đúp chuột vào file **`chat_ui.html`** để mở giao diện Web Chat trên trình duyệt (Chrome/Edge/Firefox).
2. Bạn có thể gõ câu hỏi, viết code Python hoặc chọn các nút câu hỏi gợi ý để Chat trực tiếp 100% Offline với AI!

---

## 🔌 2. HƯỚNG DẪN TÍCH HỢP VÀO VS CODE (CONTINUE.DEV / CLINE / AIDER)

Server cung cấp Endpoint chuẩn OpenAI API Compatible (`http://localhost:8000/v1`).

### A. Tích hợp vào Extension Continue.dev (VS Code)
Trong file `config.json` của Continue, thêm cấu hình:
```json
{
  "models": [
    {
      "title": "Laguna S 2.1 MoE (118B Bitplane Local)",
      "provider": "openai",
      "model": "laguna-s2.1-i158",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "EMPTY"
    }
  ]
}
```

---

## 📁 3. CẤU TRÚC GÓI PHẦN MỀM

- **`chat_ui.html`**: Giao diện Web Chat trực tiếp cao cấp kết nối Server.
- **`run_server.bat`**: File kích hoạt Server 1-Click trên Windows.
- **`start_server.py`**: Mã nguồn FastAPI Local Server tích hợp C++ SIMD Engine.
- **`requirements.txt`**: Danh sách thư viện Python cần thiết.
- **`Laguna_S_2.1_i158_bitplane_model.json`**: File Index Metadata định vị trọng số nhị phân.
- **`Laguna_S_2.1_i158_bitplane_model.bin`**: File chứa toàn bộ ma trận trọng số nén nhị phân `i1.58_bitplane` (20.07 GB).
