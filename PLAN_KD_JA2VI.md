# PLAN MASTER: BITNET 200M FROM-SCRATCH (AUTO-DETECT)

## ⏱ 1. Ước Tính Thời Gian & Cấu Hình Steps

### Bước 1: Distillation (KD) 11.88M Câu Bằng Gemini 3.1 Live API (0đ)
- **5 Keys**: ~39.6 giờ (~1.6 ngày)
- **7 Keys**: ~28.2 giờ (~1.1 ngày)
- **10 Keys**: ~19.8 giờ (< 1 ngày)

### Bước 2: Build Tokenized Binary Dataset
- **23.76M sequences** (Auto-detect format 2 chiều).
- Thời gian build: ~20 - 30 phút.

### Bước 3: Train BitNet 200M Trên Modal GPU L40S
- **Tổng Steps**: **45.000 Steps** (~2.8 Tỷ tokens, 1 Epoch).
- **Tốc độ L40S**: ~55.000 tok/s.
- **Thời gian train**: **~25 - 28 Giờ**.
- **Chi phí GPU Modal**: ~$7.50 - $9.00 USD.

---

## 🎯 2. Chỉ Số Kỳ Vọng So Với Google Translate & Haiku

| Tiêu chí | Google Translate | Claude 3.5 Haiku | **BitNet 200M (Model mới)** |
|---|---|---|---|
| **Điểm Đánh Giá Chất Lượng** | 8.8 / 10 | 9.3 / 10 | **8.7 - 9.0 / 10** |
| **Tỷ Lệ Câu Dùng Được** | ~82% | ~91% | **85% - 88%** *(Vượt mục tiêu 80%)* |
| **Tốc Độ CPU Offline** | N/A | N/A | **~180 - 200 tok/s (~100ms)** |
| **Kích Thước GGUF** | N/A | N/A | **~130 MB** |

### Tỷ Lệ Thành Công Theo Dạng Câu:
- **Câu Ngắn (1-10 từ)**: **96% - 98%**
- **Câu Trung Bình (10-25 từ)**: **90% - 93%**
- **Câu Dài Phức Tạp (>25 từ)**: **80% - 84%**
- **Câu Bẫy / Thành Ngữ / Keigo**: **78% - 82%**

---

## 🚀 Các Bước Khởi Chạy Máy Mới

```bash
git clone https://github.com/trituenguyen97/Bit-Translate.git
cd Train-model-translate

pip install --break-system-packages google-genai websockets python-dotenv requests torch

# Điền 5-10 keys vào file .env (gemini_key_1 ... gemini_key_10)
python3 scripts/run_kd_5keys_parallel.py
modal run cloud/modal_finetune_kd.py --model-size 200m --max-steps 45000
```
