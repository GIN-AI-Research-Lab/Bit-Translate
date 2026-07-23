# Provider API cho sinh data & Knowledge Distillation (KD) JA-VI — cập nhật 2026-07-21

> Chạy: `bash scripts/gen_any.sh <provider> <todo.json> [script]` — key đọc từ `.env` (đã có sẵn chỗ điền).
> **⚠️ LƯU Ý AUDIT 2026-07-20:** Pilot 300 câu/model + audit lớp 3 phân tầng đã **LOẠI qwen-max (16.7% lỗi), qwen-turbo (13% lỗi), qwen-flash (11% lỗi)** do sai nghĩa/false-friend/đảo phủ định có hệ thống. **CHỈ DÙNG `gemini-flash-lite-latest` VÀ `qwen-plus` LÀM THẦY.**

## Xếp hạng provider & model thầy (cho JA-VI KD / sinh data)

| # | Provider | Model | Đánh giá qua Audit (2026-07-20) | Trạng thái & Ghi chú |
|---|---|---|---|---|
| 1 | **Gemini (Paid/Free)** | `gemini-flash-lite-latest` | **Acc 4.90** (thắng Google 10/10 domain, 0/30 lỗi audit) | **THẦY CHÍNH (Thằng top 1)**. Paid tier RPM cao ($0.10/$0.40 per MTok) dùng cho GĐ1 KD 200k câu. |
| 2 | **DashScope (Alibaba)** | `qwen-plus` | **Acc 4.72** (thắng Google 10/10 domain, 0/30 lỗi audit) | **THẦY PHỤ (Thắng top 2)**. 1M token free quota. CHỈ endpoint Singapore (`dashscope-intl`). |
| 3 | **DashScope (Bị loại)** | `qwen-max`, `qwen-turbo`, `qwen-flash` | **CẤM DÙNG (11%–16.7% lỗi)** | False-friend IT (`エージェント→đại lý`, `イメージ→hình ảnh`), đảo phủ định (`貫く→xuyên thủng`). |
| 4 | **Zhipu bigmodel.cn** | `glm-4.7-flash` | FREE VĨNH VIỄN, RPM ~50 | Dùng cho data phụ/phrasebook đơn giản, **không dùng làm KD thầy chính**. |
| 5 | **DeepSeek** | `deepseek-v4-flash` | 5M token (hạn 30 ngày) | Dự phòng JSON kỷ luật. `deepseek-chat` đã ngưng hỗ trợ. |

## Chiến lược phân việc (KD & Data Synthesis)

- **KD Thầy chính (Sequence-Level KD)**: `gemini-flash-lite-latest` (Paid/Free) + `qwen-plus` (Free quota). KHÔNG dùng các model tier yếu hơn làm teacher.
- **Mode dễ, volume phụ**: Zhipu `glm-4.7-flash` (dịch phrasebook, câu đơn).
- **Quy trình Lọc 4 Lớp bắt buộc**: Rule filter (`filter_kd_corpus.py`) → LaBSE ≥ 0.55 (`labse_score.py`) → Đọc mẫu phân tầng Claude (n=30-100/model, loại nếu lỗi >5%) → Audit pilot.

## Nguồn
- Audit details: `PLAN_KD_JA2VI.md` §3.1 & `STATUS.md`
- DashScope: [alibabacloud.com — free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)
