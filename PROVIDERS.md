# Provider API free cho sinh data JA-VI — kiểm chứng 2026-07-16

> Chạy: `bash scripts/gen_any.sh <provider> <todo.json> [script]` — key đọc từ `.env` (đã có sẵn chỗ điền).
> Thông tin dưới đây là hiện trạng ĐÃ KIỂM CHỨNG tháng 7/2026 (khác vài chỗ so với thông tin cũ trên mạng).

## Xếp hạng nên đăng ký (cho JA-VI, ưu tiên đăng ký từ trên xuống)

| # | Provider | Free thật sự | Model nên dùng | JA-VI | Ghi chú quan trọng |
|---|---|---|---|---|---|
| 1 | **DashScope quốc tế** (Alibaba) | **1M in + 1M out / MỖI model**, 90 ngày | `qwen-plus` (hết thì `qwen-turbo`, `qwen-max`, `qwen-flash` — mỗi cái quota riêng!) | ⭐⭐⭐ top đầu, ít "Hán hoá" | CHỈ endpoint Singapore (`dashscope-intl`); bật **Free Quota Only** trong console để không bị tính phí lố |
| 2 | **Zhipu bigmodel.cn** | **GLM-4.7-Flash / 4.5-Flash FREE VĨNH VIỄN**, không giới hạn token | `glm-4.7-flash` | ⭐⭐ tốt Đông Á | Giới hạn ~1 request/giây (RPM 50 an toàn); cần xác minh SĐT; treo chạy cả đêm được |
| 3 | **SiliconFlow** | ¥14 credit + vài model **always-free** (danh sách xoay vòng) | `Qwen/Qwen3-8B` (free, 30 RPM) ; credit dùng cho Qwen lớn | ⭐⭐ (8B hơi yếu, cần review kỹ) | "14 triệu token" thực ra là **¥14 credit**; Qwen2.5-72B KHÔNG còn free |
| 4 | **DeepSeek** | Tặng **5M token, HẠN 30 NGÀY** (không phải 10M) | `deepseek-v4-flash` | ⭐⭐⭐ JSON kỷ luật | ⚠️ `deepseek-chat` bị khai tử 24/07/2026; chỉ đăng ký khi SẴN SÀNG dùng ngay (credit hết hạn) |
| 5 | Gemini (2 key hiện có) | 500 RPD/model/ngày, reset **14:00 VN** | `gemini-flash-lite-latest`, `gemini-2.5-flash` | ⭐⭐⭐ | Quota theo TỪNG model — cạn model này đổi model khác |
| 6 | OpenRouter | model `:free` quota nhỏ | `qwen/qwen3-coder:free` | ⭐ | Dự phòng |

## Chiến lược phân việc (không phụ thuộc một nhà nào)

- **Mode khó** (pk phủ định kép, dn đa nghĩa, idh idiom, harvest gloss): DashScope `qwen-plus` / Gemini 2.5-flash + thinking — cần model mạnh, sai là dạy hư model.
- **Mode dễ, cần VOLUME** (ht hội thoại, pb phrasebook, hop, ps): Zhipu `glm-4.7-flash` (miễn phí vô hạn, treo qua đêm) + SiliconFlow luân phiên.
- **Bung nước rút trước deadline**: DeepSeek 5M (đăng ký đúng lúc cần, dùng dồn trong tuần).
- MỌI batch đều qua **review 3 lớp**: rule filter → Claude đọc mẫu phân tầng (mode nào lỗi >5% thì đọc full) → LaBSE ≥0.8 lúc mix. Blacklist: `eval/vong2_review_blacklist.jsonl`.

## Nguồn
- SiliconFlow: [pricepertoken.com](https://pricepertoken.com/endpoints/siliconflow/free), [docs.siliconflow.cn](https://docs.siliconflow.cn/en/userguide/rate-limits/rate-limit-and-upgradation)
- Zhipu: [freellm.net](https://freellm.net/providers/z-ai-zhipu-ai), [tokenmix.ai](https://tokenmix.ai/blog/glm-free-api-access-tiers-2026)
- DeepSeek: [api-docs.deepseek.com](https://api-docs.deepseek.com/quick_start/pricing/), [pricepertoken.com](https://pricepertoken.com/endpoints/deepseek/free)
- DashScope: [alibabacloud.com — free quota](https://www.alibabacloud.com/help/en/model-studio/new-free-quota)
