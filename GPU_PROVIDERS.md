# Free-tier GPU cloud cho train/fine-tune — kiểm chứng 2026-07-20

> Khác với [PROVIDERS.md](PROVIDERS.md) (API LLM để SINH DATA JA-VI), file này là **thuê GPU thật** (A10G/L40S/A100/H100...) để burst train/fine-tune ngoài máy local (Ryzen 5600X + 3060 Ti 8GB).
> Đã điều tra ~35 provider qua 2 vòng deep-research (fan-out search + adversarial verify 3-vote). Kết luận chung: **không ai cho thuê GPU cao cấp free, tự phục vụ, tái tục, nhiều giờ/tháng cả** — chỉ có vài lựa chọn free thật nhưng eo hẹp, và nhiều claim phổ biến trên mạng bị kiểm chứng là SAI/lỗi thời.

## Tier 1 — Free thật, tự đăng ký, dùng được ngay

| Provider | Free tier | GPU | Chu kỳ | Đăng ký | Thẻ? | Giới hạn |
|---|---|---|---|---|---|---|
| **HF Spaces ZeroGPU** (chưa đăng nhập) | 2 phút/ngày | RTX Pro 6000 Blackwell (48–96GB) | Reset hàng ngày | Không cần tài khoản | Không | Ưu tiên thấp; ~60s/lần gọi |
| **HF Spaces ZeroGPU** (tài khoản free) | 5 phút/ngày | RTX Pro 6000 Blackwell (48–96GB) | Reset hàng ngày | Chỉ email | Không | Thêm cap ~3 lần chạy/ngày (dễ báo "hết quota" sớm); tự host Space riêng cần PRO $9/tháng |
| **Google Colab (free)** | Tới 12h/phiên | T4, 16GB | Theo phiên | Tài khoản Google | Không | GPU không đảm bảo có sẵn; hay bị ngắt |
| **Modal** | $30/tháng credit | A100, H100 (trả theo giây) | **Renew hàng tháng** | Email + thẻ lưu | Thẻ không bị trừ tới khi hết $30 | ~12h A100/tháng rồi tính phí thật |
| **Replicate "Try for Free"** | Số lần chạy giới hạn, không công bố cụ thể | Tuỳ model hosted sẵn | Một lần/model (không tái tục) | Email | Không | Chỉ chạy được model có sẵn trong collection, KHÔNG phải thuê GPU tổng quát |

→ **Đánh giá thực tế**: ZeroGPU chỉ đủ sanity-check/inference nhanh, không đủ train nghiêm túc. Colab tương đương hoặc kém 3060 Ti. **Modal $30/tháng vẫn là lựa chọn tốt nhất** để burst lên A100 vài giờ.

## Tier 2 — Có credit đáng kể nhưng phải APPLY/xét duyệt (không "dễ đăng ký")

| Provider | Credit | GPU | Cách lấy | Vấn đề |
|---|---|---|---|---|
| Nebius AI Lift | Tới $150k + $10k inference | H100/H200/B200 | Cần NVIDIA Inception + duyệt 7–10 ngày làm việc | Ưu tiên startup có vốn đầu tư, không phải cá nhân |
| Vast.ai Startup Program | Tới $2,500 | Consumer GPU → A100/H200 | Liên hệ trực tiếp, "chốt sau khi thảo luận" | Không tự động, không tự phục vụ |
| RunPod Startup Program | $1,000 (Starter tier) | — | Phải apply | "Venture backing (Seed→Series B+) strongly preferred" |
| CoreWeave Accelerator | Không công bố số tiền | — | Phải apply (rolling) | Chỉ dành công ty/tổ chức, không nhận cá nhân |
| OVHcloud Free Trial (US) | $200 một lần (tới $300 nếu mua thêm Managed DB) | Không rõ có áp dụng cho GPU hay không | Signup online | **Bắt buộc thẻ + phí xác minh $0.99**; hết hạn 90 ngày; trang official không liệt kê GPU trong phạm vi credit |

## Tier 3 — Free tier vô dụng hoặc không tồn tại cho GPU (đã kiểm chứng)

| Provider | Vấn đề |
|---|---|
| **Google Cloud Free Trial** | $300/90 ngày nhưng **loại trừ hoàn toàn GPU/TPU** (quota GPU = 0 trong trial). Cần thẻ (chỉ hold, không trừ) nhưng vô dụng cho mục đích GPU. |
| **AWS SageMaker Studio Lab** | T4 free (4h/24h) nhưng **đóng đăng ký người dùng mới từ 30/7/2026** — tài khoản cũ vẫn dùng được. |
| **TensorDock** | Zero free tier — pure prepaid, bắt buộc nạp tối thiểu $5 trước khi dùng. |
| **Together.ai** | Tự công bố "does not currently offer free trials" — bắt buộc nạp tối thiểu $5. Gói $25 cũ đã rút ~7/2025. |
| **Lambda Labs** (pricing chính) | Trang pricing chính thức không đề cập free tier/trial nào. |
| **Oracle Cloud "Always Free" có A10 GPU** | Claim phổ biến này **BỊ BÁC BỎ khi kiểm chứng (0-3)** — nhiều khả năng SAI/lỗi thời, đừng tin theo mà không tự kiểm tra lại trên oracle.com. |
| **Saturn Cloud** ("30h/tháng free, GPU 16GB") | Nguồn gốc là **blog 2021 đã lỗi thời** (tự nhận đã đổi định vị thành "enterprise AI platform"), không nêu rõ GPU model cụ thể. Trang pricing HIỆN TẠI (2026) không có free tier định lượng nào — chỉ trả phí theo giờ (A10G ~$1.5/h, H100 ~$2.95/h). **Đừng dựa vào con số 16GB/30h này.** |

## Chưa xác định rõ (claim đưa ra bị bác bỏ khi kiểm chứng — KHÔNG đồng nghĩa "không có", chỉ là chưa xác nhận được)

- **Fal.ai** — chưa rõ có free tier hay không; GPU lineup xác nhận được: RTX PRO 6000, H100, H200, B200, B300 (đều trả phí, rẻ nhất $1.10/h) — chưa rõ có A10G/L40S hay không.
- **Novita AI** — claim "$100 sandbox credit, không cần thẻ" bị bác bỏ (0-3); một nguồn khác nói chỉ $1. Thực tế chưa rõ.
- **Baseten** — catalog GPU xác nhận có A10G, A100, H100, H100 MIG, T4, L4, B200 (không có L40S) nhưng đều là **trả phí**; có nhắc "credit miễn phí cho account mới" nhưng số tiền/cơ chế/GPU áp dụng chưa xác minh được.
- **Paperspace/DigitalOcean Gradient** — claim "chỉ có GPU cũ M4000/P5000" bị bác bỏ (1-2) → có thể có GPU mới hơn, cần tự kiểm tra.
- **Kaggle** — claim về giới hạn session/notebook public-by-default bị bác bỏ (0-3), thực tế chưa rõ.
- **Lightning AI** — mọi con số cụ thể (T4→H200, 80h/tháng, 15 credit, 4h/phiên) đều bị bác bỏ (0-3) → cần tự kiểm tra trực tiếp trên lightning.ai.

## Hoàn toàn chưa điều tra (gap, không phải bằng chứng "không có")

Crusoe Cloud, Genesis Cloud, Scaleway (có claim chưa kiểm chứng về gói startup €36k), Alibaba Cloud (compute GPU — khác DashScope LLM API đã có trong PROVIDERS.md; có claim chưa kiểm chứng về $120k lifetime credit APAC), Tencent Cloud, IBM Cloud, Google Vertex AI.

## Kết luận / khuyến nghị

Không có provider nào cho thuê A10G/L40S/A100/H100 miễn phí kiểu tự phục vụ, không cần apply, nhiều giờ/tháng. Xếp theo mức độ hữu dụng cho nhu cầu train BitNet:
1. **Modal $30/tháng** — thực tế nhất để burst A100 vài giờ khi cần.
2. **HF ZeroGPU / Colab free** — chỉ hợp sanity-check, không đủ train nghiêm túc.
3. Nếu dự án đạt tới mức "startup" hoặc có thể chứng minh nghiêm túc — cân nhắc apply Nebius AI Lift / RunPod Startup Program / Vast.ai Startup Program (nhưng tốn thời gian chờ duyệt, không phải giải pháp tức thời).

## Nguồn (chính, đã kiểm chứng qua adversarial verify)

- HF ZeroGPU: [huggingface.co/docs/hub/spaces-zerogpu](https://huggingface.co/docs/hub/en/spaces-zerogpu)
- Google Colab: [research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html)
- Modal: [modal.com/pricing](https://modal.com/pricing)
- Google Cloud Free Trial: [docs.cloud.google.com/free/docs/free-cloud-features](https://docs.cloud.google.com/free/docs/free-cloud-features)
- AWS SageMaker Studio Lab (đóng cửa 30/7/2026): [docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- Nebius AI Lift: [nebius.com/blog/posts/ai-lift-startups-innovation-with-nvidia](https://nebius.com/blog/posts/ai-lift-startups-innovation-with-nvidia)
- Vast.ai Startup Program: [vast.ai/startup](https://vast.ai/startup)
- RunPod: [docs.runpod.io/references/referrals](https://docs.runpod.io/references/referrals), [runpod.io/startup-program](https://www.runpod.io/startup-program)
- CoreWeave: [coreweave.com/blog/coreweave-launches-startup-accelerator-program](https://www.coreweave.com/blog/coreweave-launches-startup-accelerator-program-democratizing-access-to-gpu-compute-in-the-cloud)
- TensorDock: [tensordock.com/cloud-gpus.html](https://www.tensordock.com/cloud-gpus.html)
- Together.ai: [docs.together.ai/docs/billing-credits](https://docs.together.ai/docs/billing-credits)
- Lambda Labs: [lambda.ai/pricing](https://lambda.ai/pricing), [docs.lambda.ai/public-cloud/billing](https://docs.lambda.ai/public-cloud/billing/)
- Replicate: [replicate.com/collections/try-for-free](https://replicate.com/collections/try-for-free)
- Baseten: [baseten.co/pricing](https://www.baseten.co/pricing/)
- OVHcloud: [us.ovhcloud.com/public-cloud/free-trial](https://us.ovhcloud.com/public-cloud/free-trial/)
- Saturn Cloud (pricing hiện tại — KHÔNG có free tier định lượng): trang pricing chính thức saturncloud.io (blog 2021 dẫn chiếu tự nhận lỗi thời)
