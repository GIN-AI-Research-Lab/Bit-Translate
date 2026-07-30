# Dự án: Model dịch Việt ↔ Nhật 1.58-bit siêu nhẹ chạy CPU

> ⚠️ **TRẠNG THÁI HIỆN TẠI (Cập nhật 2026-07-30):**
> - **Vòng V7A XONG — bản deploy chính là `v7a_avg` (152M, 18L)**: thẻ model + số liệu đầy đủ ở **`TONGKET_V7A.md`**; nhật ký tiến độ & vị trí checkpoint: **`STATUS.md`** (mục 🟢 V7A) và `ARTIFACTS.md`.
> - Bench 200 câu chấm mù: v7a i2s deploy **đứng đầu acc 78,5%** (Google 69,0), câu ngắn đè Google, câu dài +6..+8 so v6/gate. Gate kết luận **18L chưa đến trần** — chưa grow 24L (`PLAN_V7A.md`).
> - **Thầy sinh KD**: Gemini Live API (6 key, gần như miễn phí). **Modal đã cạn tiền cả 2 tài khoản** — train tiếp chỉ còn đường miễn phí là máy A desktop (§7).
> - File ngữ cảnh này giữ vai trò định hướng tổng thể về phần cứng và kiến thức nền BitNet.

## 1. Mục tiêu

- Model dịch **text-to-text** Việt ↔ Nhật, **hai chiều trong một model**.
- Kiến trúc trọng số **1.58-bit (BitNet, ternary −1/0/+1)**, train from-scratch.
- Kích thước mục tiêu: **100–200M tham số** → khi chạy chỉ **~25–50MB**, inference trên **CPU** (AVX2) bằng bitnet.cpp.
- KHÔNG làm speech ở giai đoạn này (ASR dùng whisper.cpp có sẵn nếu cần sau).

## 2. Phần cứng & môi trường

> ⚠️ **CÓ HAI MÁY — kiểm đang ở máy nào TRƯỚC KHI áp cấu hình ở §7.**
> Nhận biết nhanh: `python -c "import torch;print(torch.cuda.is_available())"` → `False` là laptop.

**Máy A — desktop ở nhà** (máy mà §7 nói tới):
- GPU: RTX 3060 Ti **8GB VRAM** (Ampere, có Tensor Cores → dùng mixed precision).
- CPU: Ryzen 5 5600X (6 nhân / 12 luồng).
- RAM: 48GB DDR4-3200 (kiểm tra dual-channel).
- OS: Windows → **chạy toàn bộ pipeline trong WSL2** (DataLoader/multiprocessing nhanh hơn Windows native).

**Máy B — laptop công ty** (nơi phần lớn việc eval/phân tích đang diễn ra):
- CPU **Intel Core Ultra 5 225H**, 14 lõi/14 luồng, base 1,70 GHz. GPU chỉ có Intel Arc iGPU.
- **KHÔNG có torch CUDA** → không train được ở đây, chỉ chạy eval/inference trên CPU.
- **Có E-core, và nhiều luồng làm CHẬM đi**: đo thật 140 tok/s ở 4–6 luồng, 102 ở 8 luồng,
  **21 ở 12 luồng**. Luôn đặt `OMP_NUM_THREADS=5 MKL_NUM_THREADS=5`, đừng để torch tự chọn.
- Inference là **memory-bound** (§6) → chạy 2 job model cùng lúc KHÔNG nhanh hơn, chúng chỉ
  chia đôi băng thông RAM. Đo thật: job bị bỏ đói xuống 0,74 lõi và chậm 5,6×. **Chạy tuần tự.**
- **Kiểm đã cắm sạc**: trên pin, clock tụt xuống 1,22 GHz (dưới cả base). `BatteryStatus` = 1
  là đang xả pin, 2 là điện lưới.

**Train thật sự chạy ở đâu**: từ vòng 4 trở đi train trên **Modal (GPU L40S thuê giờ, ~$1/1000
step)**, không train local ở máy nào. Cấu hình §7 là cho máy A, giữ lại làm tham chiếu.

- Ổ cứng: chừa **50–100GB** trống cho data thô + file trung gian (bộ data sạch cuối chỉ ~1GB).

## 3. Kỳ vọng thực tế (đã thống nhất — đừng hứa quá)

- Chất lượng đích: **"tốt, dùng được"** — KHÔNG phải "gần tuyệt đối" (không tồn tại với mọi model, càng không với 100–200M params).
- Train from-scratch trên 1 GPU 3060 Ti: **nhiều ngày tới vài tuần** chạy liên tục.
- Nút thắt lớn nhất là **data Việt-Nhật khan hiếm**, không phải phần cứng.
- Chất lượng đến từ **vòng lặp data (3–5 vòng)**: train → đo → phân tích lỗi → bổ sung data → train lại.

## 4. Điểm kỹ thuật then chốt về BitNet (tránh nhầm)

- BitNet **buộc train from-scratch** (không quantize model FP16 có sẵn xuống 1.58-bit được).
- Khi TRAIN: giữ **master weights FP16/BF16** (quantization-aware training, straight-through estimator); chỉ ép ternary lúc forward. ⇒ **VRAM khi train ≈ train FP16 cùng size**. Lợi 1.58-bit chỉ có khi INFERENCE.
- Vì vậy 8GB VRAM giới hạn model ở ~100–200M params (kèm thủ thuật ở mục 7). Model lớn hơn: cân nhắc DeepSpeed ZeRO-Offload đẩy optimizer state ra 48GB RAM (chậm hơn, chấp nhận được).

## 5. Lộ trình 6 bước

### Bước 1 — Gom data (mục tiêu: 2–5 triệu cặp VI-JA sạch)
Nguồn theo thứ tự ưu tiên:
1. **OPUS** (opus.nlpl.eu): OpenSubtitles VI-JA (lớn nhất nhưng nhiễu), CCMatrix, CCAligned, WikiMatrix, Tatoeba (sạch, ít), TED2020.
2. Bắc cầu qua tiếng Anh: **PhoMT** + **MTet** (EN-VI) và **JParaCrawl** (EN-JA) → tạo cặp VI-JA gián tiếp qua pivot.
3. **Sinh data bằng LLM** theo khung JLPT: lấy danh sách ngữ pháp/từ vựng N5→N1 làm hạt giống, sinh cặp câu VI-JA; **dồn ngân sách token cho N2–N1** (nội dung nhiều hơn hẳn N5–N4). ~1M token output ≈ 10–15k cặp dùng được sau lọc.
4. **Back-translation** sau khi có model sơ bộ (dùng chính model hai chiều để sinh thêm cho chiều yếu).

Lưu ý pháp lý: kiểm tra license từng corpus (một số CC-BY-NC); kiểm tra điều khoản nhà cung cấp LLM về dùng output để train; không dùng nguyên văn đề thi JLPT (có bản quyền), chỉ dùng danh sách từ/ngữ pháp cộng đồng.

### Bước 2 — Lọc & chuẩn hóa
- Khử trùng lặp (exact + near-dup).
- Lọc tỉ lệ độ dài bất thường giữa hai vế.
- Chấm điểm khớp nghĩa bằng **LaBSE** (hoặc LASER), bỏ cặp dưới ngưỡng (tune ngưỡng ~0.7–0.8).
- Tiếng Việt: chuẩn hóa Unicode **NFC**. Tiếng Nhật: chuẩn hóa full/half-width.
- Tách test set: dùng **FLORES-200** (vie_Latn ↔ jpn_Jpan) làm benchmark trung lập + giữ riêng dev set từ data sạch.

### Bước 3 — Tokenizer
- Train **SentencePiece** (unigram hoặc BPE) **vocab 32k**, CHỈ trên tiếng Việt + tiếng Nhật của dự án.
- Thêm token đặc biệt thẻ hướng: `>>vie<<`, `>>jpn<<`.

### Bước 4 — Kiến trúc & train
- **Transformer encoder-decoder** ~100–200M params (ví dụ khởi điểm: 6+6 layers, d_model 512–768, FFN 2048–3072 — tinh chỉnh để vừa VRAM).
- Các lớp linear dùng **BitLinear** (BitNet b1.58): ternary weights + 8-bit activations, RMSNorm, straight-through estimator.
- Data hai chiều: mỗi cặp sinh 2 mẫu (VI→JA và JA→VI) với thẻ hướng ở đầu input.

### Bước 5 — Đánh giá & vòng lặp
- Metric chính: **chrF** (sacrebleu) — ổn hơn BLEU cho tiếng Nhật; BLEU tiếng Nhật nếu dùng thì tokenize bằng MeCab.
- Đo trên FLORES-200 + dev set; lấy Google Translate làm mốc so sánh.
- Đọc lỗi bằng mắt → phân loại (kính ngữ, câu dài, chủ đề yếu…) → sinh/bổ sung data nhắm đúng chỗ yếu → train tiếp. Lặp 3–5 vòng.

### Bước 6 — Đóng gói inference
- Convert sang định dạng **bitnet.cpp / GGUF**.
- Chạy CPU: threads = 6 (số nhân vật lý), yêu cầu AVX2; kỳ vọng **hàng trăm token/giây** (memory-bound: token/s ≈ băng thông RAM ÷ size model).
- Nếu cần "dịch khi đang gõ": dùng chiến lược **re-translation** (dịch lại mỗi lần input đổi) — model nhẹ nên chạy lại nhiều lần vô tư; KHÔNG cần train simultaneous.

## 6. Baseline đối chứng (quyết định mở — nên làm)

Trước hoặc song song với 1.58-bit from-scratch: **fine-tune một model dịch đa ngữ có sẵn** (vd. dòng NLLB distilled / Opus-MT) trên cùng data, rồi quantize INT8/INT4. Mục đích: có mốc chất lượng trong vài ngày để đo chính xác 1.58-bit from-scratch đánh đổi bao nhiêu chất lượng lấy độ nhẹ.

## 7. Cấu hình train cụ thể (tăng tốc theo thứ tự tác động)

1. **Mixed precision**: torch.cuda.amp, ưu tiên BF16 (Ampere hỗ trợ).
2. **Pre-tokenize toàn bộ data** thành binary; nạp hết vào RAM (48GB dư sức) — GPU không bao giờ chờ I/O. DataLoader: num_workers 6–10, pin_memory=True.
3. Batch lớn nhất VRAM chịu được + **gradient accumulation**; **8-bit Adam** (bitsandbytes) giảm ~4x bộ nhớ optimizer.
4. **gradient checkpointing** chỉ bật nếu thiếu VRAM (đổi ~30% tốc độ).
5. **torch.compile** (PyTorch 2.x).
6. Hạ tầng: chạy trong tmux/nohup để train nhiều ngày độc lập với phiên làm việc; log ra file + checkpoint định kỳ (mỗi N bước) để cúp điện không mất; theo dõi nhiệt GPU (tránh thermal throttling).
7. Windows-side chỉ là vệ sinh: High Performance power plan, không sleep, đóng app chiếm GPU. Đừng kỳ vọng tinh chỉnh Windows cho hơn vài %.

## 8. Cấu trúc thư mục gợi ý

```
project/
  CLAUDE.md            # file này
  data/raw/            # corpus tải về (OPUS, PhoMT, MTet, JParaCrawl…)
  data/interim/        # sau từng bước lọc (giữ lại để quay lui)
  data/clean/          # cặp câu cuối cùng + train/dev/test
  data/synthetic/      # data sinh từ LLM (JLPT) + back-translation
  tokenizer/           # SentencePiece model
  src/                 # model (BitLinear), train loop, eval
  scripts/             # tải data, lọc LaBSE, chuẩn hóa, convert GGUF
  checkpoints/
  eval/                # kết quả chrF theo từng vòng lặp
```

## 9. Việc con người phải tự làm (Claude Code không thay được)

- Chờ train (nhiều ngày) — Claude Code khởi chạy & kiểm tra log, không cần mở phiên liên tục.
- **Đọc và đánh giá bản dịch bằng mắt** ở mỗi vòng lặp — quyết định chất lượng cuối cùng.
- Quyết định ngân sách token khi sinh data bằng LLM.
