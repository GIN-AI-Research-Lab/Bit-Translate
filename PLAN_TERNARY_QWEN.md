# PLAN: Qwen 0.5B → BitNet ternary bằng QAT warm-start ("nấc C")

> Viết 2026-07-29. Thí nghiệm cho câu hỏi: **có convert được model FP16 có sẵn ra
> BitNet không, thay vì train from-scratch?** Chạy thử SƠ BỘ trên desktop 3060 Ti;
> nếu tín hiệu tốt mà máy chậm thì chuyển Modal L40S; tín hiệu xấu thì dừng sớm.

## 0. Bằng chứng đã có — đọc trước khi làm

- **Nấc A (convert thuần công thức) ĐÃ ĐO, SẬP HOÀN TOÀN** (`eval/ternary_lab/`,
  commit `1807b06`): TWN per-tensor/per-row trên Qwen2.5-0.5B-Instruct cho ppl
  ×25.000–48.000, output rác, **tệ hơn đoán ngẫu nhiên 6–20×** (ppl 0,94–3,1M so
  uniform ~152k). A3 (least-squares α) = A2 từng bit — họ công thức data-free đã
  bão hoà, đừng tìm "công thức tốt hơn".
- **Nấc A KHÔNG bác nấc C.** Khác biệt căn bản:
  ```
  Nấc A:  W_fp16 --chiếu--> ternary → dùng luôn        (thông tin mất vĩnh viễn)
  Nấc C:  W_master = W_fp16 GIỮ NGUYÊN
          mỗi step: forward dùng weight_quant(W_master) → loss → STE → sửa W_master
          → master TRƯỜN dần sang cấu hình chịu được phép chiếu, kho kiến thức còn
  ```
  Đây chính là cơ chế `BitLinear` + `_ste_weight` trong `src/bitnet.py` — model 152M
  của dự án đã sống bằng nó 16.400 step. Khác duy nhất: init từ Qwen thay vì random.
- Mốc nội bộ liên quan: model train-ternary có 34,6% zero và **số 0 là thiết yếu**
  (bỏ đi → câu dài sập còn 1%); đổi công thức lượng tử trên model đã train → 80%→13%.
  ⇒ model PHẢI được học với ĐÚNG công thức quant sẽ dùng lúc inference.

## 1. Pipeline 6 bước

1. **Phẫu thuật** (`scripts/qwen_bitnet_surgery.py` — đã viết, chạy được):
   thay 168 `nn.Linear` (q/k/v/o/gate/up/down × 24 block) bằng `BitLinearB`
   (BitLinear của dự án + GIỮ BIAS fp32 — Qwen q/k/v CÓ bias, BitNet chuẩn không,
   bias không quantize). Copy nguyên weight FP16 vào master. Giữ nguyên
   embedding + lm_head (tied) + mọi norm.
2. **Chèn sub_norm**: `attn_sub_norm` RMSNorm(896) trước o_proj, `ffn_sub_norm`
   RMSNorm(4864) trước down_proj — **khởi tạo weight=1** (gần identity, không phá
   hành vi đầu). Cùng thủ thuật "chèn identity" đã dùng khi grow 12L→18L.
3. **Activation quant 8-bit** per-token absmax (`activation_quant` sẵn trong
   BitLinear) — bật cùng weight quant từ đầu, vì inference sẽ dùng cả hai.
4. **Loss = self-distillation**: teacher = Qwen FP16 gốc ĐÓNG BĂNG (fp16,
   eval-only), student = bản phẫu thuật.
   `loss = KL(log_softmax(student/T) || softmax(teacher/T))·T² + 0,5·CE(student, ids)`
   (T=2). Data: text thô ja/vi/en bất kỳ, không cần nhãn — smoke test dùng
   ~50–100MB (HF `wikimedia/wikipedia` subset ja + vi stream, hoặc copy 100k dòng
   từ `kd_v6_merged.jsonl` nếu máy có data dự án).
5. **Train ngắn có gate**: LR 1e-4 cosine, warmup 200 step, seq 512.
   **Đo ppl held-out mỗi 250–500 step** (giữ 2k câu không train).
   - Kỳ vọng nếu warm-start CÓ giá trị: ppl rơi thẳng đứng vài trăm step đầu
     (từ vùng "tệ hơn random" về vài lần FP16) — mạng chỉ "chỉnh biên" quanh Δ.
   - **LUẬT DỪNG: sau 2.000–3.000 step ppl chưa xuống < 2× baseline FP16
     (baseline đã đo: ppl_JA 63,9 / ppl_VI 52,2 trên dev dự án) → giả thuyết
     warm-start THẤT BẠI → dừng, không đốt thêm.**
6. **Nếu gate qua**: so đầy đủ với BitNet-152M của dự án (bench dịch, judge mù
   cùng phiên). Deploy i2_s là việc RIÊNG: bitnet.cpp chỉ có graph
   `bitnet-b1.58`, Qwen-ternary cần viết thêm graph — giai đoạn nghiên cứu đo
   bằng PyTorch là đủ.

## 2. Cấu hình cho 3060 Ti (8GB VRAM, 48GB RAM, WSL2)

| thành phần | ước VRAM |
|---|---:|
| student 0.5B master BF16 | ~1,0GB |
| gradient BF16 | ~1,0GB |
| Adam 8-bit (bitsandbytes) | ~1,0GB |
| teacher 0.5B FP16 (eval-only, no-grad) | ~1,0GB |
| activations @ seq 512, micro-batch 2–4, grad-ckpt | ~1,5–2,5GB |
| **tổng** | **~5,5–6,5GB → vừa 8GB** |

- Micro-batch 2–4 × grad-accum 32–64 → ~0,25–0,5M token/step hiệu dụng.
- BF16 autocast (Ampere OK). Gradient checkpointing bật nếu OOM.
- Tốc độ dự kiến trên 3060 Ti: chậm hơn L40S ~4–6× → 2.000 step có thể mất
  6–12 giờ. **Smoke sơ bộ chỉ cần 300–500 step** để thấy hình dạng đường ppl —
  vài giờ là đủ ra quyết định. Nếu hình dạng tốt → chuyển Modal
  (~$2–3/1000 step cho 0.5B, gate 3k step ≈ $6–10).
- HF cache: laptop đã tải model ở `D:/Bit-Translate-data/hf_cache`; desktop tải
  lại (~1GB): `HF_HOME=<ổ rộng> python -c "from transformers import AutoModelForCausalLM as M; M.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"`.

## 3. Ba con số quyết định (chưa ai biết — thí nghiệm này đo)

1. **Tốc độ hồi ppl 500 step đầu** — hồi nhanh = warm-start có giá trị thật;
   ì = train from-scratch trá hình, dừng.
2. **Điểm hội tụ so BitNet-152M train xịn** — nếu 0.5B-ternary thua model 152M
   hiện có thì hướng này vô nghĩa với dự án.
3. **Sub_norm identity có ổn với activation quant không** — nếu loss nổ ở vài chục
   step đầu: freeze mọi thứ trừ 2 sub_norm trong 200–300 step đầu rồi mở dần.

## 4. Việc KHÔNG làm

- KHÔNG "cải tiến công thức" data-free (đã chứng minh bão hoà tại per-row).
- KHÔNG khởi động từ bản đã ternary hoá (init tệ hơn random — đã đo).
- KHÔNG viết lại BitLinear/STE — dùng nguyên `src/bitnet.py`.
- KHÔNG lo deploy trước khi gate ppl qua.

## 5. Trạng thái file

- `scripts/qwen_bitnet_surgery.py` — phẫu thuật + smoke forward, chạy CPU/GPU.
- `eval/ternary_lab/` — số liệu nấc A (đối chứng bắt buộc trích dẫn khi báo cáo).
- Train loop QAT-KD: **CHƯA VIẾT** — viết trên máy GPU (transformers +
  bitsandbytes; vòng loss KL+CE ~80 dòng; tái dùng ý tưởng gate/dev-eval từ
  `scripts/train.py --dev-every`).
