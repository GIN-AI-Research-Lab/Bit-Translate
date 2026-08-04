# HƯỚNG 1 (PTQ-ONLY): tăng tốc runner MoE 30B-A3B bằng cách port 2 kỹ thuật đã validate ở 0.6B

**Ngày**: 2026-08-04 · **Trạng thái**: thiết kế, CHƯA code · **Ràng buộc**: PTQ-only tuyệt đối —
không backprop qua toàn model, không KD/teacher-student nhiều bước, không cần GPU để chạy
runner (chỉ cần GPU/CPU cho bước export/calibrate 1 lần, giống `export_embed_int8.py` cũ).
**Tiền đề**: `RESEARCH_TQ33_SPEED_100.md` (0.6B, đã XONG) + `RESEARCH_TQ33_RUNNER_30B.md` +
README.md Bài 16–19 (30B-A3B) + đọc trực tiếp `qwen3moe_runner_tq33.c` hôm nay.

> Đây KHÔNG phải ý tưởng mới — là port 2 kỹ thuật ĐÃ CHỨNG MINH đúng/an toàn ở 0.6B sang
> runner 30B, nơi cả hai vẫn CHƯA được áp. Rủi ro kỹ thuật thấp vì phương pháp đã validate;
> rủi ro chính là công sức port + cần checkpoint 30B (hiện không có ở máy A).

---

## 0. Tóm tắt

Runner MoE hiện tại (`qwen3moe_runner_tq33.c`, Bài 19) đạt tốc độ bền vững 13,0–14,4 tok/s
@8+ luồng — KHÔNG nhanh hơn baseline Q4_K_XL/IQ1_S (~16,0–16,5 tok/s) cùng máy, dù nhẹ hơn
nhiều (6,88GB vs 8,4–16,5GB). Đọc trực tiếp source hôm nay xác nhận **2 lỗ hổng tốc độ y hệt
0.6B TỪNG CÓ và đã sửa xong, nhưng CHƯA port sang runner 30B**:

1. **lm_head + embed_tokens giữ bf16 nguyên, KHÔNG lượng tử hóa** (comment trong code:
   *"embed/lm_head/router KHONG luong tu hoa — giu bf16 de KHONG tang gap doi"*), và với
   30B là **2 tensor RIÊNG** (`tie_word_embeddings=false`, khác 0.6B tied) — tệ hơn 0.6B vì
   2 ma trận lớn thay 1. Bài 19 tự ghi lm_head bf16 chiếm **18–37% thời gian/token**.
2. **Threading vẫn dùng Event** (`CreateEvent`/`SetEvent`, dòng 459–522) — ĐÚNG pattern cũ
   mà `RESEARCH_TQ33_SPEED_100.md` §3.4 đã đo tốn 5–20µs/lần × hàng trăm lần/token, và fix
   bằng spin-barrier+padding cache-line đã nâng runner 0.6B 4T từ 57–60 lên 71–80 tok/s.

Cả hai đều là kỹ thuật **PTQ/kernel thuần** (không train), đã validate đúng ở 0.6B, chỉ cần
port. Ước tính thô (§3) gộp cả hai: **+30–70% tok/s** cho runner MoE, tùy tỉ trọng thời gian
lm_head thật đo trên 30B (chưa đo trực tiếp, chỉ có ước tính Bài 19).

---

## 1. Việc 1 — int8 per-row cho lm_head + embed_tokens (ưu tiên cao nhất, lợi lớn nhất)

### 1.1 Khác biệt so với 0.6B cần lưu ý

- 0.6B: `tie_word_embeddings=true` → 1 ma trận [151936×1024] dùng cho cả input lookup và
  output head → nén 1 lần dùng 2 chỗ.
- 30B-A3B: `tie_word_embeddings=false` → **2 ma trận riêng**. Cần export/validate riêng
  từng cái. Vocab/hidden size của 30B chưa xác nhận lại ở máy A — bước đầu tiên là đọc
  `config.json` thật (đã có sẵn quy trình trong `export_embed_int8.py`, chỉ cần tổng quát
  hóa từ `models--Qwen--Qwen3-0.6B` sang đường dẫn checkpoint 30B).
- Router (`block_sparse_moe.gate`) nhỏ (128×hidden hoặc tương đương) — GIỮ bf16, không đáng
  nén (bài học từ chính comment code: router lượng tử hóa rủi ro lật routing, không đáng
  ~vài trăm KB tiết kiệm được).

### 1.2 Phương pháp (nguyên bản `export_embed_int8.py`, tổng quát hóa)

- Symmetric per-row: `scale[v] = max|W[v,:]| / 127`, clip ±127 (tránh ±128 saturate maddubs,
  đúng lý do đã ghi trong RESEARCH_TQ33_SPEED_100.md §2).
- Validate TRƯỚC khi đụng runner C (numpy, trên oracle thật đã có sẵn hạ tầng
  `make_oracle_multi.py`/`ref_forward_pytorch.py` — chỉ cần trỏ sang ckpt 30B):
  - rel-err logits so oracle FP.
  - top-1/top-5 tại vị trí cuối trên ≥3 prompt thật (2 VI 1 JA, giữ đúng bộ đã dùng ở 0.6B
    để so sánh được chuẩn nhiễu).
  - **Ngưỡng PASS giữ nguyên chuẩn 0.6B đã chấp nhận**: rel-err logits thêm vào do int8-head
    phải NHỎ HƠN nhiễu TQ33 sẵn có của phần linear (so với oracle, không so với bf16-head).
    Nếu KHÔNG đạt (30B nhạy hơn vì lm_head untied lớn hơn/khác phân phối) → dừng, giữ bf16
    lm_head, chỉ nén embed_tokens (rẻ hơn, ít rủi ro hơn vì chỉ ảnh hưởng input, được
    RMSNorm layer 0 hấp thụ như đã đo ở 0.6B mục 2 RESEARCH_TQ33_SPEED_100.md).

### 1.3 Kernel

Tái dùng `row_dot_i8_avx2`/`row_dot_i8_vnni` từ `qwen3_runner_tq33_fast.c` gần như nguyên
văn (đã có, đã kiểm chứng runtime bit-identical) — chỉ cần 2 instance (embed lookup 1 hàng,
lm_head full GEMV) thay 1, vì không tied.

## 2. Việc 2 — spin-barrier + padding cache-line thay Event (rủi ro thấp nhất, port cơ học)

Copy nguyên cơ chế từ `qwen3_runner_tq33_fast.c` (generation counter + `_mm_pause`, mỗi biến
nóng 1 cache-line riêng — bài học "false-sharing nâng 4T từ 57–60 lên 71–80 tok/s" của
RESEARCH_TQ33_SPEED_100.md §3.4). Runner MoE hiện threading theo EXPERT (không theo GEMV như
0.6B, đã là lựa chọn đúng riêng cho MoE — KHÔNG đổi phần này), chỉ đổi CƠ CHẾ đồng bộ
start/done giữa master–worker, không đổi cách chia việc theo expert. Rủi ro thấp vì đây là
thay đổi cơ học (sync primitive), không đụng số học — kiểm bit-identical trước/sau bằng
đúng oracle 48/48 layer đã có (`validate_30b_layers_compare.py`).

**Cờ dự phòng bắt buộc theo đúng luật lab** (RESEARCH_TQ33_SPEED_100.md §3.5: đừng tự tin
work-stealing/affinity mù) — GIỮ tĩnh theo expert như hiện tại, chỉ đổi sync primitive, không
thử work-stealing/affinity cho MoE ở vòng này (chưa có bằng chứng cho MoE, khác cấu trúc
GEMV nhỏ của 0.6B).

## 3. Ước tính lợi ích (kỳ vọng ghi trước, không đoán mò)

| Nguồn | % thời gian hiện tại (Bài 19) | Hệ số cải thiện | Lợi ước tính |
|---|---:|---:|---|
| lm_head+embed bf16→int8 | 18–37% | ~4× (622MB→~156MB×2 tensor, tỉ lệ 0.6B) | tổng thời gian giảm ~13–28% → **+15–39% tok/s** |
| Event→spin-barrier | chưa đo riêng cho MoE (0.6B đo được +25–40% ở 4T) | — | **+10–25% tok/s** (thận trọng hơn 0.6B vì MoE ít điểm sync/token hơn — 1 lần/expert-batch, không phải 141/token) |
| **Gộp cả hai** | — | — | **+25–70% tok/s**, mục tiêu thực dụng ~13-14 → **17-22 tok/s** |

**Kill criteria**: nếu sau khi port cả hai mà tok/s KHÔNG tăng ≥15% → dừng, nghi ngờ bottleneck
thật nằm ở chỗ khác (expert routing/decode TQ33 compute-bound, đúng bài học "TQ33-active thật
~486MB/token không phải 10-15MB ước tính ban đầu sai 32 lần" — Bài 19) → đo lại breakdown
thời gian bằng timer per-phase (embed/attn/expert/head) TRƯỚC khi tối ưu thêm.

## 4. Việc CẦN có trước khi làm (chưa có ở máy A)

- Checkpoint 30B TQ33 (6,88GB, `expv_Qwen3-30B-A3B_2x4.pt`/tương đương) — hiện trên Modal
  volume `tritue12`/`free30`, CHƯA tải về máy A. Cần tải trước khi export/validate.
- `config.json` thật của Qwen3-30B-A3B để xác nhận vocab/hidden/lm_head shape (không đoán
  theo 0.6B).
- Oracle 30B đã có sẵn từ Bài 19 (validate 48/48 layer) — TÁI DÙNG, không cần sinh lại.

## 5. Bản đồ file dự kiến

| File | Vai trò | Trạng thái |
|---|---|---|
| `export_embed_int8_moe.py` (mới, tổng quát hóa từ `export_embed_int8.py`) | export int8 embed+lm_head riêng cho 30B | chưa code |
| `qwen3moe_runner_tq33_fast.c` (mới, fork từ `qwen3moe_runner_tq33.c`) | + int8 head/embed + spin-barrier | chưa code |
| `compare_phase3_moe.py` (tổng quát hóa `compare_phase3.py`) | validate rel-err/top-k so oracle 30B | chưa code |

Không đụng `qwen3moe_runner_tq33.c` gốc (giữ nguyên để so sánh/rollback, đúng quy ước lab).
