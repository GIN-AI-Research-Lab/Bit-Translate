# HANDOFF — Nghiên cứu low-bit/sub-1-bit + kế hoạch chạy trên Laguna S2.1 (máy khác)

> **CẬP NHẬT 02/08 sáng — đường biên thế hệ 2 (xem README Bài 10 cho bảng đầy đủ):**
> 1.94bpw→471 · **1.56→400 (kỷ lục, 5.8×FP)** · 1.02→767 · 0.70→1.055 · 0.62→1.843.
> Công thức thắng: gauge (up↔down, v↔o; sanity-check FP bắt buộc) → Wanda mask CỐ ĐỊNH →
> sequential 2-pass, scale+bias HỌC ĐƯỢC, norm co-tune, guard step-0 → polish (adapter+tail-KL)
> CHỈ khi bpw<1. Đã bác bằng ablation: mask refresh giữa pass (×2.2 hại), relative-MSE (hại ja).
> Script chuẩn: `exp_q_gauge_bias_budget.py` (≥1.5bpw) + `exp_p_tail_polish.py` NOREFRESH NORELMSE (<1bpw).
> Lưu ý Laguna: gauge up↔down áp cho TỪNG expert (256×48); v↔o cần map GQA 72/8 head;
> QK-norm Qwen3-Next của Laguna → KHÔNG gauge q/k. Ước tính có kiểm chứng từ dưới bằng thang IQ (L1).

> Cập nhật 2026-08-01 18:25, máy B. Người nhận: phiên làm việc trên máy khác (mạnh hơn),
> mục tiêu: lặp lại thang thí nghiệm trên **Laguna S2.1 (MoE 118B, 14.5B active)**.
> Toàn bộ chi tiết từng bài: `eval/lowbit_ptq/README.md` (Bài 0–7). File này là bản đồ hành động.

## 1. Kết quả đã chốt trên Qwen3-0.6B (mốc để so với Laguna)

Eval: PPL trên dev vi/ja held-out (`D:\Bit-Translate-data\clean_v7g\dev.{vi,ja}`), FP32 = vi 69.0 / ja 125.1.
Mọi format: reconstruction (STE, track-best, không train toàn model), kế toán bit ĐỦ (payload+mask+scale).

| Format | bpw thật | PPL vi | Ghi chú |
|---|---:|---:|---|
| int4-g32 | 4.50 | 86 | ~FP, trần không-train = Q4 có sẵn |
| dense ternary + fix-pack | 2.08 | 4.401 | fix-pack ăn 2.6× so absmean/80step |
| ternary 2:4 Wanda (exp_i, 80st) | 1.94 | 6.833 | sweet-spot; đang kiểm lại ở exp_k |
| ternary 1:4 | 1.40 | 30.506 | |
| binary 2:4 | 1.65 | 33.354 | structured cứu binary 56× |
| binary dense | 1.50 | 1.861.464 | mất mức 0 = thảm họa |
| IQ1_S llama.cpp + imatrix | 1.56 | 20.492 | PTQ codebook xịn nhất cũng sụp |

**5 bài học chuyển giao được sang model bất kỳ:**
1. **Mức 0 là vua**: binary (bỏ 0) tệ hơn sparse-ternary 50× dù nhiều bit hơn.
2. **Mask cấu trúc N:M > mask tự do** cả về bit (0.5 vs 0.81) lẫn chất lượng (giữ năng lực đều mỗi group).
3. **Fix-pack bắt buộc**: scale Lloyd trên survivor (KHÔNG absmean cả group), mask Wanda `|W|·‖X‖`
   (không phải |W| thuần), ≥100 step, track-best (LR cao không track-best → tệ hơn cả round!).
4. **Teacher-forcing là chặn trên lạc quan** — sai số ~20%/lớp × 28 lớp = PPL nổ; sequential/block-wise
   (BRECQ-lite, exp_k arm2) là cơ chế bù liên lớp duy nhất không cần train.
5. **Định luật kích thước**: 0.6B là ca khó nhất (ít dư thừa chức năng). STBLLM 0.55bit sống trên 7B
   (PPL 31.7) trong khi 0.6B chết ở 1.4bit. → **Laguna 118B là NƠI ĐÚNG để thử sub-2-bit** — đây
   chính là lý do khoa học của việc chuyển máy.

## 2. Thí nghiệm ĐANG CHẠY trên máy B (đừng tắt máy trước ~21:30!)

| Job | Nội dung | Kết quả rơi vào |
|---|---|---|
| exp_k (3 arm) | fix-pack TF dense/2:4 + **sequential block-wise** | `eval/lowbit_ptq/exp_k_results.json` (ghi tiến dần) |
| exp_l (6 arm, tự nối đuôi exp_k) | thang **sub-1-bit thật**: scale tensor/row/f8-g64 × mask 1:4/1:8/2:8 × ternary/binary (0.70–1.58 bpw) | `eval/lowbit_ptq/exp_l_results.json` + `exp_l_run.log` |

Đã có: exp_k arm1b dense fix-pack 2.08bpw → vi 4.401/ja 8.250. Nếu tắt máy giữa chừng: JSON giữ
các arm đã xong (bài học `chuoi-nen-chet-khi-tat-may` — job nền chết âm thầm khi shutdown/sleep).

## 3. Kho file (tất cả trong `eval/lowbit_ptq/`)

| File | Vai trò |
|---|---|
| `README.md` | Giáo trình 7 bài + số liệu đầy đủ + đính chính kế toán bit |
| `exp_b_weight_error_table.py` | sàn rate-distortion trên trọng số thật |
| `exp_c_incoherence.py` | Hadamard thật vs giả |
| `exp_e_ste_qat_toy.py` | vì sao QAT cứu được ternary (PTQ 52%→QAT 21% output err) |
| `exp_f/g/h/i` | reconstruction toàn model / mixed-precision / sub-1.58 / STBLLM-style |
| `exp_k_fixpack_sequential.py` | **fix-pack + sequential BRECQ-lite** — file nền để port |
| `exp_l_true_sub1bit.py` | **thang sub-1-bit kế toán đủ** — file nền để port |
| `inspect_laguna_static.py` | **đã chạy được trên file 89GB** — mẫu đọc/dequant streaming |
| `inspect_architectures.py`, `inspect_weight_geometry.py` | mổ kiến trúc + hình học SVD |

Text calib/eval dùng chung (copy sang máy mới): `D:\Bit-Translate-data\ptq_ladder\{calib.txt,test_vi.txt,test_ja.txt}`
(vi+ja từ corpus dự án; imatrix thì phải chạy LẠI trên chính Laguna — imatrix là per-model, text thì tái dùng).

## 4. KẾ HOẠCH LAGUNA S2.1 trên máy khác

Model: `D:\Bit-Translate-data\models\Laguna-S-2.1-Q4_K_M.gguf` (89.44GB, arch=`laguna`,
48 block, 256 expert dùng 10/token, hidden 3072, vocab 100352; attention Q8_0, expert Q4_K).
Nguồn gốc HF: `bartowski/Laguna-S-2.1-GGUF`.

### Phase L1 — llama.cpp stock (GIÁ TRỊ CAO NHẤT, làm trước)
Trả lời trực tiếp: *"118B có cứu sub-2-bit không?"* bằng tooling sẵn, không cần code mới.

1. **Build/tải llama.cpp MỚI NHẤT** — b10199 trên máy B KHÔNG nhận arch `laguna`; bản
   quantize được của bartowski chứng tỏ mainline mới có hỗ trợ. Kiểm: `llama-cli -m <gguf> -p "hi" -n 8`.
2. **Baseline PPL Q4_K_M** trên `test_vi.txt`/`test_ja.txt` (`llama-perplexity -m ... -f ... -c 512`).
   Đây là mốc "FP" thực dụng của Laguna.
3. **imatrix**: `llama-imatrix -m <Q4 gguf> -f calib.txt -o imat.gguf -c 512` — bước NẶNG NHẤT
   (forward 118B trên ~180k token; cần GPU offload hoặc kiên nhẫn; MoE cần calib đủ dài để mọi
   expert được kích hoạt — nếu thiếu, tăng calib lên 1–2MB text).
4. **Thang lượng tử + PPL từng mức** (`--allow-requantize --imatrix imat.gguf`):
   `IQ2_XS (2.31) → IQ2_XXS (2.06) → IQ1_M (1.75) → IQ1_S (1.56) → TQ2_0 (2.06 ternary) → TQ1_0 (1.69)`.
   Disk: mỗi bản ~13–34GB → làm tuần tự, xóa dần, chừa ~40GB trống.
5. **So đường cong với Qwen 0.6B** (bảng §1): nếu IQ1_M/IQ1_S trên Laguna chỉ tăng PPL vài chục %
   thay vì ×300 như 0.6B → định luật kích thước xác nhận, sub-2-bit KHẢ THI trên Laguna bằng tooling sẵn.

**Dự đoán ghi trước** (để đối chiếu, tránh tự huyễn): IQ2_XS gần như chắc "thở được" (top model 100B+
thường chịu 2.3bpw tốt); IQ1_M ranh giới (kỳ vọng tăng PPL 1.5–3×); TQ1_0/TQ2_0 (RTN ternary, không
imatrix-aware bằng IQ) nhiều khả năng vẫn xấu. Cẩn thận: PPL của MoE nhạy với routing — so bằng CÙNG
test file, cùng ctx.

### Phase L2 — port format của ta lên expert Laguna (research-grade)
Bulk của Laguna là expert (Q4_K): mục tiêu ternary-2:4-fix-pack hóa EXPERT, giữ attention Q8_0.

- **Đọc/dequant từng expert streaming** (không cần RAM lớn): mẫu code ĐÃ CHẠY ĐƯỢC trong
  `inspect_laguna_static.py` (gguf.GGUFReader + `gguf.quants.dequantize`; tensor 3D → lặp dim 0).
- Port từ `exp_l_true_sub1bit.py`: `q_ternary/q_binary` (Lloyd survivor, scale group tham số hóa),
  `wanda_nm_mask`, `tf_reconstruct`. Chúng là hàm thuần per-matrix — chạy được mọi nơi.
- **Activation cho Wanda/reconstruction**: không có runtime HF cho Laguna. Ba lựa chọn theo thứ tự:
  (i) nếu llama.cpp mới chạy được → không hook được activation, dùng **proxy Gaussian + outlier**
  (mask kém hơn Wanda thật một bậc — chấp nhận, ghi rõ); (ii) dựng lại pack expert thật cho
  `native_engine/engine_v1.py` (repack từ GGUF 89GB thay bin cũ đã xóa — xem
  `native_engine/repack_for_engine.py` + `KETQUA_ENGINE.md` §4); (iii) chỉ đo output-error per-expert
  với proxy X, bỏ PPL — nhanh nhất nhưng yếu nhất.
- **Đo PPL cho format tự chế**: cần engine tự viết (llama.cpp không có type ternary-2:4).
  Đường tắt hợp lệ: xuất bản "ternary 2:4 mô phỏng trong Q8_0" (dequant ternary rồi requantize Q8_0
  — giữ GIÁ TRỊ ternary, vỏ Q8) → llama.cpp chạy được → PPL đo được chất lượng format (tốc độ thì không).
  Đây là trick đáng làm nhất của L2: **PPL thật cho format tự chế mà không cần viết engine.**

### Phần cứng máy mới cần
- Disk trống ≥ 130GB (89 gốc + 1 bản quantize + imatrix + tmp). RAM ≥ 64GB cho PPL IQ1/IQ2 CPU
  (mmap giúp, nhưng MoE đọc 10 expert/token → SSD nhanh quan trọng). GPU ≥ 24GB VRAM giúp imatrix
  nhanh gấp nhiều lần (offload `-ngl`).

## 5. Sau khi có số Laguna — khép vòng
So 3 điểm: (0.6B: sụp ở <2bit) ↔ (Laguna 118B: ?) ↔ (paper 7B: STBLLM 0.55bit sống).
Nếu Laguna sống ở IQ1/ternary-2:4 → viết được kết luận nghiên cứu hoàn chỉnh về định luật
kích-thước-vs-bit trên dải 0.6B→118B, với số đo thật toàn bộ. Nếu không → nghi vấn MoE
(routing nhạy lượng tử hơn dense) — tự nó là một phát hiện.
