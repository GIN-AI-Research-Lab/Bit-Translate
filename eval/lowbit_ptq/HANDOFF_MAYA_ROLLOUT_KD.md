# HANDOFF — Rollout-KD × Bit-Curriculum (sub-1.58bit) — chạy trên Máy A (3060 Ti 8GB)

**Ngày**: 2026-08-04 · **Từ**: phiên Claude máy B (laptop, không CUDA) · **Cho**: phiên chạy trên
máy A (desktop, RTX 3060 Ti 8GB, WSL2 — xem CLAUDE.md §2).

**Vì sao chuyển máy**: gradient training (QAT/KD) đo THẬT trên CPU máy B quá chậm để dùng thật
(S1 calib 5 câu mất ~47 phút; S2 KD ~34s/bước) — một chuỗi 4 bậc curriculum thật (400-1800
bước/bậc) sẽ mất nhiều ngày trên CPU. Máy A có GPU thật (CUDA) nên khả thi.

**Đọc trước khi chạy**: `RESEARCH_NOVEL_SUBBIT_PLAN.md` (cùng thư mục) — file đó là thiết kế đầy
đủ + bằng chứng + tiêu chí PASS/FAIL ghi trước. File NÀY chỉ là hướng dẫn thao tác cụ thể, không
lặp lại nội dung thiết kế.

---

## 0. Tóm tắt ý tưởng (3 dòng, xem chi tiết ở RESEARCH_NOVEL_SUBBIT_PLAN.md §0-1)

"Lời nguyền dưới 2-bit" không nằm ở trọng số (đã vét cạn bằng PTQ) mà ở **hàm mục tiêu**: mọi KD
trước giờ là teacher-forced trên ngữ cảnh vàng, trong khi lỗi thật (loop khi tự sinh) chỉ lộ ra
trên ngữ cảnh TỰ SINH (exposure bias, bị lượng tử hóa khuếch đại). Giải pháp thử: **Rollout-KD
(on-policy/DAgger) + bit-curriculum bằng họ mask N:M lồng nhau** (0.331→1.564bpw, mask chỉ THÊM ô
khi lên bậc, không refresh). Đã toy-verify $0 trên máy B (4/4 gate pass) + smoke CPU chạy hết
5 bậc không lỗi. **Chưa chạy thật trên GPU nào cả** — đây là việc của máy A.

---

## 1. Việc ĐÃ XONG ở máy B (không cần lặp lại)

| Việc | File | Trạng thái |
|---|---|---|
| Toy-verify thiết kế (nested mask, masked-STE, rollout-KD toy, router-margin) | `exp_aa_subbit_toys.py` + `exp_aa_results.json` | 4/4 PASS |
| Probe exposure-bias trên model thật (FP/int4/PTQ-ternary) | `exp_ac_divergence_probe.py` + `exp_ac_results.json` | Chạy CPU thật, 3 chế độ tách bạch rõ |
| Trainer chính (curriculum × rollout × highway + battery + divergence) | `exp_ab_subbit_curriculum.py` | Code xong, **smoke CPU** (2 bước/bậc, `exp_ab_results.json`) chạy hết 5 bậc không crash — **CHƯA chạy thật (steps thật) trên bất kỳ máy nào** |
| App Modal (P0 probe + 4 arm + gate), idempotent, `--detach` | `cloud/modal_qat_subbit.py` | Sẵn sàng, **chưa chạy** — cần nạp credit (4 ví đều cạn) |
| Kế hoạch chạy + tiêu chí PASS/FAIL ghi trước + rủi ro | `RESEARCH_NOVEL_SUBBIT_PLAN.md` | Đầy đủ |

Đã verify trước khi bàn giao: cả 4 file Python compile sạch (`py_compile`), `modal_qat_subbit.py`
không tự chạy gì khi import (mọi hàm nặng đều sau `@app.function`, phải gọi `modal run` mới chạy),
không có credential lộ.

---

## 2. Việc CẦN LÀM ở Máy A — **ưu tiên chạy LOCAL trên GPU 3060 Ti trước, KHÔNG cần Modal**

**Điểm mấu chốt hay bị bỏ sót**: `exp_ab_subbit_curriculum.py` chạy ĐỘC LẬP, tự làm S1 (nén
gauge+Wanda+sequential) từ model FP gốc — **KHÔNG cần checkpoint GEN4 có sẵn trên Modal volume**.
3060 Ti có CUDA thật (khác máy B) → chạy được thật 100% MIỄN PHÍ, không cần đợi nạp Modal credit.
Modal (`cloud/modal_qat_subbit.py`) chỉ là lựa chọn nếu muốn chạy song song nhiều arm hoặc 3060 Ti
quá chậm — không bắt buộc.

### Bước 1 — chuẩn bị môi trường (trong WSL2, theo CLAUDE.md §2)

```bash
git pull origin main
cd eval/lowbit_ptq
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# kỳ vọng: True, 'NVIDIA GeForce RTX 3060 Ti'
```

Cần model Qwen3-0.6B tải sẵn (HF cache) và data đã có sẵn trong repo (`cloud/qat_data/train_slice.{vi,ja}`,
`dev.{vi,ja}` — đã commit, ~10MB). Nếu HF cache path khác máy B, sửa `--model-glob` hoặc set biến
môi trường trỏ đúng nơi (mặc định code trỏ `D:\Bit-Translate-data\hf_cache\...` — đường Windows,
đổi path tương ứng nếu máy A tổ chức khác).

### Bước 2 — SMOKE THẬT TRÊN GPU trước (bắt buộc, ~5-10 phút, kiểm VRAM 8GB có đủ không)

Đây là bước **CHƯA từng được đo** (smoke cũ chỉ chạy CPU). VRAM 8GB train full W+bias (không phải
LoRA-adapter, đúng recipe GEN4 "EfficientQAT-style") có thể SÁT NGƯỠNG với batch 16 — kiểm trước:

```bash
python exp_ab_subbit_curriculum.py --device cuda --smoke
```

Theo dõi `nvidia-smi` song song. Nếu OOM: giảm `--batch` (mặc định 16 → thử 4 hoặc 8) và/hoặc
`--seq` (mặc định 128 → 64), hoặc bật gradient checkpointing nếu code hỗ trợ (chưa có cờ riêng —
cần thêm nếu OOM dai dẳng, xem CLAUDE.md §7 mục 4). Nếu smoke pass: đọc `exp_ab_results.json` mới
đè lên, kỳ vọng cấu trúc giống bản CPU đã có (numbers vô nghĩa vì chỉ 2 bước/bậc — đúng, không phải bug).

### Bước 3 — chạy 2 arm rẻ nhất để có tín hiệu sớm về ý chính (E — rollout-KD)

Không cần chờ curriculum (D̃) — câu hỏi quan trọng nhất là "rollout-KD có ăn không", đo bằng
`ab_base` (2:4 thẳng, không rollout) vs `ab_roll` (2:4 + rollout). Cùng cấu hình `_run_arm` trong
`cloud/modal_qat_subbit.py` (copy nguyên, chỉ đổi `--device cuda` chạy local):

```bash
# A1 base (mốc, không rollout)
python exp_ab_subbit_curriculum.py --device cuda \
  --ladder 2:4 --stage-steps 4000 --rollout 0 \
  --train-vi cloud/qat_data/train_slice.vi --train-ja cloud/qat_data/train_slice.ja \
  --dev-vi cloud/qat_data/dev.vi --dev-ja cloud/qat_data/dev.ja \
  --batch 16 --seq 128 --lr 2e-5 --kd-temp 2.0 --ce-w 0.1 --perm-gauge 1 \
  --tag ab_base_maya --out D:/Bit-Translate-data/qat_ckpts/ab_base_maya.pt

# A2 rollout (cùng cấu hình + rollout-KD)
python exp_ab_subbit_curriculum.py --device cuda \
  --ladder 2:4 --stage-steps 4000 --rollout 1 \
  --train-vi cloud/qat_data/train_slice.vi --train-ja cloud/qat_data/train_slice.ja \
  --dev-vi cloud/qat_data/dev.vi --dev-ja cloud/qat_data/dev.ja \
  --batch 16 --seq 128 --lr 2e-5 --kd-temp 2.0 --ce-w 0.1 --perm-gauge 1 \
  --tag ab_roll_maya --out D:/Bit-Translate-data/qat_ckpts/ab_roll_maya.pt
```

Thêm `--kd-en/--kd-code/--kd-zh/--kd-math` nếu muốn KD-mix đầy đủ (cần chuẩn bị 4 file text —
xem `_prep_kd_mix()` trong `modal_qat_subbit.py` để biết nguồn HF dataset; bỏ qua nếu chỉ muốn
test riêng vi/ja trước, rẻ hơn).

**Ước tính thời gian (KHÔNG phải đo thật — suy từ tỉ lệ compute L40S/3060 Ti, coi là thô)**:
Modal L40S chạy arm này ~2.5-4h. 3060 Ti chậm hơn đáng kể (ít VRAM hơn có thể ép batch nhỏ hơn,
compute thô cũng thấp hơn nhiều lần) — dự đoán rất thô **~12-30h/arm**, tức khả thi chạy qua đêm/
vài ngày (đúng kỳ vọng CLAUDE.md §3 "nhiều ngày tới vài tuần"), nhưng CẦN đo lại 50-100 bước đầu
rồi ngoại suy thời gian thật trước khi cam kết chạy hết 4000 bước.

### Bước 4 — đọc kết quả theo tiêu chí ghi trước (KHÔNG đổi tiêu chí sau khi thấy số)

Copy nguyên từ `RESEARCH_NOVEL_SUBBIT_PLAN.md` §3.4 — chỉ nhắc lại câu hỏi chính:

| Câu hỏi | PASS | FAIL |
|---|---|---|
| E ăn không? (roll vs base) | pass-rate battery +≥8đ tuyệt đối HOẶC loop-rate −≥30% tương đối; geo6 không tệ hơn >10% | báo cáo âm tính, giữ loop-cut/divergence làm công cụ đo |

`exp_ab_subbit_curriculum.py` tự ghi `battery` (pass-rate/loop-rate/transcripts) và `divergence`
(self-KL vs TF-KL theo bucket) vào JSON kết quả — **đọc transcripts bằng mắt** (CLAUDE.md §9),
đừng chỉ tin con số PPL/pass_rate.

### Bước 5 — nếu E có tín hiệu tốt, chạy tiếp curriculum (D̃) và/hoặc cân nhắc Modal

Nếu 3060 Ti chạy 1 arm mất >2-3 ngày và muốn chạy 4 arm song song hoặc nhanh hơn, dùng
`cloud/modal_qat_subbit.py` (đã sẵn, xem lệnh trong docstring đầu file) — nhưng **cần nạp credit
Modal trước** (04/08: cả 4 ví tuent1997/tritue12/nguyentuanngai/free30 đều cạn).

---

## 3. Rủi ro/giới hạn cần nhớ khi đọc kết quả (rút gọn từ plan §6 — đọc bản đầy đủ để hiểu rõ)

1. **Capacity wall 0.6B**: 0.6B là ca khó nhất lab tự chọn cho sub-2bit — khả năng thật là
   rollout-KD giảm loop rõ nhưng pass-rate vẫn xa FP/Q4. Giá trị vòng này khi đó = cơ chế đã
   chứng minh + công thức mang lên 30B (nơi capacity đủ hơn).
2. **1 seed, battery 22 prompt**: đủ cho hiệu ứng lớn, không đủ cho hiệu ứng nhỏ — chênh <8 điểm
   coi là "chưa kết luận", không phải "âm tính".
3. **VRAM 8GB CHƯA từng test thật** với recipe train full-W+bias (không LoRA) — bước 2 (smoke
   GPU) là để lộ vấn đề này SỚM, đừng bỏ qua.
4. **Bias trong checkpoint**: đóng gói TQ33 cuối cùng cần runner cộng bias/hàng (vài dòng C,
   chưa làm) — đừng ép bỏ bias hậu kỳ (F0: mất ×4000, xem memory lab).

---

## 4. Bản đồ file đầy đủ (đã commit cùng đợt với file này)

```
eval/lowbit_ptq/RESEARCH_NOVEL_SUBBIT_PLAN.md      # thiết kế đầy đủ, đọc trước
eval/lowbit_ptq/HANDOFF_MAYA_ROLLOUT_KD.md         # file này — hướng dẫn thao tác
eval/lowbit_ptq/exp_aa_subbit_toys.py              # 4 toy-gate (đã chạy, xem .json)
eval/lowbit_ptq/exp_aa_results.json
eval/lowbit_ptq/exp_ac_divergence_probe.py         # probe exposure-bias (đã chạy CPU, xem .json)
eval/lowbit_ptq/exp_ac_results.json
eval/lowbit_ptq/exp_ab_subbit_curriculum.py        # TRAINER CHÍNH — chạy ở bước 2-3 trên
eval/lowbit_ptq/exp_ab_results.json                # kết quả SMOKE CPU (2 bước/bậc, không phải kết quả thật)
cloud/modal_qat_subbit.py                          # app Modal, dự phòng nếu cần scale/song song
cloud/qat_data/{train_slice,dev}.{vi,ja}           # data (đã có sẵn trong repo)
```

Không file nào trong đợt này đụng tới TQ33/kernel tốc độ (nhánh đó độc lập, đã xong — xem
`RESEARCH_TQ33_SPEED_100.md`).
