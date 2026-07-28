# ARTIFACTS — checkpoint & model đang ở đâu

> Bản đồ vị trí mọi checkpoint/model/data. **Cập nhật 2026-07-21.**
> ⚠️ Các file nặng dưới đây **KHÔNG nằm trong git** (`.gitignore` loại `checkpoints/ cloud_backup/ dist/ data/`).
> **Backup:** data + checkpoint train + model deploy đã có **bản trên GitHub Release** (mục áp cuối) — mất ổ local vẫn tải lại được.

## Checkpoint để TRAIN TIẾP (Phương án A - KD ja→vi từ v4)

| Model | Checkpoint Tag / Release | State File | Dùng để / Ghi chú |
|---|---|---|---|
| **100M v4** (bản mạnh nhất) | `autosave-100m-v4` (bản `v4_aa`) | `last.pt` @ step 15500 | **Mốc nền train KD GĐ1** (judge acc 2.78 ja→vi) ⭐ |
| **100M ft** (clean-finetune) | `autosave-100m-ft` | `last.pt` @ step 15600 | Milestone finetune thử nghiệm (+0.05 acc, overfit ở 15700) |
| **100M v3** | `autosave-100m-v3` | `last.pt` @ step 13000 | Milestone v3 cũ (judge acc 2.08) |
| **292M** (scale 300m) | `autosave-scale300m` (bản `p_aa`/`p_ab`) | `last_b.pt` @ step 30000 | **Nền 292M** (nếu mở lại scale; không dùng wave0/step32500 do overfit) |

## Data KD & Fix sạch (Sẵn có cho GĐ1/GĐ2)

| File / Asset | Vị trí / Tag | Size / Số lượng | Nội dung |
|---|---|---|---|
| **All KD Clean v4** | `data/synthetic/kd_clean/all_kd_v4.jsonl` | 32.390 câu | Data KD sạch đã qua QC LaBSE 0.55 + filter |
| **All Fix v3** | `data/synthetic/fix/all_fix_v3.jsonl` | 2.759 cặp | Data cặp fix lỗi |
| **JA Pool 841k** | `data/synthetic/kd/ja_pool.txt` | ~806k câu chưa dịch | Pool câu JA đơn ngữ làm nguồn dịch KD GĐ1/GĐ2 |
| **BT Vòng 3b** | Release `bt-vong3b` | 841.039 câu | Pair (VI 292M, JA gold) — nguồn câu JA |

## GitHub Release & Autosave Tags (Repo `trituenguyen97/Bit-Translate`)

- **`autosave-100m-v4`**: Backup 100M v4 step 15500 (nền KD GĐ1).
- **`autosave-scale300m`**: Backup 292M step 30000.
- **`v1.0-step14000`**: Model baseline deploy `vija-1p58-step14000-i2s.gguf` (68MB).
- **`train-assets-vong3a`**: Premix data vong 3a (23,45M seq).
- **`bt-vong3b`**: Output back-translation 841k câu.

## Cloud & Local Nodes
- Modal Cloud: Script training `cloud/modal_finetune_kd.py` & `cloud/modal_train_100m_v4.py`.
- Local i2_s CPU Runtime: 6 luồng WSL, ~321 tok/s inference, ~155ms/câu.


---

## VÒNG 6 (2026-07-28) — phân loại theo KHÔNG THỂ TÁI TẠO vs tái tạo được

Phân loại này quyết định cái gì phải đẩy lên Release, cái gì chỉ cần ghi lệnh tái tạo.

### ⛔ KHÔNG tái tạo được rẻ — PHẢI đẩy Release

| Artifact | Đường dẫn local | Nặng | Vì sao không tái tạo được |
|---|---|---:|---|
| **`v6_avg.pt`** | `D:/Bit-Translate-data/checkpoints_v6/v6_avg.pt` | 706 MB | Trung bình 7 mốc 15000-16400. Là **model tốt nhất hiện có**. Tái tạo = train lại 16.400 step ≈ **$16 + 6 giờ GPU** |
| **`kd_v6.jsonl`** | `D:/Bit-Translate-data/raw/kd_v6.jsonl` | 1,09 GB | 2.171.977 câu do thầy Gemini dịch. Tái tạo = **6 giờ + hết quota 6 key**. Đây là thứ đắt nhất của cả vòng |

### ✅ Tái tạo được — chỉ cần ghi lệnh, ĐỪNG đẩy Release

| Artifact | Nặng | Lệnh tái tạo | Thời gian |
|---|---:|---|---:|
| `kd_v6_merged.jsonl` | 4,6 GB | `python scripts/merge_v6.py` (cần `kd_v5_merged.jsonl` + `kd_v6.jsonl`) | ~5 phút |
| `clean_v6/` | 4,1 GB | bước "tách train/dev" trong `scripts/overnight_v6.sh` | ~4 phút |
| `bin_v6/` | 1,8 GB | `python scripts/binarize_ja2vi.py --clean <clean_v6> --out <bin_v6>` | ~4 phút |
| `curve_v6/` 8 mốc neo | 4,6 GB | `modal volume get vija-100m-kd-vol checkpoints_v6/stepN.pt` | ~1 phút/mốc |
| 66 mốc `stepN.pt` | 38 GB | Còn NGUYÊN trên Modal volume `vija-100m-kd-vol/checkpoints_v6` | — |
| `data/full_11.88m_ja_clean.txt` | 979 MB | Lọc lại từ CC-100 | — |

**`bin_v6` cũng đã có sẵn trên volume của profile `tritue12`** (upload 2026-07-28 để dự phòng
hết credit) — xem `[[doi-tai-khoan-modal-khi-het-credit]]`.

### Kết quả đo vòng 6 — ĐÃ COMMIT vào git (nhẹ, 19,7 MB)

| File | Nội dung |
|---|---|
| `eval/dev_curve_v6.json` | Đường cong dev loss 10 điểm — **held-out đầu tiên của dự án** |
| `eval/judge_v6/RESULT.json` + `judges/` | Điểm judge mù từng câu, 3 hệ v5/v6/google cùng phiên |
| `eval/bench_v6.jsonl` | 200 câu bench TED do `v6_avg` dịch |
| `eval/bench_held_v6.jsonl` | 120 câu Quốc hội held-out (bench DUY NHẤT đo được câu dài) |
| `eval/dom6_*.jsonl` | 3 probe miền (cntt, y tế, nông lâm ngư) |
| `eval/probe_v6.jsonl` | 80 phép thử ngữ pháp — v6 đạt 76/80 (v5: 71) |
| `eval/skill_density_v6new.json` | Mật độ kỹ năng của 2,17M câu mới |
| `logs/KETQUA_V6.txt` | Báo cáo gộp |
| `logs/train_v4.log`, `train_v5.log` | Đường cong loss các vòng trước (giữ có chủ ý) |
| `ISSUES.md` | Issue #1 câu dài — MỞ |

⚠️ `logs/kd_*.log` bị gitignore: hàng chục MB thông báo lỗi quota lặp lại, không có giá trị.
