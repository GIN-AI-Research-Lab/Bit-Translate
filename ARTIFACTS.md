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

