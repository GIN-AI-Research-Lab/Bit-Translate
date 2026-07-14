# ARTIFACTS — checkpoint & model đang ở đâu

> Bản đồ vị trí mọi checkpoint/model/data. **Cập nhật 2026-07-14.**
> ⚠️ Các file nặng dưới đây **KHÔNG nằm trong git** (`.gitignore` loại `checkpoints/ cloud_backup/ dist/ data/`).
> Ổ đĩa local `e:\Bit-Translate\` là **bản duy nhất** (trừ bản trên GitHub Release). Nên backup thêm 1 chỗ.

## Checkpoint để TRAIN TIẾP (Bước 5)

| File | Vị trí | Size | Nội dung | Dùng để |
|---|---|---|---|---|
| **`last_final_14000.pt`** | `cloud_backup/` | 1.3GB | fp32 model + **optimizer**, step 14000 | **Resume train đầy đủ** (đã verify `torch.load` OK) ⭐ |
| `last_snapshot.pt.zst` | `cloud_backup/` | 1.2GB | fp32 model + optimizer, step **5900** (nén zstd) | Milestone cũ, resume được nếu cần lùi |
| `step4000.pt` | `checkpoints/` | 438MB | fp32 model-only, step 4000 | Milestone sớm (không có optimizer) |

## Checkpoint model-only (deploy / convert)

| File | Vị trí | Size | Nội dung |
|---|---|---|---|
| `ckpt14000.pt` | `cloud_backup/` | 268MB | fp16 model-only, step 14000 (bản final) |
| `ckpt7700.pt` | `cloud_backup/` | 268MB | fp16 model-only, step 7700 |

## Model deploy i2_s (GGUF, 68MB, chạy bitnet.cpp CPU)

| File | Vị trí | Ghi chú |
|---|---|---|
| **`vija14000_i2s.gguf`** | WSL `~/vija-test/` | Bản FINAL đang chạy demo (localhost:8765) |
| (bản copy) | `cloud_backup/vija-1p58-step14000-i2s.gguf` | Dùng làm GitHub Release asset |
| `vija{4000,5900,7700}_i2s.gguf` | WSL `~/vija-test/` | Các milestone để so sánh |

## Data train

| File | Vị trí | Size | Nội dung |
|---|---|---|---|
| `vija_data.tar.zst` | `cloud_backup/` | 319MB | `data/bin/` (train/dev tokens u16 + index) + `data/clean/flores` |

Giải nén để train tiếp: `tar -I zstd -xf cloud_backup/vija_data.tar.zst` → ra `data/bin/`.

## GitHub Release

- Repo: `github.com/trituenguyen97/Bit-Translate` (private) — code + docs; binary gitignored.
- Release **v1.0-step14000**: https://github.com/trituenguyen97/Bit-Translate/releases/tag/v1.0-step14000
- Assets: `vija-1p58-step14000-i2s.gguf` (68MB) + `spm_vija_32k.model` (tokenizer).
- Tải về: `gh release download v1.0-step14000 --repo trituenguyen97/Bit-Translate` (hoặc từ trang release).

## Cloud node (ckey.vn)

- `root@n2.ckey.vn -p 2614` (RTX 4070 Ti) — **đã/đang tắt sau khi kéo hết artifact về local**. Không còn gì cần trên node.

## Chất lượng theo step

Xem `eval/history.tsv` — chrF FLORES devtest (i2_s, n=100):
step 4000 → 5900 → 7700 → **14000**: vi→ja 17.3→21.3, ja→vi 35.8→**41.7**.
So Google Translate ja→vi: model đạt ~77% chrF (41.7 vs 54.0).
