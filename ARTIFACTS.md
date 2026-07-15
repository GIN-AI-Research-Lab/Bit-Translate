# ARTIFACTS — checkpoint & model đang ở đâu

> Bản đồ vị trí mọi checkpoint/model/data. **Cập nhật 2026-07-14.**
> ⚠️ Các file nặng dưới đây **KHÔNG nằm trong git** (`.gitignore` loại `checkpoints/ cloud_backup/ dist/ data/`).
> **Backup:** data + checkpoint train + model deploy đã có **bản trên GitHub Release** (mục cuối) — mất ổ local vẫn tải lại được. Các milestone cũ (5900/7700/4000) chỉ có ở local.

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

## GitHub Release (repo `github.com/trituenguyen97/Bit-Translate`, private)

Code + docs ở git; binary ở 2 release dưới đây (cần `gh auth login` để tải vì repo private):

**Deploy — `v1.0-step14000`**: https://github.com/trituenguyen97/Bit-Translate/releases/tag/v1.0-step14000
- `vija-1p58-step14000-i2s.gguf` (68MB, i2_s CPU) + `spm_vija_32k.model` (tokenizer).

**Train — `train-assets-step14000`**: https://github.com/trituenguyen97/Bit-Translate/releases/tag/train-assets-step14000
- `vija_data.tar.zst` (319MB, `data/bin/` + flores) + `last_final_14000.pt` (1.3GB, fp32+optimizer).
- → **Đây là bản backup trên GitHub của data + checkpoint** (đồng bộ với `cloud_backup/`).

**Data Vòng 1 — `train-assets-vong1`**: https://github.com/trituenguyen97/Bit-Translate/releases/tag/train-assets-vong1
- `vong1-data.tar.gz` (18MB) = `data/synthetic/` + `data/glossary/` (248k cặp nhắm đích + glossary 18k term + test-set).
- Giải nén: `tar xzf vong1-data.tar.gz` → ra `data/synthetic/` + `data/glossary/`.

**Kết quả Vòng 1 — `vong1-step19000`** *(⏳ SẼ CÓ ~tối 15-07: watcher `cloud/watch_vong1.sh` trên node tự upload khi train chạm 19000)*:
- `last_final_19000.pt` (fp32+optimizer — resume Vòng 2) | `ckpt_19000_fp16.pt` (nhẹ, convert/deploy)
- `vija_19000_f16.gguf` (GGUF F16 — về local `llama-quantize ... I2_S 1` ra i2_s 68MB)
- `bt_vong1.tar.gz` (55.846 cặp back-translation — tốn GPU mới có) | `train_vong1_19000.log`

**Dựng node mới để train (một mạch):**
```bash
git clone git@github.com:trituenguyen97/Bit-Translate.git ~/Train-model-translate
cd ~/Train-model-translate && bash cloud/setup_cloud.sh && gh auth login
gh release download train-assets-step14000 --repo trituenguyen97/Bit-Translate
tar -I zstd -xf vija_data.tar.zst && mkdir -p checkpoints && mv last_final_14000.pt checkpoints/last.pt
# 12GB VRAM: sed -i 's/8192 --grad-accum 16/4096 --grad-accum 32/' cloud/run_cloud.sh
# Train TIẾP: tăng --max-steps trong cloud/run_cloud.sh (>14000)
nohup bash cloud/run_cloud.sh > checkpoints/cloud.log 2>&1 &
```

**Dựng node train VÒNG 1 (một lệnh — BT + LaBSE + mix + train):**
```bash
git clone git@github.com:trituenguyen97/Bit-Translate.git ~/Train-model-translate
cd ~/Train-model-translate && bash cloud/setup_cloud.sh && gh auth login
bash cloud/prep_vong1.sh          # tự tải 2 release, back-translate, lọc, trộn, train nền
tail -f checkpoints/train.log     # theo dõi (mục tiêu step 19000)
```
`prep_vong1.sh` cờ env: `SKIP_BT=1` (bỏ back-translation), `SKIP_LABSE=1` (bỏ lọc),
`NO_TRAIN=1` (chỉ chuẩn bị data), `BT_MAX=n` (chạy thử), `NEW_FRAC=0.3` (tỉ trọng data mới).
Xong 19000 → convert GGUF → eval `probe64`/chrF (gate ở `PLAN_BUOC5.md` §2.6).

## Cloud node (ckey.vn)

- `root@n2.ckey.vn -p 2614` (RTX 4070 Ti) — **đã/đang tắt sau khi kéo hết artifact về local**. Không còn gì cần trên node.

## Chất lượng theo step

Xem `eval/history.tsv` — chrF FLORES devtest (i2_s, n=100):
step 4000 → 5900 → 7700 → **14000**: vi→ja 17.3→21.3, ja→vi 35.8→**41.7**.
So Google Translate ja→vi: model đạt ~77% chrF (41.7 vs 54.0).
