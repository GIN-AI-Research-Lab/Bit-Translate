#!/usr/bin/env bash
# Chạy MỘT LẦN trên máy cloud (A4000, template PyTorch) sau khi giải nén.
# Template PyTorch đã có sẵn torch + CUDA -> chỉ cài thêm vài thứ nhỏ.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# build-essential (gcc) + linux-libc-dev: torch.compile/Triton cần trình biên dịch
# C + header để build kernel. Image conda thường THIẾU gcc -> phải cài.
# zstd: giải nén data base (vija_data.tar.zst) khi prep_vong1.sh tải release.
$SUDO apt-get update -y || true
$SUDO apt-get install -y build-essential linux-libc-dev zstd || echo "WARN: không cài được build-essential/linux-libc-dev/zstd (torch.compile lỗi -> bỏ --compile; thiếu zstd -> không giải nén được base)"

# deps Python (torch đã có sẵn trong template PyTorch).
# sentence-transformers: LaBSE lọc synthetic ở mix_and_binarize (Vòng 1). Cài SAU
# torch nên KHÔNG động vào bản torch+CUDA của template (nếu lỡ hỏng, verify dưới sẽ báo).
pip install -U sentencepiece numpy sacrebleu
pip install -U sentence-transformers || echo "WARN: thiếu sentence-transformers -> chạy prep_vong1.sh với SKIP_LABSE=1 (bỏ lọc LaBSE)"

# gh CLI: tải release private (data + checkpoint). Không có thì cài best-effort.
if ! command -v gh >/dev/null 2>&1; then
  $SUDO apt-get install -y gh 2>/dev/null || echo "WARN: chưa có gh. Cài: https://github.com/cli/cli/blob/trunk/docs/install_linux.md rồi 'gh auth login'"
fi

echo "=== kiểm tra môi trường ==="
python3 -c "import torch,sentencepiece,numpy,sacrebleu; print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "=== setup xong ==="
echo "  - Train Vòng 1 (BT+LaBSE+mix+train): gh auth login && bash cloud/prep_vong1.sh"
echo "  - Train base/tiếp thường:            bash cloud/run_cloud.sh"
