#!/usr/bin/env bash
# Chạy MỘT LẦN trên máy cloud (A4000, template PyTorch) sau khi giải nén.
# Template PyTorch đã có sẵn torch + CUDA -> chỉ cài thêm vài thứ nhỏ.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
command -v sudo >/dev/null 2>&1 && SUDO="sudo"

# build-essential (gcc) + linux-libc-dev: torch.compile/Triton cần trình biên dịch
# C + header để build kernel. Image conda thường THIẾU gcc -> phải cài.
$SUDO apt-get update -y || true
$SUDO apt-get install -y build-essential linux-libc-dev || echo "WARN: không cài được build-essential/linux-libc-dev (torch.compile sẽ lỗi; bỏ --compile trong run_cloud.sh thì vẫn train được)"

# deps Python (torch đã có sẵn trong template PyTorch)
pip install -U sentencepiece numpy sacrebleu

echo "=== kiểm tra môi trường ==="
python3 -c "import torch,sentencepiece,numpy,sacrebleu; print('torch',torch.__version__,'| CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "=== setup xong. Tiếp: bash cloud/run_cloud.sh ==="
