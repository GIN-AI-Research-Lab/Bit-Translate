#!/usr/bin/env bash
# bootstrap.sh — node thuê MỚI (trắng) -> train 292M chạy, MỘT lệnh, không sửa tay.
#
# Yêu cầu DUY NHẤT khi thuê node: chọn template/image có PyTorch + CUDA.
# (torch ~5GB không đóng gói lên release được — template của nhà cung cấp lo việc này.)
#
# Dán 4 dòng này trên node mới:
#   export GH_TOKEN=<token>     # lấy ở máy đang auth: gh auth token
#   curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz | tar xz -C /tmp
#   /tmp/gh_2.63.0_linux_amd64/bin/gh repo clone trituenguyen97/Bit-Translate bt && cd bt
#   bash cloud/bootstrap.sh
#
# Script tự làm (idempotent — chạy lại không hại):
#   gh vĩnh viễn + auth | gcc (Triton) + zstd | pip sentencepiece/gguf |
#   tải premix bin (452MB) | resume từ autosave-scale300m nếu có (không thì from-scratch) |
#   launch train + backup daemon (30'/lần) + watcher đóng gói.
# Env tuỳ chọn: M300_MT/M300_GA (mặc định 4096/32 — ĐÃ KIỂM CHỨNG trên 12GB),
#   WATCH_TAG (mặc định scale300m-step25000), M300_COMPILE=0 (node Blackwell/RTX50).
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=/opt/conda/bin:/usr/local/bin:$PATH
REPO=trituenguyen97/Bit-Translate
say(){ echo -e "\n==== $* ===="; }

# gh release download không có progress bar -> tự hiển thị % (so size file đang tải
# với tổng size asset khớp pattern trên release). Dùng: dl_release <tag> <pattern> [dir]
dl_release(){
  local tag=$1 pat=$2 dir=${3:-.} total=0 pid cur f name size
  # tổng size các asset khớp pattern (glob-match bằng case, không dùng regex jq)
  while IFS=$'\t' read -r name size; do
    case "$name" in $pat) total=$((total + size));; esac
  done < <(gh release view "$tag" --repo "$REPO" --json assets \
           --jq '.assets[] | "\(.name)\t\(.size)"' 2>/dev/null || true)
  # bịt spinner của gh (giữ log để in nếu fail); poll size file để in %
  gh release download "$tag" --repo "$REPO" --pattern "$pat" --dir "$dir" --clobber \
    >"$dir/.dl.log" 2>&1 &
  pid=$!
  if [ "$total" -gt 0 ]; then
    while kill -0 "$pid" 2>/dev/null; do
      cur=0
      for f in "$dir"/$pat; do
        [ -f "$f" ] && cur=$((cur + $(stat -c%s "$f" 2>/dev/null || echo 0)))
      done
      printf '\r   %s: %3d%%  (%d/%d MB)   ' "$pat" $((cur * 100 / total)) $((cur / 1048576)) $((total / 1048576))
      sleep 2
    done
    printf '\r   %s: hoàn tất  (%d MB)          \n' "$pat" $((total / 1048576))
  fi
  if ! wait "$pid"; then echo; cat "$dir/.dl.log" >&2; rm -f "$dir/.dl.log"; return 1; fi
  rm -f "$dir/.dl.log"
}

# ---- 1) gh + auth ----
if ! command -v gh >/dev/null 2>&1; then
  say "Cài gh"
  VER=2.63.0
  curl -fL --progress-bar "https://github.com/cli/cli/releases/download/v${VER}/gh_${VER}_linux_amd64.tar.gz" | tar xz -C /tmp
  cp "/tmp/gh_${VER}_linux_amd64/bin/gh" /usr/local/bin/gh
fi
if ! gh auth status >/dev/null 2>&1; then
  [ -n "${GH_TOKEN:-}" ] || { echo "!! chưa auth: export GH_TOKEN=<token> rồi chạy lại (lấy token: 'gh auth token' ở máy local)"; exit 1; }
  # lưu token vĩnh viễn (gh cấm --with-token khi GH_TOKEN đang set -> tạm bỏ env)
  echo "$GH_TOKEN" | GH_TOKEN= GITHUB_TOKEN= gh auth login --with-token
fi
gh auth setup-git >/dev/null 2>&1 || true

# ---- 2) toolchain ----
command -v gcc  >/dev/null 2>&1 || { say "Cài gcc (Triton cần C compiler)"; apt-get update -qq && apt-get install -y gcc; }
command -v zstd >/dev/null 2>&1 || { say "Cài zstd"; apt-get install -y zstd 2>/dev/null || conda install -y zstd; }

# ---- 3) python: torch bắt buộc có sẵn; lib nhẹ thì tự cài ----
say "Kiểm python/torch/GPU"
python3 - <<'EOF'
import torch, numpy
assert torch.cuda.is_available(), "torch không thấy GPU — thuê node phải chọn template PyTorch/CUDA!"
print("torch", torch.__version__, "| numpy", numpy.__version__, "| GPU:", torch.cuda.get_device_name(0),
      "| VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
mm = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
if not ((2, 4) <= mm <= (2, 6)):
    print(f"⚠ torch {torch.__version__} NGOÀI dải đã kiểm chứng 2.4-2.6 (cloud/requirements-node.txt)"
          f" — vẫn chạy tiếp, nhưng nếu compile lỗi lạ thì nghi version trước, thử M300_COMPILE=0.")
EOF
python3 -c "import sentencepiece" 2>/dev/null || pip install sentencepiece
python3 -c "import gguf"          2>/dev/null || pip install gguf

# ---- 4) data + checkpoint ----
mkdir -p checkpoints dist
if [ ! -f data/bin/train.tokens.u16 ]; then
  say "Tải premix bin (452MB, base + vòng1 + vòng2 đã mix/LaBSE)"
  dl_release train-assets-vong2 'bin_mix_vong2.tar.zst'
  tar -I zstd -xf bin_mix_vong2.tar.zst && touch data/bin/.premixed_vong2 && rm -f bin_mix_vong2.tar.zst
fi
if [ ! -f checkpoints/last.pt ]; then
  if gh release view "${BK_TAG:-autosave-scale300m}" --repo "$REPO" --json assets \
       --jq '.assets[].name' 2>/dev/null | grep -q '\.step$'; then
    say "Thấy autosave -> khôi phục checkpoint mới nhất"
    bash cloud/restore_300m.sh
  else
    say "Không có autosave -> from-scratch từ step 0"
  fi
fi

# ---- 5) launch cả 3: train + backup + watcher ----
export M300_MT=${M300_MT:-4096} M300_GA=${M300_GA:-32}
say "Launch train (MT=$M300_MT GA=$M300_GA, compile=${M300_COMPILE:-1}) + backup 30' + watcher"
nohup bash cloud/run_300m.sh > checkpoints/scale300m.log 2>&1 &
sleep 3
nohup bash cloud/backup_300m.sh > checkpoints/backup.log 2>&1 &
WATCH_TAG=${WATCH_TAG:-scale300m-step25000} nohup bash cloud/watch_vong1.sh > checkpoints/watch.log 2>&1 &
sleep 2
say "XONG. Theo dõi: bash watch.sh | tail -f checkpoints/train.log (compile warmup ~10-20' mới có step)"
