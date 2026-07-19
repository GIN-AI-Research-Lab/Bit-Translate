"""Test nhanh Modal: GPU gì, có bf16 không, matmul chạy được không.
Chạy:  .venv/bin/python -m modal run cloud/modal_gpu_test.py
(Modal tự bật container GPU, chạy hàm, in kết quả, rồi tắt — chỉ tính tiền ~10-20s.)
Đổi gpu="A10G" thành "L4"/"A100"/"H100" để thử GPU khác.
"""
import modal

app = modal.App("vija-gpu-test")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(gpu="L40S", image=image, timeout=300)
def check():
    import torch
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print("VRAM:", round(p.total_memory / 1e9, 1), "GB")
    print("bf16 hỗ trợ:", torch.cuda.is_bf16_supported())
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    print("bf16 matmul OK:", tuple(y.shape), "torch", torch.__version__)


@app.local_entrypoint()
def main():
    check.remote()
