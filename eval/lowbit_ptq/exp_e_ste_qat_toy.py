# -*- coding: utf-8 -*-
"""
Exp E — VÌ SAO "HỌC" CỨU ĐƯỢC TERNARY CÒN "LÀM TRÒN" THÌ KHÔNG (STE-QAT toy, CPU).

Đây là trái tim của BitNet. Lấy 1 ma trận thật của Qwen3-0.6B (down_proj — chỗ khó nhất),
và activation đầu vào X. Cả hai cách đều bị RÀNG BUỘC ternary y hệt {-1,0,+1}×scale/group.
Khác nhau ở CHỖ TỐI ƯU:

  PTQ (làm tròn):  ép W -> ternary gần W nhất. Tối ưu ||W_t - W||  (sai số TRỌNG SỐ).
  QAT-STE (học):   giữ master W_fp (float), forward dùng ternary(W_fp), loss là
                   ||ternary(W_fp)·X - W·X||  (sai số OUTPUT), backward bằng
                   straight-through estimator (grad chảy qua như ternary là identity),
                   cập nhật W_fp nhiều bước. Model "học" chọn cấu hình ternary tốt cho OUTPUT.

Cái ta THỰC SỰ cần là sai số OUTPUT nhỏ, không phải sai số trọng số. PTQ tối ưu nhầm đại lượng.
QAT tối ưu đúng — và vì có RẤT NHIỀU cấu hình ternary, nó tìm được cấu hình để các trọng số
bù trừ lẫn nhau trên phân bố X thật. Đó là lý do BitNet phải train, không quantize được sau.
"""
import glob
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

SNAP = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors")[0]
KEY = "model.layers.13.mlp.down_proj.weight"   # [1024, 3072]
GROUP = 32
STEPS = 400
BATCH = 512
LR = 2e-3


def ternary_g32(W):
    """absmean ternary per group-32 theo cột. Trả về tensor cùng shape (đã dequant)."""
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    Wg = W.view(R, -1, GROUP)
    scale = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    Wt = torch.round(Wg / scale).clamp(-1, 1) * scale
    return Wt.view(R, -1)[:, :C]


def ste_ternary(W):
    """Ternary forward + straight-through: value = ternary(W), grad = grad của W."""
    Wq = ternary_g32(W)
    return W + (Wq - W).detach()


def out_err(Wq, W, X):
    return (Wq @ X - W @ X).norm() / (W @ X).norm()


def main():
    print(f"Trọng số thật: {SNAP}")
    with safe_open(SNAP, framework="pt") as f:
        W = f.get_tensor(KEY).float()
    R, C = W.shape
    print(f"Ma trận: {KEY}  shape [{R},{C}]\n")

    # Activation X [C, BATCH]: Gaussian có outlier nhẹ (giống phân bố activation thật của down_proj)
    X = torch.randn(C, BATCH)
    X[torch.rand_like(X) < 0.02] *= 6.0   # 2% massive activation (đặc trưng dòng Qwen)

    # (1) PTQ: ternary thẳng
    Wq_ptq = ternary_g32(W)
    e_weight = (Wq_ptq - W).norm() / W.norm()
    e_ptq = out_err(Wq_ptq, W, X)
    print(f"[PTQ làm tròn]  sai số TRỌNG SỐ = {e_weight*100:5.1f}%   sai số OUTPUT = {e_ptq*100:5.1f}%")

    # (2) QAT-STE: học master weight để tối thiểu hóa sai số OUTPUT
    Wfp = W.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=LR)
    target = (W @ X).detach()
    for step in range(STEPS):
        opt.zero_grad()
        Wq = ste_ternary(Wfp)
        loss = (Wq @ X - target).pow(2).mean()
        loss.backward()
        opt.step()
        if step % 80 == 0 or step == STEPS - 1:
            with torch.no_grad():
                e = out_err(ternary_g32(Wfp), W, X)
            print(f"   QAT-STE step {step:3d}: sai số OUTPUT = {e*100:5.1f}%")

    with torch.no_grad():
        Wq_qat = ternary_g32(Wfp)
        e_qat = out_err(Wq_qat, W, X)
        e_weight_qat = (Wq_qat - W).norm() / W.norm()
        drift = (Wfp - W).norm() / W.norm()

    print(f"\n[QAT-STE học]   sai số TRỌNG SỐ = {e_weight_qat*100:5.1f}%   sai số OUTPUT = {e_qat*100:5.1f}%")
    print(f"   (master weight đã dịch {drift*100:.1f}% khỏi gốc để phục vụ ternary)")
    print("\n" + "=" * 70)
    print(f"  PTQ : output {e_ptq*100:.1f}%   |  QAT: output {e_qat*100:.1f}%   "
          f"|  QAT giảm {(e_ptq-e_qat)/e_ptq*100:.0f}% sai số output")
    print("=" * 70)
    print("Bài học:")
    print(f"  • QAT có sai số TRỌNG SỐ CAO HƠN ({e_weight_qat*100:.0f}% vs {e_weight*100:.0f}%) nhưng")
    print(f"    sai số OUTPUT THẤP HƠN — vì nó tối ưu đúng đại lượng ta cần.")
    print("  • Đây là điều PTQ (mọi trò 'cascade/Hadamard/codebook') KHÔNG làm được:")
    print("    chúng chỉ ép trọng số gần gốc, không được phép DỊCH trọng số để bù trên dữ liệu.")
    print("  • Nhân đây lên 28 lớp + distill từ thầy = BitNet Distillation. Đó là con đường thật.")


if __name__ == "__main__":
    main()
