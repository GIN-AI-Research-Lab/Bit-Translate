# -*- coding: utf-8 -*-
"""
Mổ HÌNH HỌC trọng số: "không gian hướng" của mỗi ma trận layer.
- Spectrum SVD: năng lượng trải theo bao nhiêu hướng? (stable rank, rank@90/99% năng lượng)
- Hướng chi phối: có channel/chiều outlier không? (max/median norm cột)
- Ternary phá hướng nào: sai số ternary rơi vào các hướng CHÍNH (nguy hiểm) hay đều (incoherent)?
- Ternary có giữ được không gian con chính không? (góc giữa top-32 hướng của W và ternary(W))

So v7a (ternary-trained master) vs Qwen3-0.6B (FP) — model được train ternary có hình học khác?
"""
import glob
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(5)

V7A = r"D:\Bit-Translate-data\ckpt_v7a_ng\last.pt"
QWEN_ST = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors")[0]
K = 32  # số hướng chính để xét subspace


def ternary(W):
    s = W.abs().mean()
    return torch.round(W / s).clamp(-1, 1) * s


def geometry(name, W):
    W = W.float()
    m, n = W.shape
    fro2 = (W ** 2).sum().item()
    sv = torch.linalg.svdvals(W)                      # singular values giảm dần
    s2 = (sv ** 2)
    smax2 = s2[0].item()
    stable_rank = fro2 / smax2                        # ||W||_F^2 / ||W||_2^2  (1..min(m,n))
    cum = torch.cumsum(s2, 0) / s2.sum()
    r90 = int((cum < 0.90).sum().item()) + 1
    r99 = int((cum < 0.99).sum().item()) + 1
    dim = min(m, n)
    # hướng chi phối: chuẩn cột (input directions)
    cnorm = W.norm(dim=0)
    col_outlier = (cnorm.max() / cnorm.median()).item()

    # ternary: sai số rơi vào hướng chính hay đều?
    T = ternary(W)
    R = W - T
    err = (R.norm() / W.norm()).item()
    # top-K hướng phải (right singular vectors) của W
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Vk = Vh[:K].t()                                   # [n, K] không gian con input chính
    e_top = (R @ Vk).pow(2).sum().item() / (R.pow(2).sum().item() + 1e-9)  # % sai số nằm trong top-K
    frac_expected = K / dim                           # nếu sai số ĐỀU mọi hướng thì ~ K/dim
    concentration = e_top / frac_expected             # >1 = sai số dồn vào hướng chính (xấu)

    # ternary giữ không gian con chính? góc giữa top-K của W và của T
    Ut, St, Vht = torch.linalg.svd(T, full_matrices=False)
    align = torch.linalg.svdvals(Vk.t() @ Vht[:K].t()).mean().item()  # cos góc chính TB (1=trùng)

    return dict(shape=f"{m}x{n}", stable_rank=stable_rank, dim=dim, r90=r90, r99=r99,
                col_outlier=col_outlier, tern_err=err, err_top=e_top,
                concentration=concentration, subspace_align=align)


def show(rows):
    print(f"{'ma trận':26s}{'shape':>10s}{'stblRank':>9s}{'r90%':>6s}{'r99%':>6s}"
          f"{'colOut':>8s}{'ternErr':>8s}{'errTopK':>8s}{'concen':>8s}{'align':>7s}")
    print("-" * 108)
    for name, g in rows:
        print(f"{name:26s}{g['shape']:>10s}{g['stable_rank']:9.1f}{g['r90']:6d}{g['r99']:6d}"
              f"{g['col_outlier']:8.2f}{g['tern_err']*100:7.1f}%{g['err_top']*100:7.1f}%"
              f"{g['concentration']:8.2f}{g['subspace_align']:7.3f}")


def main():
    print("=" * 108)
    print("HÌNH HỌC TRỌNG SỐ — không gian hướng của layer weight")
    print("=" * 108)
    print("Đọc cột: stblRank=số hướng 'hiệu dụng' (cao=trải đều, khó nén low-rank)")
    print("  r90/r99%=số hướng giữ 90/99% năng lượng | colOut=chi phối của cột mạnh nhất (max/median)")
    print("  ternErr=sai số ternary | errTopK=%sai số nằm trong 32 hướng chính")
    print("  concen=errTopK / (32/dim): >1 sai số DỒN vào hướng chính (xấu), ~1 ĐỀU (incoherent)")
    print("  align=cos góc giữa 32 hướng chính của W vs ternary(W) (1=ternary giữ nguyên hướng)\n")

    sd = torch.load(V7A, map_location="cpu", weights_only=False)["model"]
    rows_v7a = [
        ("v7a L0 attn.wq", geometry("wq", sd["blocks.0.attn.wq.weight"])),
        ("v7a L0 attn.wo", geometry("wo", sd["blocks.0.attn.wo.weight"])),
        ("v7a L9 ffn.gate", geometry("gate", sd["blocks.9.ffn.gate.weight"])),
        ("v7a L9 ffn.down", geometry("down", sd["blocks.9.ffn.down.weight"])),
    ]
    print(">>> MODEL v7a 152M (master FP32, đã train ternary-QAT)")
    show(rows_v7a)

    from safetensors import safe_open
    with safe_open(QWEN_ST, framework="pt") as f:
        rows_qwen = [
            ("qwen L13 q_proj", geometry("q", f.get_tensor("model.layers.13.self_attn.q_proj.weight"))),
            ("qwen L13 o_proj", geometry("o", f.get_tensor("model.layers.13.self_attn.o_proj.weight"))),
            ("qwen L13 gate_proj", geometry("g", f.get_tensor("model.layers.13.mlp.gate_proj.weight"))),
            ("qwen L13 down_proj", geometry("d", f.get_tensor("model.layers.13.mlp.down_proj.weight"))),
        ]
    print("\n>>> MODEL Qwen3-0.6B (FP bf16, CHƯA train ternary)")
    show(rows_qwen)

    print("\n" + "=" * 108)
    print("Ý NGHĨA CHO CONVERT: nếu stblRank THẤP (vài hướng) => tách low-rank FP + ternary residual bõ công.")
    print("Nếu concen >> 1 => ternary phá đúng hướng quan trọng => cần incoherence (Hadamard) xoay trước.")
    print("align thấp => ternary xoay lệch không gian con => chỉ QAT (train) mới kéo lại được.")


if __name__ == "__main__":
    main()
