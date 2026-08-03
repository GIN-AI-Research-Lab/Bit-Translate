# -*- coding: utf-8 -*-
"""
Exp Y (H1-probe) — Gauge-canonical folding: transformer sau train có dư thừa XUYÊN-LAYER
ẩn dưới gauge (perm + diagonal) không?

Phép thử giết-nhanh: canonical-hóa 28 bộ MLP của Qwen3-0.6B dưới đúng bộ gauge exact
(diagonal up↔down + hoán vị kênh intermediate), đo khoảng cách cặp, so với NULL
(khoảng cách sau khi phá alignment bằng shuffle). Nếu real ≈ null -> H1 chết, $0, một buổi.

Kỳ vọng ghi trước: 70% real ≈ null (chết); 30% có cụm cặp < 0.8×null (đáng gấp thử).
"""
import glob
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)

MODEL_GLOB = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"


def canonicalize(gate, up, down):
    """Chuẩn hóa gauge: (1) diagonal — cột down về norm 1, hấp thụ vào up;
    (2) hoán vị — sort kênh intermediate theo khóa bất biến (norm up-row sau chuẩn hóa)."""
    g, u, d = gate.float().clone(), up.float().clone(), down.float().clone()
    cn = d.norm(dim=0).clamp(min=1e-12)          # [inter]
    d = d / cn[None, :]
    u = u * cn[:, None]
    key = u.norm(dim=1)                          # bất biến sau diagonal
    idx = key.argsort(descending=True)
    return g[idx, :], u[idx, :], d[:, idx]


def rel_dist(A, B):
    return (A - B).norm().item() / max((A.norm() * B.norm()).sqrt().item(), 1e-12)


def main():
    from transformers import AutoModelForCausalLM
    mdir = glob.glob(MODEL_GLOB)[0]
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32)
    blocks = []
    for blk in model.model.layers:
        blocks.append(canonicalize(blk.mlp.gate_proj.weight.data,
                                   blk.mlp.up_proj.weight.data,
                                   blk.mlp.down_proj.weight.data))
    n = len(blocks)
    print(f"{n} block đã canonical-hóa. Đo khoảng cách cặp (gate/up/down trung bình)...")

    def block_dist(a, b, shuffle_null=False):
        ga, ua, da = blocks[a]
        gb, ub, db = blocks[b]
        if shuffle_null:
            perm = torch.randperm(ub.shape[0])
            gb, ub, db = gb[perm, :], ub[perm, :], db[:, perm]
        return (rel_dist(ga, gb) + rel_dist(ua, ub) + rel_dist(da, db)) / 3

    real = []
    for a in range(n):
        for b in range(a + 1, n):
            real.append((block_dist(a, b), a, b))
    real.sort()
    null = [block_dist(a, b, shuffle_null=True)
            for a, b in [(i, (i + 7) % n) for i in range(n)]]
    null_mean = sum(null) / len(null)
    real_mean = sum(x[0] for x in real) / len(real)
    print(f"NULL (shuffle) mean : {null_mean:.4f}")
    print(f"REAL canonical mean : {real_mean:.4f}")
    print("10 cặp GẦN nhất:")
    for dst, a, b in real[:10]:
        print(f"  block {a:2d} ~ block {b:2d}: {dst:.4f}  ({dst/null_mean*100:.1f}% của null)")
    verdict = "CÓ TÍN HIỆU — đáng gấp thử + đo PPL" if real[0][0] < 0.8 * null_mean \
        else "REAL ≈ NULL — H1 CHẾT (đúng nhánh 70% kỳ vọng ghi trước)"
    print(f"PHÁN QUYẾT probe: {verdict}")


if __name__ == "__main__":
    main()
