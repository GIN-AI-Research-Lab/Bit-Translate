# -*- coding: utf-8 -*-
"""
Exp C — Hadamard incoherence: LÀM ĐÚNG vs LÀM GIẢ (trên trọng số Qwen3-0.6B thật).

Bài học: đây chính là kỹ thuật QuIP#/QuaRot/SpinQuant/QTIP dùng để đẩy ngưỡng bit
xuống thấp. Antigravity đặt tên "Hadamard incoherent" nhưng ruột chỉ lật dấu ngẫu nhiên
→ với quantizer theo |W| thì lật rồi lật lại = KHÔNG LÀM GÌ (đã bị agent chứng minh diff=0).

Ở đây so 3 điều kiện, cùng scheme ternary-g32 và int2-g32, đo sai số Frobenius tương đối
(vì phép xoay trực giao Q bảo toàn ||·||_F nên ||W'-Wq'||/||W'|| chính là sai số OUTPUT thật):
  (0) baseline  : quantize thẳng W
  (F) fake flip : lật dấu ngẫu nhiên theo cột rồi quantize rồi lật lại (kiểu Antigravity)
  (H) real Hadamard incoherence: W' = W @ Q,  Q = random-Hadamard trực giao;
      quantize W' rồi map lại bằng Q^T. Runtime phải xoay activation x -> Q^T x.

Kỳ vọng đúng lý thuyết: (F) ≈ (0) chính xác (diff≈0, vô dụng); (H) < (0) thật sự vì Q làm
phân bố phần tử phẳng hơn, dập outlier → dễ quantize hơn. Nhưng mức giảm KHÔNG đủ đưa
ternary về vùng dùng được → xác nhận: incoherence giúp ~1 mức bit, không cứu nổi 1.58-bit.
"""
import glob
import os
import sys

import numpy as np
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")

SNAP = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors")[0]

# Ma trận đại diện: q/o (attention), gate/down (FFN), ở layer đầu/giữa/cuối
TARGETS = [
    ("model.layers.0.mlp.down_proj.weight", "L0 down_proj"),
    ("model.layers.13.self_attn.q_proj.weight", "L13 q_proj"),
    ("model.layers.13.self_attn.o_proj.weight", "L13 o_proj"),
    ("model.layers.13.mlp.gate_proj.weight", "L13 gate_proj"),
    ("model.layers.13.mlp.down_proj.weight", "L13 down_proj"),
    ("model.layers.27.mlp.down_proj.weight", "L27 down_proj"),
]


def sylvester_hadamard(n):
    """Hadamard chuẩn hóa n×n, n phải là lũy thừa 2 (trực giao: H@H.T = I)."""
    assert n & (n - 1) == 0, f"n={n} không phải lũy thừa 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def random_orthogonal_for_dim(n, seed):
    """Random-Hadamard trực giao cho chiều n. n=2^k dùng Hadamard;
    n=1024*m dùng block-diag m khối Hadamard-1024. Kèm dấu ngẫu nhiên hai bên (kiểu QuIP)."""
    rng = np.random.RandomState(seed)
    if n & (n - 1) == 0:
        H = sylvester_hadamard(n)
    elif n % 1024 == 0:
        h = sylvester_hadamard(1024)
        m = n // 1024
        H = np.zeros((n, n), dtype=np.float64)
        for i in range(m):
            H[i * 1024:(i + 1) * 1024, i * 1024:(i + 1) * 1024] = h
    else:
        raise ValueError(f"chiều {n} chưa hỗ trợ")
    s_left = rng.choice([-1.0, 1.0], size=n)
    s_right = rng.choice([-1.0, 1.0], size=n)
    return (s_left[:, None] * H) * s_right[None, :]


def group_view(w):
    R, C = w.shape
    pad = (32 - C % 32) % 32
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    return w.reshape(w.shape[0], w.shape[1] // 32, 32), C


def ternary_g32(w):
    """3 mức Lloyd-Gaussian per group-32 (ngưỡng 0.612σ; giá trị = mean|w| phần đuôi)."""
    wg, C = group_view(w)
    sigma = np.maximum(wg.std(axis=2, keepdims=True), 1e-8)
    mask = np.abs(wg) > 0.6120 * sigma
    cnt = np.maximum(mask.sum(axis=2, keepdims=True), 1)
    v = (np.abs(wg) * mask).sum(axis=2, keepdims=True) / cnt
    wq = (np.sign(wg) * v * mask).reshape(w.shape[0], -1)[:, :C]
    return wq


def int2_g32(w):
    """int2 codebook đối xứng ±0.4528σ, ±1.5104σ (Lloyd-Gaussian 4 mức) per group-32."""
    wg, C = group_view(w)
    sigma = np.maximum(wg.std(axis=2, keepdims=True), 1e-8)
    levels = np.array([-1.5104, -0.4528, 0.4528, 1.5104])
    z = wg / sigma
    idx = np.abs(z[..., None] - levels).argmin(axis=-1)
    wq = (levels[idx] * sigma).reshape(w.shape[0], -1)[:, :C]
    return wq


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


SCHEMES = {"ternary-g32": ternary_g32, "int2-g32": int2_g32}


def main():
    print(f"Trọng số thật: {SNAP}\n")
    print(f"{'Ma trận':16s} {'scheme':12s} {'(0)base':>9s} {'(F)fake':>9s} {'(H)Hadam':>9s} "
          f"{'H giảm':>8s} {'F=base?':>9s}")
    print("-" * 82)
    agg = {s: {"base": [], "had": []} for s in SCHEMES}

    with safe_open(SNAP, framework="pt") as f:
        for key, label in TARGETS:
            W = f.get_tensor(key).float().numpy().astype(np.float64)
            _, in_dim = W.shape
            Q = random_orthogonal_for_dim(in_dim, seed=hash(key) & 0xFFFF)
            Wr = W @ Q  # xoay incoherence (phải chiều in)

            # fake sign-flip theo cột (tái hiện Antigravity)
            rng = np.random.RandomState(1)
            s = rng.choice([-1.0, 1.0], size=in_dim)
            Wf = W * s[None, :]

            for sname, qfn in SCHEMES.items():
                e_base = rel_err(W, qfn(W))
                # fake: quantize W*s rồi lật lại; đo output error trên W
                W_fake_hat = qfn(Wf) * s[None, :]
                e_fake = rel_err(W, W_fake_hat)
                # real Hadamard: quantize Wr; sai số output = ||Wr - q(Wr)|| / ||Wr|| (Q trực giao)
                e_had = rel_err(Wr, qfn(Wr))
                drop = (e_base - e_had) / e_base * 100
                same = abs(e_fake - e_base) < 1e-9
                print(f"{label:16s} {sname:12s} {e_base*100:8.1f}% {e_fake*100:8.1f}% "
                      f"{e_had*100:8.1f}% {drop:7.1f}% {'YES(0)' if same else 'khác':>9s}")
                agg[sname]["base"].append(e_base)
                agg[sname]["had"].append(e_had)
            print()

    print("=" * 82)
    for sname in SCHEMES:
        b = np.mean(agg[sname]["base"]) * 100
        h = np.mean(agg[sname]["had"]) * 100
        print(f"  TB {sname:12s}: baseline {b:.1f}%  ->  +Hadamard {h:.1f}%  "
              f"(giảm {(b-h)/b*100:.1f}% tương đối)")
    print("=" * 82)
    print("Đọc kết quả: cột (F)fake trùng khít (0)base = trò 'Hadamard' của Antigravity vô dụng.")
    print("Cột (H) thấp hơn (0) = incoherence THẬT có tác dụng, mạnh nhất ở down_proj/o_proj")
    print("(nơi nhiều outlier). Nhưng ternary vẫn ~35-40% >> 10% của int4 => chưa cứu được 1.58-bit.")


if __name__ == "__main__":
    main()
