# -*- coding: utf-8 -*-
"""Xuất tensor thật (gen4-0.6B) sang format TQ33-block64 (12 byte/64w) + vector x
+ y tham chiếu cho microbench C. Block64 = [11 byte codes (8 code x 11bit LSB-first)
+ 1 byte scale f8]. Code 11-bit = cặp nhóm-4 (id0*33+id1), mỗi nhóm ≤2 nonzero."""
import struct
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
KEY = "model.layers.27.mlp.down_proj.weight"     # [1024, 3072]
OUT = r"D:\Bit-Translate-data\tq33_bench"
G_SCALE = 64


def build_patterns():
    pats = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    v = (a, b, c, d)
                    if sum(1 for x in v if x != 0) <= 2:
                        pats.append(v)
    return pats


def f8_e4m3_from_float(x):
    """Scale bake của lab đã NẰM trên lưới f8-e4m3 (đi qua to_f8 lúc train) —
    ở đây lưu chỉ số xấp xỉ: dùng bảng 256 giá trị e4m3 dựng từ torch."""
    return x  # scale giữ fp32 trong file .scales (đơn giản cho microbench; C đọc fp32)


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    W = state[KEY].float()
    R, C = W.shape
    assert C % G_SCALE == 0
    print(f"{KEY}: {R}x{C}")
    PATS = build_patterns()
    PAT_ID = {}
    for i, p in enumerate(PATS):
        PAT_ID[p] = i

    # tách t, s
    Wv = W.view(R, C // G_SCALE, G_SCALE)
    s = Wv.abs().amax(dim=2)                                    # [R, C/64]
    t = torch.where(s.unsqueeze(2) > 0,
                    torch.round(Wv / s.unsqueeze(2).clamp(min=1e-12)),
                    torch.zeros_like(Wv)).clamp(-1, 1).to(torch.int8)
    # encode block64: 16 nhóm-4 -> 8 cặp -> 8 code 11bit -> 11 byte + scale
    nb = C // G_SCALE
    codes = np.zeros((R, nb, 11), dtype=np.uint8)
    t_np = t.numpy()
    for r in range(R):
        for b in range(nb):
            grp = t_np[r, b].reshape(16, 4)
            bits = 0
            acc = 0
            out = []
            for pair in range(8):
                i0 = PAT_ID[tuple(int(v) for v in grp[2 * pair])]
                i1 = PAT_ID[tuple(int(v) for v in grp[2 * pair + 1])]
                code = i0 * 33 + i1
                acc |= code << bits
                bits += 11
                while bits >= 8:
                    out.append(acc & 0xFF)
                    acc >>= 8
                    bits -= 8
            if bits:
                out.append(acc & 0xFF)
            assert len(out) == 11
            codes[r, b] = out
    codes.tofile(os.path.join(OUT, "codes.bin"))
    s.numpy().astype(np.float32).tofile(os.path.join(OUT, "scales.bin"))
    x = torch.randn(C)
    y_ref = (t.float() * s.unsqueeze(2)).view(R, C) @ x
    x.numpy().astype(np.float32).tofile(os.path.join(OUT, "x.bin"))
    y_ref.numpy().astype(np.float32).tofile(os.path.join(OUT, "y_ref.bin"))
    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"{R} {C} {G_SCALE}\n")
    packed = R * nb * 12
    print(f"codes+scales: {packed/1e6:.2f} MB  ({packed*8/(R*C):.3f} bpw)"
          f" | fp32 gốc: {R*C*4/1e6:.1f} MB")
    print("Xuất xong ->", OUT)


if __name__ == "__main__":
    main()
