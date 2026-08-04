# -*- coding: utf-8 -*-
"""Encode TOÀN BỘ linear (q/k/v/o attn + gate/up/down MoE, 48 layer x 128 expert = 18624
tensor) của ckpt 30B-A3B S1-native sang format TQ33 block64 (12B/64w = 1.5bpw), dùng
safe_ckpt_reader (né bug torch.load(mmap=True) trên file 61GB — xem safe_ckpt_reader.py).

Giữ nguyên bf16 (không TQ33): embed_tokens, lm_head, model.norm, input/post_attention_layernorm,
q_norm/k_norm, mlp.gate (router). Bỏ hẳn: mọi .bias (đã xác nhận =0, ckpt train --no-bias).

Output: D:/Bit-Translate-data/tq33_30b/
  manifest.json      -- tensor TQ33: {name: {shape:[R,C], nb: C//64*R, file: "linears/<idx>.tq33"}}
  extras.bin + extras_manifest.json -- tensor giữ nguyên bf16, đọc tuần tự theo manifest
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from safe_ckpt_reader import open_reader  # noqa: E402

CKPT = "D:/Bit-Translate-data/pack_local/pytorch_model.bin"
OUT = "D:/Bit-Translate-data/tq33_30b"
G_SCALE = 64


def build_pat_lut_b3():
    """Bang tra base-3(nhom-4, ((v+1)*[27,9,3,1]).sum()) -> pattern-id (0..32)."""
    pats = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    v = (a, b, c, d)
                    if sum(1 for x in v if x != 0) <= 2:
                        pats.append(v)
    assert len(pats) == 33
    lut_b3 = torch.full((81,), -1, dtype=torch.int32)
    for pid, p in enumerate(pats):
        b3 = sum((x + 1) * m for x, m in zip(p, (27, 9, 3, 1)))
        lut_b3[b3] = pid
    return lut_b3


LUT_B3 = build_pat_lut_b3()
POW3 = torch.tensor([27, 9, 3, 1], dtype=torch.int8)


def encode_linear(W):
    """W: [R,C] bf16/float, C%64==0. Tra (codes: np.uint8[R,C//64,11], scales: np.float32[R,C//64])."""
    W = W.float()
    R, C = W.shape
    assert C % G_SCALE == 0, f"C={C} khong chia het {G_SCALE}"
    nb = C // G_SCALE
    Wv = W.view(R, nb, G_SCALE)
    s = Wv.abs().amax(dim=2)                                     # [R, nb]
    t = torch.where(s.unsqueeze(2) > 0,
                    torch.round(Wv / s.unsqueeze(2).clamp(min=1e-12)),
                    torch.zeros_like(Wv)).clamp(-1, 1).to(torch.int8)
    recon = (t.float() * s.unsqueeze(2)).view(R, C)
    if not torch.equal(recon, W):
        raise ValueError("KHONG lossless — dung lai kiem tra, dung encode tiep")
    tg = t.view(R, nb, G_SCALE // 4, 4)                            # 16 nhom-4 / block64
    base3 = ((tg + 1) * POW3).sum(-1)                              # [R, nb, 16]
    gid = LUT_B3[base3.to(torch.int64)]
    if int((gid < 0).sum().item()) != 0:
        raise ValueError("pattern ngoai bang 33 — vi pham 2:4!")
    gid = gid.view(R, nb, 8, 2)                                    # 8 cap nhom / block64
    codeword = gid[..., 0] * 33 + gid[..., 1]                      # [R,nb,8] moi so 0..1088 (11bit)
    codeword_np = codeword.numpy().astype(np.uint16)
    # dong goi 8 x 11bit -> 11 byte, LSB-first, dung numpy vector hoa (khong loop Python)
    bits = np.zeros((R, nb, 88), dtype=np.uint8)                   # 88 bit = 11 byte
    for k in range(8):
        code_k = codeword_np[:, :, k]
        for b in range(11):
            bits[:, :, k * 11 + b] = (code_k >> b) & 1
    codes = np.packbits(bits, axis=-1, bitorder="little")         # [R, nb, 11]
    return codes, s.numpy().astype(np.float32)


def main():
    os.makedirs(os.path.join(OUT, "linears"), exist_ok=True)
    prefix, tensors_meta, extra, get, zf = open_reader(CKPT)
    print(f"tong {len(tensors_meta)} tensor trong ckpt; geo6={extra.get('meta.geo6')}")

    linear_names = []
    for k, m in tensors_meta.items():
        if k.endswith(".bias"):
            continue
        if any(s in k for s in (".weight",)) and (
            ("self_attn." in k and any(p in k for p in ("q_proj", "k_proj", "v_proj", "o_proj")))
            or ("mlp.experts." in k and any(p in k for p in ("gate_proj", "up_proj", "down_proj")))
        ):
            linear_names.append(k)
    linear_names.sort()
    print(f"can encode TQ33: {len(linear_names)} tensor (ky vong 18624)")

    extras_names = [k for k in tensors_meta if k not in linear_names and not k.endswith(".bias")]
    print(f"giu nguyen bf16: {len(extras_names)} tensor (embed/lm_head/norm/router/q_norm/k_norm)")

    manifest = {}
    t0 = time.time()
    total_bytes = 0
    for i, name in enumerate(linear_names):
        W = get(name)
        codes, scales = encode_linear(W)
        R, C = W.shape
        nb = C // G_SCALE
        fname = f"linears/{i:05d}.tq33"
        with open(os.path.join(OUT, fname), "wb") as f:
            f.write(codes.tobytes())
            f.write(scales.tobytes())
        manifest[name] = {"shape": [R, C], "nb": nb, "file": fname}
        total_bytes += codes.nbytes + scales.nbytes
        if i % 500 == 0 or i == len(linear_names) - 1:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(linear_names)}] {name}  "
                  f"({total_bytes/1e9:.2f}GB ghi, {dt:.0f}s, ~{(i+1)/max(dt,1e-9):.1f} tensor/s)",
                  flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # extras: ghi tuan tu vao 1 blob + index (giu nguyen dtype bf16 -> uint16 view)
    extras_manifest = {}
    off = 0
    with open(os.path.join(OUT, "extras.bin"), "wb") as f:
        for name in extras_names:
            t = get(name)
            raw = t.view(torch.uint16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
            f.write(raw)
            extras_manifest[name] = {"shape": list(t.shape), "dtype": str(t.dtype),
                                      "offset": off, "nbytes": len(raw)}
            off += len(raw)
    with open(os.path.join(OUT, "extras_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(extras_manifest, f)

    zf.close()
    tot_size = total_bytes + off
    print(f"\nXONG: {tot_size/1e9:.2f} GB tong ({total_bytes/1e9:.2f}GB TQ33 + {off/1e9:.2f}GB extras-bf16)")
    print(f"bpw hieu dung tren linear: {total_bytes*8/sum(m['shape'][0]*m['shape'][1] for m in manifest.values()):.3f}")


if __name__ == "__main__":
    main()
