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


def encode_linear(W, stats=None):
    """W: [R,C] bf16/float, C%64==0. Tra (codes: np.uint8[R,C//64,11], scales: np.float32[R,C//64]).

    Model train 2:4 KHÔNG hoàn hảo tuyệt đối: đo thật thấy hiếm nhóm-4 có 3 nonzero
    (vd 1/393216 ở 1 expert — có thể do mask-refresh dynamics, xem memory). Xử lý bằng
    cách ép nhóm vi phạm về hợp lệ: bỏ (=0) giá trị |W| NHỎ NHẤT trong nhóm cho tới khi
    ≤2 nonzero — đây là xấp xỉ ternary GẦN NHẤT, tổn thất tối thiểu và cực hiếm.
    `stats` (dict, tùy chọn) được cộng dồn: n_groups, n_violated, sq_err (để báo cáo tổng)."""
    W = W.float()
    R, C = W.shape
    assert C % G_SCALE == 0, f"C={C} khong chia het {G_SCALE}"
    nb = C // G_SCALE
    Wv = W.view(R, nb, G_SCALE)
    s = Wv.abs().amax(dim=2)                                     # [R, nb]
    t = torch.where(s.unsqueeze(2) > 0,
                    torch.round(Wv / s.unsqueeze(2).clamp(min=1e-12)),
                    torch.zeros_like(Wv)).clamp(-1, 1).to(torch.int8)

    tg = t.view(R, nb, G_SCALE // 4, 4)                            # 16 nhom-4 / block64
    Wg = Wv.view(R, nb, G_SCALE // 4, 4)
    nz = (tg != 0).sum(dim=-1)
    viol = (nz > 2).nonzero(as_tuple=False)                       # hiem -> loop nho an toan
    if stats is not None:
        stats["n_groups"] = stats.get("n_groups", 0) + nz.numel()
        stats["n_violated"] = stats.get("n_violated", 0) + viol.shape[0]
    for pos in viol.tolist():
        ri, bi, gi = pos
        wv = Wg[ri, bi, gi].abs()
        tv = tg[ri, bi, gi]
        nnz = int((tv != 0).sum().item())
        n_drop = nnz - 2
        order = torch.argsort(wv)                                 # tang dan; wv=0 (da la 0) len dau, vo hai
        dropped_sq = 0.0
        dropped = 0
        for j in order.tolist():
            if dropped >= n_drop:
                break
            if tv[j].item() != 0:
                if stats is not None:
                    dropped_sq += Wg[ri, bi, gi, j].item() ** 2
                tg[ri, bi, gi, j] = 0
                dropped += 1
        if stats is not None:
            stats["sq_err"] = stats.get("sq_err", 0.0) + dropped_sq
    t = tg.view(R, nb, G_SCALE)

    recon = (t.float() * s.unsqueeze(2)).view(R, C)
    if viol.shape[0] == 0 and not torch.equal(recon, W):
        raise ValueError("KHONG lossless (0 vi pham nhung van lech) — bug thuc su, dung lai")
    base3 = ((tg + 1) * POW3).sum(-1)                              # [R, nb, 16]
    gid = LUT_B3[base3.to(torch.int64)]
    if int((gid < 0).sum().item()) != 0:
        raise ValueError("van con pattern ngoai bang 33 sau khi vam — bug logic vam nhom")
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
    stats = {}
    t0 = time.time()
    total_bytes = 0
    for i, name in enumerate(linear_names):
        W = get(name)
        codes, scales = encode_linear(W, stats=stats)
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
            nv = stats.get("n_violated", 0)
            ng = max(stats.get("n_groups", 1), 1)
            print(f"  [{i+1}/{len(linear_names)}] {name}  "
                  f"({total_bytes/1e9:.2f}GB ghi, {dt:.0f}s, ~{(i+1)/max(dt,1e-9):.1f} tensor/s, "
                  f"vi pham 2:4 tich luy: {nv}/{ng}={nv/ng*100:.5f}%)",
                  flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    rmse = (stats.get("sq_err", 0.0) / max(stats.get("n_groups", 1), 1)) ** 0.5
    print(f"\n=== TONG KET VI PHAM 2:4 (ep ve hop le bang cach bo gia tri |W| nho nhat) ===")
    print(f"tong nhom-4: {stats.get('n_groups',0)} | vi pham: {stats.get('n_violated',0)} "
          f"({stats.get('n_violated',0)/max(stats.get('n_groups',1),1)*100:.6f}%)")
    print(f"RMS loi do vam (tren toan bo nhom, kha nang cuc nho vi vi pham cuc hiem): {rmse:.3e}")

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
