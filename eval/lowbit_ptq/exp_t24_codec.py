# -*- coding: utf-8 -*-
"""
Exp T24 — CODEC "TQ33" tự chế cho 2:4-ternary: 3 nhóm-4 (12 trọng số, ≤2 nonzero/nhóm)
gói vào 16 bit (33^3 = 35937 < 65536) + scale f8/g64 -> 1.458 bpw TỔNG, giải mã 1 lần
tra LUT 64K (L2-resident).

Phase 1 (Python reference, $0, local 0.6B):
  1) Thống kê ckpt bake thật: nonzero/nhóm-4, tính bpw thật của format.
  2) Encode -> decode -> ĐỐI CHIẾU BIT-LEVEL với tensor gốc (lossless bắt buộc).
  3) (bước sau) PPL với weight decode lại = PPL bake gốc.

Ckpt bake của lab: trong scale-group 64, nonzero có |w| = s duy nhất -> tách (t, s) exact.
"""
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")

CKPT = r"D:\Bit-Translate-data\qat_ckpts\qat_gen4_n4.pt"
G_SCALE = 64          # scale-group của ckpt lab 0.6B (g64-f8)
GROUP = 4             # nhóm mask 2:4

# ---- bảng 33 pattern hợp lệ của nhóm-4 (≤2 nonzero, giá trị ∈ {-1,0,+1}) ----
def build_patterns():
    pats = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                for d in (-1, 0, 1):
                    v = (a, b, c, d)
                    if sum(1 for x in v if x != 0) <= 2:
                        pats.append(v)
    assert len(pats) == 33, len(pats)
    return pats

PATS = build_patterns()
PAT_ID = {p: i for i, p in enumerate(PATS)}


def analyze_and_codec(W):
    """W: tensor bake [R, C] (t*s). Trả (stats, ok_lossless, bpw)."""
    R, C = W.shape
    assert C % 12 == 0 or C % GROUP == 0
    # tách scale per-64: s = max|w| trong scale-group
    Wv = W.view(R, C // G_SCALE, G_SCALE)
    s = Wv.abs().amax(dim=2, keepdim=True)                    # [R, C/64, 1]
    t = torch.where(s > 0, torch.round(Wv / s.clamp(min=1e-12)), torch.zeros_like(Wv))
    t = t.clamp(-1, 1)
    # kiểm tách exact: t*s == W?
    recon = (t * s).view(R, C)
    exact_split = torch.equal(recon, W)
    # thống kê nonzero per nhóm-4
    tg = t.view(R, C // GROUP, GROUP)
    nz = (tg != 0).sum(dim=2)
    hist = torch.bincount(nz.flatten().to(torch.int64), minlength=5)[:5]
    viol = int((nz > 2).sum().item())
    # encode 3 nhóm -> 1 codeword 16 bit (pad nhóm-zero cho chia hết 12 — format thật cũng vậy)
    t_flat = t.reshape(R, C).to(torch.int8)
    padC = (12 - C % 12) % 12
    if padC:
        t_flat = torch.nn.functional.pad(t_flat, (0, padC))
    t12 = t_flat.view(R, -1, 12)
    ids = torch.zeros(t12.shape[:2], dtype=torch.int32)
    for g in range(3):
        grp = t12[:, :, g * 4:(g + 1) * 4]
        gid = torch.zeros(grp.shape[:2], dtype=torch.int32)
        # tra id pattern bằng số hóa base-3 rồi map
        base3 = ((grp + 1) * torch.tensor([27, 9, 3, 1], dtype=torch.int8)).sum(-1)
        lut_b3 = torch.full((81,), -1, dtype=torch.int32)
        for pid, p in enumerate(PATS):
            b3 = sum((x + 1) * m for x, m in zip(p, (27, 9, 3, 1)))
            lut_b3[b3] = pid
        gid = lut_b3[base3.to(torch.int64)]
        assert int((gid < 0).sum().item()) == 0, "pattern ngoài bảng 33!"
        ids = ids * 33 + gid
    assert int(ids.max().item()) < 65536
    # decode ngược qua LUT 64K -> 12 trọng số
    lut = torch.zeros(33 ** 3, 12, dtype=torch.int8)
    for i0 in range(33):
        for i1 in range(33):
            base = (i0 * 33 + i1) * 33
            row01 = list(PATS[i0]) + list(PATS[i1])
            for i2 in range(33):
                lut[base + i2] = torch.tensor(row01 + list(PATS[i2]), dtype=torch.int8)
    t_dec = lut[ids.to(torch.int64)].view(R, -1)[:, :C].contiguous()
    t_dec = t_dec.view(R, C // G_SCALE, G_SCALE)
    W_dec = (t_dec.float() * s).view(R, C)
    lossless = torch.equal(W_dec, W)
    # bpw thật: 16 bit/12w + 8 bit scale/64w
    bpw = 16 / 12 + 8 / G_SCALE
    return dict(exact_split=exact_split, hist=hist.tolist(), viol=viol,
                lossless=lossless, bpw=bpw)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = sd["state_dict"] if "state_dict" in sd else sd
    keys = [k for k in state if k.endswith(".weight") and "layers." in k
            and ("proj" in k) and state[k].dim() == 2]
    print(f"{len(keys)} linear bake trong ckpt; codec thử 6 tensor đại diện...")
    tot_hist = torch.zeros(5, dtype=torch.long)
    all_ok = True
    t0 = time.time()
    for k in keys[:4] + keys[-2:]:
        W = state[k].float()
        r = analyze_and_codec(W)
        tot_hist += torch.tensor(r["hist"])
        all_ok &= (r["lossless"] and r["viol"] == 0)
        print(f"  {k.split('model.layers.')[-1]:44s} split={r['exact_split']} "
              f"lossless={r['lossless']} viol={r['viol']} nz-hist={r['hist']}")
    tot = int(tot_hist.sum().item())
    print(f"\nPhân bố nonzero/nhóm-4 (6 tensor): "
          + ", ".join(f"{i}nz={int(v)/tot*100:.1f}%" for i, v in enumerate(tot_hist[:3])))
    print(f"bpw format TQ33 = 16/12 + 8/64 = {16/12 + 8/64:.3f}")
    print(f"({time.time()-t0:.0f}s) PHÁN QUYẾT: "
          + ("LOSSLESS ✓ — format hợp lệ, sang bước PPL + C++"
             if all_ok else "CÓ VI PHẠM — xem lại giả định 2:4/tách scale"))


if __name__ == "__main__":
    main()
