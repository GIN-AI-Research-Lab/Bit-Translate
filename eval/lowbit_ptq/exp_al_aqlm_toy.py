# -*- coding: utf-8 -*-
"""
Buoc 0 (RESEARCH_AQLM_CODEBOOK_PTQ.md muc 4.1): AQLM (k-means thuan, KHONG gradient-refine)
vs q_ternary (Lloyd 3 vong, dung xuyen suot lab) tren TENSOR THAT cua 1 model MoE NHE
(OLMoE-1B-7B, Allen AI: 7B tong / 1B active, 64 expert/layer, expert FFN 1024x2048 -
nho hon rat nhieu 30B-A3B/Laguna 118B). Tai dung DUNG 1-2 tensor qua HTTP range-read
(khong tai ca model - ky thuat giong Kimi-K3-in-C "sampled tensors over HTTP range reads").

Chay: python eval/lowbit_ptq/exp_ad_aqlm_toy.py
"""
import io
import json
import os
import struct
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# q_ternary + helper (copy nguyen tu exp_l_true_sub1bit.py - KHONG import truc tiep vi file do
# co code top-level tu glob D:\Bit-Translate-data\... (khong ton tai o may nay) ngay khi import.
# Khong sua file goc (nguyen tac lab: khong dung file da validate).
def to_f8(s):
    sign = torch.sign(s)
    a = s.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a))
    m = a / (2 ** e)
    return sign * (2 ** e) * (torch.round(m * 8) / 8)


def to_f16(s):
    return s.to(torch.float16).to(torch.float32)


SCALE_Q = {"f16": to_f16, "f8": to_f8}


def group_views(W, mask, sgroup):
    R, C = W.shape
    M = torch.ones_like(W) if mask is None else mask.float()
    if sgroup == "tensor":
        return W.reshape(1, 1, -1), M.reshape(1, 1, -1), (R, C, 0)
    if sgroup == "row":
        return W.view(R, 1, C), M.view(R, 1, C), (R, C, 0)
    G = int(sgroup)
    pad = (G - C % G) % G
    Wp = F.pad(W, (0, pad)) if pad else W
    Mp = F.pad(M, (0, pad)) if pad else M
    return Wp.view(R, -1, G), Mp.view(R, -1, G), (R, C, pad)


def q_ternary(W, mask, sgroup, sdtype):
    Wv, Mv, (R, C, pad) = group_views(W, mask, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num = (Wm * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    sq = SCALE_Q[sdtype](s)
    t = torch.round(Wm / sq.clamp(min=1e-8)).clamp(-1, 1) * Mv
    out = (t * sq).reshape(R, -1)[:, :C]
    return out

REPO = "allenai/OLMoE-1B-7B-0924"
SHARD = "model-00001-of-00003.safetensors"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
TENSORS = [
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
    "model.layers.0.mlp.experts.7.down_proj.weight",   # expert khac, kiem tra khong phai may man 1 tensor
]
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ad_results.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def http_range(url, start, end):
    """[start, end) bytes, HTTP Range request - khong tai ca file."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end - 1}",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class RemoteSafetensors:
    """Doc header 1 lan, sau do range-read dung tensor can - khong tai ca shard (~4-5GB)."""

    def __init__(self, url):
        self.url = url
        head = http_range(url, 0, 8)
        n = struct.unpack("<Q", head)[0]
        hdr_bytes = http_range(url, 8, 8 + n)
        self.header = json.loads(hdr_bytes)
        self.data_start = 8 + n
        log(f"safetensors header doc xong: {len(self.header)} tensor, header {n} byte "
            f"(KHONG tai {SHARD} day - chi tai header + tensor can)")

    def get(self, name):
        meta = self.header[name]
        dtype, shape, (s, e) = meta["dtype"], meta["shape"], meta["data_offsets"]
        raw = http_range(self.url, self.data_start + s, self.data_start + e)
        assert dtype in ("BF16", "F16", "F32"), f"dtype la {dtype}, chua ho tro"
        torch_dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[dtype]
        t = torch.frombuffer(bytearray(raw), dtype=torch_dtype).view(*shape).clone()
        return t.float()


# ============================ AQLM toy: residual k-means, M=2, K=256, g=8 ============================
def kmeans(X, K, iters=25, seed=0):
    """X: [N, g]. Tra ve centroids [K, g]. Init = subsample tu data (on dinh hon random)."""
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    idx0 = torch.randperm(N, generator=g)[:K]
    C = X[idx0].clone()
    for it in range(iters):
        d = torch.cdist(X, C)              # [N, K]
        assign = d.argmin(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K)
        newC.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(N))
        empty = cnt == 0
        cnt = cnt.clamp(min=1)
        newC = newC / cnt.unsqueeze(1)
        newC[empty] = C[empty]             # centroid rong: giu nguyen, khong NaN
        shift = (newC - C).norm()
        C = newC
        if shift < 1e-5:
            break
    return C, it + 1


def aqlm_kmeans_only(W, g=8, K=256, M=2, seed=0):
    """Residual multi-codebook k-means (KHONG gradient-refine - dung Buoc 0).
    Tra (What, bpw_thuc, info)."""
    R, C = W.shape
    assert C % g == 0, f"C={C} phai chia het g={g}"
    Wg = W.reshape(-1, g)                  # [n_groups, g]
    resid = Wg.clone()
    codebooks, assigns = [], []
    for m in range(M):
        cb, nit = kmeans(resid, K, seed=seed + m)
        d = torch.cdist(resid, cb)
        a = d.argmin(1)
        codebooks.append(cb)
        assigns.append(a)
        resid = resid - cb[a]
        log(f"    codebook {m + 1}/{M}: kmeans {nit} vong, resid-norm sau {resid.norm():.4f}")
    recon = sum(cb[a] for cb, a in zip(codebooks, assigns))
    What = recon.reshape(R, C)
    n_weights = R * C
    n_groups = n_weights // g
    payload_bits = M * torch.log2(torch.tensor(float(K))) * n_groups   # index bits
    codebook_bits = M * K * g * 32                                     # fp32 codebook, PER TENSOR
    bpw = (payload_bits.item() + codebook_bits) / n_weights
    return What, bpw, {"payload_bpw": payload_bits.item() / n_weights,
                       "codebook_overhead_bpw": codebook_bits / n_weights}


def rel_energy_err(W, What):
    return ((What - W).pow(2).sum() / W.pow(2).sum().clamp(min=1e-12)).sqrt().item()


def main():
    url_ok = True
    try:
        rs = RemoteSafetensors(BASE)
    except Exception as e:
        log(f"KHONG fetch duoc header remote ({type(e).__name__}: {e}) - dung lai, bao loi ro rang")
        url_ok = False
    if not url_ok:
        return

    results = {}
    for name in TENSORS:
        log(f"=== {name} ===")
        t0 = time.time()
        W = rs.get(name)
        log(f"  tai xong {tuple(W.shape)} ({W.numel() * 2 / 1e6:.2f}MB bf16 goc) trong {time.time() - t0:.1f}s")

        # AQLM k-means-only, g=8, M=2, K=256 (~2bpw payload + overhead codebook)
        What_aq, bpw_aq, info_aq = aqlm_kmeans_only(W, g=8, K=256, M=2)
        err_aq = rel_energy_err(W, What_aq)

        # ternary Lloyd baseline, 2 group size de bracket (g=8 va g=32), f8 scale
        errs_tern = {}
        for gs, sdt in ((8, "f8"), (32, "f8")):
            Wt = q_ternary(W, None, gs, sdt)
            bpw_t = 1.5849625007211563 + 8.0 / gs   # log2(3) + sbits/group
            errs_tern[f"g{gs}_{sdt}"] = {"bpw": round(bpw_t, 3),
                                         "rel_energy_err": round(rel_energy_err(W, Wt), 4)}

        row = {"shape": list(W.shape),
               "aqlm_kmeans_only": {"bpw": round(bpw_aq, 3), "rel_energy_err": round(err_aq, 4),
                                    **{k: round(v, 4) for k, v in info_aq.items()}},
               "ternary_lloyd": errs_tern}
        results[name] = row
        log(f"  AQLM(kmeans) {bpw_aq:.3f}bpw err={err_aq:.4f}  |  "
            + " | ".join(f"ternary_{k} {v['bpw']}bpw err={v['rel_energy_err']}"
                        for k, v in errs_tern.items()))

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": REPO, "note": "Buoc 0 toy: AQLM k-means-only (khong gradient-refine) "
                                          "vs q_ternary Lloyd, tren tensor THAT qua HTTP range-read",
                  "results": results}, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== KET LUAN SOM (Buoc 0, chua gradient-refine) ===")
    any_win = False
    for name, row in results.items():
        best_tern = min(row["ternary_lloyd"].values(), key=lambda v: v["rel_energy_err"])
        win = row["aqlm_kmeans_only"]["rel_energy_err"] < best_tern["rel_energy_err"]
        any_win = any_win or win
        log(f"  {name.split('.')[-2]}.{name.split('.')[-1]}: AQLM "
            f"{'THANG' if win else 'THUA'} ternary-Lloyd tot nhat "
            f"({row['aqlm_kmeans_only']['rel_energy_err']:.4f} vs {best_tern['rel_energy_err']:.4f}"
            f" @ {best_tern['bpw']}bpw, AQLM @ {row['aqlm_kmeans_only']['bpw']:.3f}bpw)")
    log(f"=> {'CO tin hieu tich cuc — xem xet Buoc 1 (gradient-refine + toan FFN)' if any_win else 'AQLM (k-means thuan) KHONG thang ngay - can gradient-refine truoc khi ket luan (dung dung Buoc 0 de bac bo han)'}")


if __name__ == "__main__":
    main()
