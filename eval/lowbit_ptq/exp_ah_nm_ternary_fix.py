# -*- coding: utf-8 -*-
"""
Sua loi baseline trong exp_af: ternary DENSE co san CUNG log2(3)=1.585bpw, khong the bieu
dien duoc bat ky moc nao trong 5 moc CANON (0.331..1.566, 4/5 moc DUOI san nay). Baseline
DUNG phai la ternary N:M SPARSE (dung dinh nghia CANON: 1:32/1:16/1:8/1:4/2:4, mask theo
magnitude top-N/M, group=64 cho scale - dung uoc dinh nm_bpw trong exp_ab_subbit_curriculum.py).

Tai dung dung 16 tensor da fetch trong exp_af (fetch lai qua range-read, nhe).

Chay: python eval/lowbit_ptq/exp_ah_nm_ternary_fix.py
"""
import io
import json
import math
import os
import struct
import sys
import time
import urllib.request

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "allenai/OLMoE-1B-7B-0924"
SHARD = "model-00001-of-00003.safetensors"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
N_EXPERTS = 16
CANON = [(1, 32), (1, 16), (1, 8), (1, 4), (2, 4)]
AQLM_FROM_AF = {0.331: 0.8572, 0.474: 0.7938, 0.699: 0.6695, 1.023: 0.5905, 1.566: 0.4344}
AQLM_BPW_FROM_AF = {0.331: 0.3125, 0.474: 0.4375, 0.699: 0.75, 1.023: 1.0, 1.566: 1.5}
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ah_results.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def http_range(url, start, end):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                "Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class RemoteSafetensors:
    def __init__(self, url):
        self.url = url
        n = struct.unpack("<Q", http_range(url, 0, 8))[0]
        self.header = json.loads(http_range(url, 8, 8 + n))
        self.data_start = 8 + n

    def get(self, name):
        meta = self.header[name]
        dtype, shape, (s, e) = meta["dtype"], meta["shape"], meta["data_offsets"]
        raw = http_range(self.url, self.data_start + s, self.data_start + e)
        torch_dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[dtype]
        return torch.frombuffer(bytearray(raw), dtype=torch_dtype).view(*shape).clone().float()


def to_f8(s):
    sign = torch.sign(s)
    a = s.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a))
    m = a / (2 ** e)
    return sign * (2 ** e) * (torch.round(m * 8) / 8)


def nm_bpw(n, m, sgroup=64, sbits=8.0):
    return n / m * math.log2(3) + math.log2(math.comb(m, n)) / m + sbits / sgroup


def nm_ternary_err(W, n, m, sgroup=64):
    """Mask = top-n/m theo |W| (giong Wanda nhung KHONG co activation - dung |W| thuan,
    dung tinh than 'PTQ khong-calib' de so sanh cong bang voi AQLM khong-calib cua exp_af)."""
    R, C = W.shape
    Wg = W.view(R, -1, m)
    idx = Wg.abs().topk(n, dim=2).indices
    mask = torch.zeros_like(Wg)
    mask.scatter_(2, idx, 1.0)
    mask = mask.view(R, C)
    Wv = W.view(R, -1, sgroup)
    Mv = mask.view(R, -1, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    sq = to_f8(s)
    Wt = (torch.round(Wm / sq.clamp(min=1e-8)).clamp(-1, 1) * sq * Mv).view(R, C)
    err = ((Wt - W).pow(2).sum() / W.pow(2).sum().clamp(min=1e-12)).sqrt().item()
    return err, nm_bpw(n, m, sgroup)


def main():
    rs = RemoteSafetensors(BASE)
    names = [f"model.layers.0.mlp.experts.{e}.down_proj.weight" for e in range(N_EXPERTS)]
    log(f"tai lai {N_EXPERTS} tensor (giong exp_af)...")
    Ws = [rs.get(n) for n in names]
    log("  xong.")

    rows = []
    for n, m in CANON:
        errs, bpws = [], []
        for w in Ws:
            e, b = nm_ternary_err(w, n, m)
            errs.append(e)
            bpws.append(b)
        mean_err, bpw = sum(errs) / len(errs), bpws[0]
        rows.append({"nm": f"{n}:{m}", "bpw": round(bpw, 4), "ternary_nm_err": round(mean_err, 4)})
        log(f"  ternary {n}:{m} @ {bpw:.4f}bpw: err {mean_err:.4f}")

    # ghep voi so AQLM da co (exp_af, khong tinh lai - AQLM khong doi, chi baseline doi)
    target_map = {0.331: 0, 0.474: 1, 0.699: 2, 1.023: 3, 1.566: 4}
    log("\n=== BANG DA SUA (AQLM tu exp_af, ternary N:M sparse THAY the ternary dense sai) ===")
    for target, idx in target_map.items():
        row = rows[idx]
        aqlm_err = AQLM_FROM_AF[target]
        aqlm_bpw = AQLM_BPW_FROM_AF[target]
        win = aqlm_err < row["ternary_nm_err"]
        log(f"  ~{target}bpw: AQLM {aqlm_err:.4f}@{aqlm_bpw:.3f}bpw  vs  "
            f"ternary-{row['nm']} {row['ternary_nm_err']:.4f}@{row['bpw']:.3f}bpw"
            f"  -> AQLM {'THANG' if win else 'THUA'}")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"ternary_nm_sparse": rows, "aqlm_from_exp_af": AQLM_FROM_AF,
                  "aqlm_bpw_from_exp_af": AQLM_BPW_FROM_AF}, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
