# -*- coding: utf-8 -*-
"""
Kiem tra tinh tong quat cua ket qua chinh (AQLM thang ternary N:M sparse): Buoc 1/2 chi
test 16/64 expert, CHI layer 0. O day mo rong: TOAN BO 64 expert, 2 layer (0 va 8 - dai
dien dau va giua model), 2 moc bien (0.331 thap nhat, 1.566 cao nhat - dung huong "tu nho
toi lon"). Dung file GGUF cuc bo (khong can range-read qua mang - nhanh hon).

Chay: python eval/lowbit_ptq/exp_aj_generalization.py
"""
import io
import json
import math
import os
import sys
import time

import gguf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
LAYERS = [0, 8]
CONFIGS = [{"target_bpw": 0.331, "M": 1, "K": 32, "g": 16, "nm": (1, 32)},
          {"target_bpw": 1.566, "M": 2, "K": 64, "g": 8, "nm": (2, 4)}]
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_aj_results.json")

torch.manual_seed(0)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dq(byname, name):
    t = byname[name]
    return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()


def to_f8(s):
    sign = torch.sign(s)
    a = s.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a))
    m = a / (2 ** e)
    return sign * (2 ** e) * (torch.round(m * 8) / 8)


def nm_bpw(n, m, sgroup=64, sbits=8.0):
    return n / m * math.log2(3) + math.log2(math.comb(m, n)) / m + sbits / sgroup


def ternary_nm_err(W, n, m, sgroup=64):
    R, C = W.shape
    Wg = W.view(R, -1, m)
    idx = Wg.abs().topk(n, dim=2).indices
    mask = torch.zeros_like(Wg).scatter_(2, idx, 1.0).view(R, C)
    Wv, Mv = W.view(R, -1, sgroup), mask.view(R, -1, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    Wt = (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C)
    return ((Wt - W).pow(2).sum() / W.pow(2).sum().clamp(min=1e-12)).sqrt().item()


def kmeans(X, K, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    C = X[torch.randperm(N, generator=g)[:K]].clone()
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(1)
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K)
        newC.index_add_(0, assign, X)
        cnt.index_add_(0, assign, torch.ones(N))
        empty = cnt == 0
        newC = newC / cnt.clamp(min=1).unsqueeze(1)
        newC[empty] = C[empty]
        if (newC - C).norm() < 1e-5:
            C = newC
            break
        C = newC
    return C


def beam_assign(X, C1, C2, beam=4, chunk=200_000):
    N = X.shape[0]
    a1 = torch.empty(N, dtype=torch.long)
    a2 = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        x = X[i:i + chunk]
        _, top1i = torch.cdist(x, C1).topk(beam, dim=1, largest=False)
        best_err = best_i1 = best_i2 = None
        for b in range(beam):
            i1 = top1i[:, b]
            resid = x - C1[i1]
            e2, i2 = torch.cdist(resid, C2).min(dim=1)
            if best_err is None:
                best_err, best_i1, best_i2 = e2, i1, i2
            else:
                better = e2 < best_err
                best_err = torch.where(better, e2, best_err)
                best_i1 = torch.where(better, i1, best_i1)
                best_i2 = torch.where(better, i2, best_i2)
        a1[i:i + chunk], a2[i:i + chunk] = best_i1, best_i2
    return a1, a2


def aqlm_err(X, M, K, g):
    Xg = X.reshape(-1, g)
    C1 = kmeans(Xg, K, seed=0)
    if M == 1:
        a1 = torch.cdist(Xg, C1).argmin(1)
        rec = C1[a1]
    else:
        resid = Xg - C1[torch.cdist(Xg, C1).argmin(1)]
        C2 = kmeans(resid, K, seed=1)
        a1, a2 = beam_assign(Xg, C1, C2, beam=4)
        rec = C1[a1] + C2[a2]
    return ((rec - Xg).pow(2).sum() / Xg.pow(2).sum().clamp(min=1e-12)).sqrt().item()


def main():
    log(f"doc GGUF: {GGUF_PATH}")
    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}

    results = []
    for layer in LAYERS:
        log(f"dequant TOAN BO 64 expert down_proj, layer {layer}...")
        t0 = time.time()
        down = dq(byname, f"blk.{layer}.ffn_down_exps.weight")   # (64, 2048, 1024)
        log(f"  xong {tuple(down.shape)} ({time.time() - t0:.0f}s)")
        Ws = [down[e] for e in range(64)]
        X_all = down.reshape(-1, down.shape[-1])                  # pool 64 expert THAT (khong phai 16)

        for cfg in CONFIGS:
            log(f"=== layer {layer}, moc {cfg['target_bpw']}bpw (64 EXPERT, khong phai 16) ===")
            t0 = time.time()
            err_aqlm = aqlm_err(X_all, cfg["M"], cfg["K"], cfg["g"])
            bpw_aqlm = cfg["M"] * math.log2(cfg["K"]) / cfg["g"]
            log(f"  AQLM (64 expert pooled): err {err_aqlm:.4f} @ {bpw_aqlm:.3f}bpw"
                f" ({time.time() - t0:.0f}s)")

            n, m = cfg["nm"]
            errs = [ternary_nm_err(w, n, m) for w in Ws]
            err_tern = sum(errs) / len(errs)
            bpw_tern = nm_bpw(n, m)
            win = err_aqlm < err_tern
            log(f"  ternary {n}:{m} (64 expert, tung tensor): err {err_tern:.4f} @ {bpw_tern:.3f}bpw"
                f" -> AQLM {'THANG' if win else 'THUA'}")

            results.append({"layer": layer, "target_bpw": cfg["target_bpw"], "n_experts": 64,
                            "aqlm_err": round(err_aqlm, 4), "aqlm_bpw": round(bpw_aqlm, 4),
                            "ternary_err": round(err_tern, 4), "ternary_bpw": round(bpw_tern, 4),
                            "aqlm_wins": bool(win)})

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== TOM TAT TONG QUAT HOA (64 expert, 2 layer) ===")
    for row in results:
        log(f"  layer {row['layer']} @ {row['target_bpw']}bpw: AQLM {row['aqlm_err']}"
            f" vs ternary {row['ternary_err']} -> {'THANG' if row['aqlm_wins'] else 'THUA'}")
    n_win = sum(1 for r in results if r["aqlm_wins"])
    log(f"=> AQLM thang {n_win}/{len(results)} to hop (layer x bpw) voi 64/64 expert (khong phai mau 16)")


if __name__ == "__main__":
    main()
