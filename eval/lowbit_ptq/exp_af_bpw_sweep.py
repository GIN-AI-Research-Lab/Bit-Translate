# -*- coding: utf-8 -*-
"""
Buoc 2 (RESEARCH_AQLM_CODEBOOK_PTQ.md muc 4, chi dao user 04/08): quet 5 moc bpw chuan cua
lab (0.331/0.474/0.699/1.023/1.566 - dung CANON ladder exp_ab_subbit_curriculum.py) tu THAP
den CAO, AQLM (k-means+beam, BO refine - xem exp_ae: refine lr=1e-2 tung REGRESS) tai moc
gan nhat dat duoc bang (M,K,g), so voi ternary-Lloyd cung bpw. Them 1 phep TRON ky thuat dau
tien: Hadamard incoherence rotation (Bai 2 exp_c_incoherence.py) TRUOC AQLM, thu o moc THAP
NHAT (kho nhat, dung huong "tu nho toi lon" uu tien cho nho truoc).

Van tren OLMoE-1B-7B, 16 expert down_proj layer 0, pool 1 codebook chung (nhu exp_ae).

Chay: python eval/lowbit_ptq/exp_af_bpw_sweep.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "allenai/OLMoE-1B-7B-0924"
SHARD = "model-00001-of-00003.safetensors"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
N_EXPERTS = 16
LAYER = 0
TENSOR_KIND = "down_proj"
BEAM = 4
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_af_results.json")

# 5 moc CANON cua lab (README Bai 9/10/13/15, exp_ab CANON) + cau hinh AQLM (M,K,g) gan nhat
LADDER = [
    {"target_bpw": 0.331, "M": 1, "K": 32, "g": 16},
    {"target_bpw": 0.474, "M": 1, "K": 128, "g": 16},
    {"target_bpw": 0.699, "M": 1, "K": 64, "g": 8},
    {"target_bpw": 1.023, "M": 2, "K": 16, "g": 8},
    {"target_bpw": 1.566, "M": 2, "K": 64, "g": 8},
]

torch.manual_seed(0)


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


def q_ternary_dense(W, sgroup):
    R, C = W.shape
    Wv = W.view(R, -1, sgroup)
    s = Wv.abs().mean(2, keepdim=True).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wv / s).clamp(-1, 1)
        num, den = (Wv * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    sq = to_f8(s)
    return (torch.round(Wv / sq).clamp(-1, 1) * sq).reshape(R, C)


def hadamard_matrix(n):
    """Sylvester construction, orthonormal. n phai la luy thua 2."""
    assert n & (n - 1) == 0, f"n={n} phai luy thua 2"
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


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


def beam_assign(X, C1, C2, beam=BEAM, chunk=200_000):
    N = X.shape[0]
    a1 = torch.empty(N, dtype=torch.long)
    a2 = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        x = X[i:i + chunk]
        d1 = torch.cdist(x, C1)
        _, top1i = d1.topk(beam, dim=1, largest=False)
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


def aqlm_fit(X, M, K, g):
    """M=1 hoac M=2 codebook, k-means + beam-search assign. Tra (err, bpw_payload)."""
    Xg = X.reshape(-1, g)
    C1 = kmeans(Xg, K, seed=0)
    if M == 1:
        a1 = torch.cdist(Xg, C1).argmin(1)
        rec = C1[a1]
    else:
        resid = Xg - C1[torch.cdist(Xg, C1).argmin(1)]
        C2 = kmeans(resid, K, seed=1)
        a1, a2 = beam_assign(Xg, C1, C2, beam=BEAM)
        rec = C1[a1] + C2[a2]
    err = ((rec - Xg).pow(2).sum() / Xg.pow(2).sum().clamp(min=1e-12)).sqrt().item()
    payload_bpw = M * math.log2(K) / g
    return err, payload_bpw


def main():
    rs = RemoteSafetensors(BASE)
    names = [f"model.layers.{LAYER}.mlp.experts.{e}.{TENSOR_KIND}.weight" for e in range(N_EXPERTS)]
    log(f"tai {N_EXPERTS} expert {TENSOR_KIND} layer {LAYER}...")
    Ws = [rs.get(n) for n in names]
    X = torch.cat([w.reshape(-1) for w in Ws]).reshape(-1, Ws[0].shape[1])   # [N_EXPERTS*R, C]
    C_in = Ws[0].shape[1]
    log(f"  xong: pool {tuple(X.shape)}, C_in={C_in}")

    results = []
    H = None
    for i, cfg in enumerate(LADDER):
        M, K, g = cfg["M"], cfg["K"], cfg["g"]
        log(f"=== moc {i + 1}/5: target {cfg['target_bpw']}bpw -> AQLM M={M} K={K} g={g} ===")
        t0 = time.time()
        err_plain, bpw_plain = aqlm_fit(X, M, K, g)
        log(f"  AQLM (khong xoay): err {err_plain:.4f} @ {bpw_plain:.4f}bpw ({time.time() - t0:.0f}s)")

        row = {"target_bpw": cfg["target_bpw"], "cfg": {"M": M, "K": K, "g": g},
              "aqlm_bpw": round(bpw_plain, 4), "aqlm_err": round(err_plain, 4)}

        # ternary-Lloyd bracket cung target
        tern_g = max(8, min(64, int(round(8.0 / max(cfg["target_bpw"] - math.log2(3), 0.05)))))
        tern_g = 2 ** round(math.log2(tern_g))
        errs_t = []
        for w in Ws:
            errs_t.append(((q_ternary_dense(w, tern_g) - w).pow(2).sum()
                          / w.pow(2).sum()).sqrt().item())
        tern_bpw = math.log2(3) + 8.0 / tern_g
        tern_err = sum(errs_t) / len(errs_t)
        row["ternary_bracket"] = {"g": tern_g, "bpw": round(tern_bpw, 4), "err": round(tern_err, 4)}
        log(f"  ternary-Lloyd g{tern_g}: err {tern_err:.4f} @ {tern_bpw:.4f}bpw"
            f" -> AQLM {'THANG' if err_plain < tern_err else 'THUA'}")

        if i == 0:   # moc THAP NHAT: thu TRON Hadamard incoherence (Bai 2) truoc AQLM
            t0 = time.time()
            if H is None:
                H = hadamard_matrix(C_in)
            Wrot = [w @ H for w in Ws]
            Xrot = torch.cat([w.reshape(-1) for w in Wrot]).reshape(-1, C_in)
            err_rot_basis, _ = aqlm_fit(Xrot, M, K, g)
            # xoay nguoc lai de so sanh dung basis goc (Hadamard truc giao -> norm bao toan,
            # nhung fit lai tren Xrot roi danh gia trong basis xoay - de doi chieu cong bang,
            # danh gia CA HAI trong basis rieng cua no, vi H truc giao bao toan L2-norm nen
            # rel-energy-err KHONG doi giua 2 basis - chi can so 2 so err truc tiep).
            log(f"  +Hadamard truoc AQLM: err {err_rot_basis:.4f} @ {bpw_plain:.4f}bpw"
                f" ({time.time() - t0:.0f}s) -> {'TOT HON' if err_rot_basis < err_plain else 'KHONG tot hon'}"
                f" ban khong xoay ({err_plain:.4f})")
            row["hadamard_combo"] = {"err": round(err_rot_basis, 4),
                                     "better_than_plain": bool(err_rot_basis < err_plain)}

        results.append(row)

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"model": REPO, "n_experts": N_EXPERTS, "ladder": results}, f,
                  ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== TOM TAT SWEEP (nho -> lon) ===")
    for row in results:
        win = row["aqlm_err"] < row["ternary_bracket"]["err"]
        log(f"  {row['target_bpw']:.3f}bpw: AQLM {row['aqlm_err']:.4f}@{row['aqlm_bpw']:.3f} "
            f"vs ternary {row['ternary_bracket']['err']:.4f}@{row['ternary_bracket']['bpw']:.3f} "
            f"-> {'THANG' if win else 'THUA'}"
            + (f" | +Hadamard: {row['hadamard_combo']['err']:.4f}" if "hadamard_combo" in row else ""))


if __name__ == "__main__":
    main()
