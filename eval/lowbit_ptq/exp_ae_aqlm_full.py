# -*- coding: utf-8 -*-
"""
Buoc 1 (RESEARCH_AQLM_CODEBOOK_PTQ.md muc 4.2): AQLM DAY DU - beam-search assignment (dung
thuat toan, khong phai residual-greedy don gian cua Buoc 0) + gradient-refine codebook,
POOL nhieu expert cung 1 bo codebook CHUNG (thuc te hon per-tensor rieng: giam overhead/
trong so + kiem tra codebook co tong quat hoa qua nhieu expert khong - cau hoi thuc dung
that, khong chi ly thuyet).

Ablation 4 buoc (dung van hoa "quy cong/toi" Bai 10 cua lab):
  (a) k-means-only          - mocc Buoc 0
  (b) + beam-search assign  - do dong gop rieng cua beam search
  (c) + gradient-refine     - do dong gop rieng cua refine (assignment CO DINH tu (b))
  (d) + 1 vong alternating  - re-assign bang codebook da refine, refine lai lan 2

Van tren OLMoE-1B-7B (Allen AI, 7B/1B active) qua HTTP range-read - KHONG tai ca model.

Chay: python eval/lowbit_ptq/exp_ae_aqlm_full.py
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

REPO = "allenai/OLMoE-1B-7B-0924"
SHARD = "model-00001-of-00003.safetensors"
BASE = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"
N_EXPERTS = 16          # pool 16/64 expert cua layer 0 (mo rong sau neu tin hieu tot)
LAYER = 0
TENSOR_KIND = "down_proj"
G, K, M = 8, 256, 2      # group, codeword/codebook, so codebook (~2bpw payload)
BEAM = 4
REFINE_STEPS = 400
REFINE_BATCH = 65536
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ae_results.json")

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


# ---------------- q_ternary (copy tu exp_l_true_sub1bit.py, xem exp_ad_aqlm_toy.py ly do) ----------------
def to_f8(s):
    sign = torch.sign(s)
    a = s.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a))
    m = a / (2 ** e)
    return sign * (2 ** e) * (torch.round(m * 8) / 8)


def q_ternary_dense(W, sgroup):
    R, C = W.shape
    G_ = int(sgroup)
    Wv = W.view(R, -1, G_)
    s = Wv.abs().mean(2, keepdim=True).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wv / s).clamp(-1, 1)
        num, den = (Wv * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    sq = to_f8(s)
    return (torch.round(Wv / sq).clamp(-1, 1) * sq).reshape(R, C)


# ---------------- AQLM day du ----------------
def kmeans(X, K, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    C = X[torch.randperm(N, generator=g)[:K]].clone()
    for it in range(iters):
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
    """Beam search THAT (khong phai residual-greedy): voi moi diem, giu top-`beam` ung vien
    codebook1, voi MOI ung vien tim codebook2 tot nhat cho residual, chon cap tong loi be nhat.
    Xu ly theo chunk de bound memory (N co the >10M diem)."""
    N = X.shape[0]
    a1 = torch.empty(N, dtype=torch.long)
    a2 = torch.empty(N, dtype=torch.long)
    for i in range(0, N, chunk):
        x = X[i:i + chunk]                                   # [n, g]
        d1 = torch.cdist(x, C1)                               # [n, K]
        top1v, top1i = d1.topk(beam, dim=1, largest=False)    # [n, beam]
        best_err = None
        best_i1 = best_i2 = None
        for b in range(beam):
            i1 = top1i[:, b]                                  # [n]
            resid = x - C1[i1]                                # [n, g]
            d2 = torch.cdist(resid, C2)                        # [n, K]
            e2, i2 = d2.min(dim=1)                             # [n]
            if best_err is None:
                best_err, best_i1, best_i2 = e2, i1, i2
            else:
                better = e2 < best_err
                best_err = torch.where(better, e2, best_err)
                best_i1 = torch.where(better, i1, best_i1)
                best_i2 = torch.where(better, i2, best_i2)
        a1[i:i + chunk], a2[i:i + chunk] = best_i1, best_i2
    return a1, a2


def refine_codebooks(X, C1, C2, a1, a2, steps=REFINE_STEPS, batch=REFINE_BATCH, lr=1e-2):
    """Assignment CO DINH (dung AQLM: alternating - co dinh assign, refine codebook value bang
    gradient). Loss = MSE tai tao, KHONG dung activation/teacher - dung PTQ scope Buoc 0/1."""
    C1r = C1.clone().requires_grad_(True)
    C2r = C2.clone().requires_grad_(True)
    opt = torch.optim.Adam([C1r, C2r], lr=lr)
    N = X.shape[0]
    g_ = torch.Generator().manual_seed(1)
    losses = []
    for st in range(steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=g_)
        x = X[idx]
        rec = C1r[a1[idx]] + C2r[a2[idx]]
        loss = F.mse_loss(rec, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if st % 100 == 99 or st == steps - 1:
            log(f"      refine step {st + 1}/{steps}: mse {loss.item():.5f}")
    return C1r.detach(), C2r.detach(), losses


def recon_err(X, C1, C2, a1, a2):
    rec = C1[a1] + C2[a2]
    return ((rec - X).pow(2).sum() / X.pow(2).sum().clamp(min=1e-12)).sqrt().item()


def bpw_of(n_weights, n_tensors, g=G, K=K, M=M):
    n_groups = n_weights // g
    payload = M * torch.log2(torch.tensor(float(K))).item() * n_groups
    codebook = M * K * g * 32 * n_tensors            # 1 codebook CHUNG cho n_tensors tensor
    return (payload + codebook) / n_weights


def main():
    rs = RemoteSafetensors(BASE)
    names = [f"model.layers.{LAYER}.mlp.experts.{e}.{TENSOR_KIND}.weight" for e in range(N_EXPERTS)]
    log(f"tai {N_EXPERTS} expert {TENSOR_KIND} layer {LAYER} qua range-read...")
    t0 = time.time()
    Ws = [rs.get(n) for n in names]
    shape = Ws[0].shape
    log(f"  xong {len(Ws)} tensor {tuple(shape)} moi ({sum(w.numel() for w in Ws) * 4 / 1e6:.1f}MB f32) "
        f"trong {time.time() - t0:.0f}s")

    R, C = shape
    X = torch.cat([w.reshape(-1, G) for w in Ws], dim=0)     # pool tat ca expert
    N = X.shape[0]
    log(f"pool: {N:,} group-{G} tu {N_EXPERTS} expert ({N * G:,} trong so)")

    # (a) k-means-only
    t0 = time.time()
    C1 = kmeans(X, K, seed=0)
    a1_km = torch.cdist(X, C1).argmin(1)
    resid = X - C1[a1_km]
    C2 = kmeans(resid, K, seed=1)
    a2_km = torch.cdist(resid, C2).argmin(1)
    err_a = recon_err(X, C1, C2, a1_km, a2_km)
    log(f"(a) k-means-only: err {err_a:.4f} ({time.time() - t0:.0f}s)")

    # (b) + beam-search assign (CUNG codebook tu (a), chi doi cach gan)
    t0 = time.time()
    a1_bm, a2_bm = beam_assign(X, C1, C2, beam=BEAM)
    err_b = recon_err(X, C1, C2, a1_bm, a2_bm)
    log(f"(b) +beam(width={BEAM}): err {err_b:.4f} ({time.time() - t0:.0f}s) "
        f"[{'tot hon' if err_b < err_a else 'KHONG tot hon'} (a)]")

    # (c) + gradient-refine (assignment CO DINH tu (b))
    t0 = time.time()
    C1r, C2r, losses = refine_codebooks(X, C1, C2, a1_bm, a2_bm)
    err_c = recon_err(X, C1r, C2r, a1_bm, a2_bm)
    log(f"(c) +refine: err {err_c:.4f} ({time.time() - t0:.0f}s) "
        f"[{'tot hon' if err_c < err_b else 'KHONG tot hon'} (b)]")

    # (d) + 1 vong alternating: re-assign voi codebook da refine, refine lan 2
    t0 = time.time()
    a1_2, a2_2 = beam_assign(X, C1r, C2r, beam=BEAM)
    C1r2, C2r2, _ = refine_codebooks(X, C1r, C2r, a1_2, a2_2, steps=REFINE_STEPS // 2)
    err_d = recon_err(X, C1r2, C2r2, a1_2, a2_2)
    log(f"(d) +1 vong alternating: err {err_d:.4f} ({time.time() - t0:.0f}s) "
        f"[{'tot hon' if err_d < err_c else 'KHONG tot hon'} (c)]")

    bpw = bpw_of(N * G, N_EXPERTS)
    log(f"bpw thuc (codebook CHUNG {N_EXPERTS} expert): {bpw:.4f}")

    # per-expert breakdown (co the 1 expert "la" bi codebook chung phuc vu kem hon)
    per_expert = []
    off = 0
    groups_per_expert = (R * C) // G
    for e in range(N_EXPERTS):
        sl = slice(off, off + groups_per_expert)
        per_expert.append(round(recon_err(X[sl], C1r2, C2r2, a1_2[sl], a2_2[sl]), 4))
        off += groups_per_expert
    log(f"per-expert err (final, codebook chung): min {min(per_expert)} max {max(per_expert)} "
        f"mean {sum(per_expert) / len(per_expert):.4f}")

    # baseline ternary tren CHINH cac tensor nay (khong pool - dung convention lab: per-tensor)
    tern_errs = {"g8": [], "g32": []}
    for w in Ws:
        tern_errs["g8"].append(((q_ternary_dense(w, 8) - w).pow(2).sum()
                                / w.pow(2).sum()).sqrt().item())
        tern_errs["g32"].append(((q_ternary_dense(w, 32) - w).pow(2).sum()
                                 / w.pow(2).sum()).sqrt().item())
    tern_g8 = sum(tern_errs["g8"]) / len(tern_errs["g8"])
    tern_g32 = sum(tern_errs["g32"]) / len(tern_errs["g32"])
    bpw_g8 = 1.5849625007211563 + 8.0 / 8
    bpw_g32 = 1.5849625007211563 + 8.0 / 32
    log(f"baseline ternary-Lloyd (mean {N_EXPERTS} expert): g8 err {tern_g8:.4f} @{bpw_g8:.3f}bpw"
        f" | g32 err {tern_g32:.4f} @{bpw_g32:.3f}bpw")

    out = {"model": REPO, "layer": LAYER, "tensor_kind": TENSOR_KIND, "n_experts": N_EXPERTS,
          "aqlm_bpw": round(bpw, 4),
          "ablation": {"a_kmeans_only": round(err_a, 4), "b_plus_beam": round(err_b, 4),
                       "c_plus_refine": round(err_c, 4), "d_plus_1round_alt": round(err_d, 4)},
          "per_expert_final_err": per_expert,
          "ternary_baseline": {"g8": {"bpw": round(bpw_g8, 3), "err": round(tern_g8, 4)},
                               "g32": {"bpw": round(bpw_g32, 3), "err": round(tern_g32, 4)}}}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== TOM TAT ===")
    log(f"AQLM full (d) @ {bpw:.3f}bpw: err {err_d:.4f}")
    log(f"ternary-Lloyd tot nhat: err {min(tern_g8, tern_g32):.4f} "
        f"@ {bpw_g8 if tern_g8 < tern_g32 else bpw_g32:.3f}bpw")
    win = err_d < min(tern_g8, tern_g32)
    log(f"=> AQLM day du {'THANG' if win else 'THUA'} ternary-Lloyd. "
        f"Dong gop tung buoc: kmeans->beam {err_a - err_b:+.4f}, "
        f"beam->refine {err_b - err_c:+.4f}, +1 vong alt {err_c - err_d:+.4f}")


if __name__ == "__main__":
    main()
