# -*- coding: utf-8 -*-
"""
Exp K — GÓI SỬA LỖI + SEQUENTIAL cho ternary 2:4. Ba arm, mỗi arm cô lập một câu hỏi:

  Arm 1b: dense ternary + fix-pack, teacher-forcing   -> sweet-spot 2:4 có sống khi NGANG bước tối ưu?
  Arm 1 : ternary 2:4 + fix-pack, teacher-forcing     -> fix-pack (scale Lloyd survivor + mask Wanda
                                                          + 100 step + calib 60 câu) ăn được bao nhiêu?
  Arm 2 : ternary 2:4 + fix-pack, SEQUENTIAL block-wise (BRECQ-lite) -> error-propagation đáng giá
                                                          bao nhiêu? (mỗi block thấy input ĐÃ lượng tử
                                                          của prefix, tối ưu 7 linear CÙNG LÚC để khớp
                                                          quỹ đạo FP -> block sau BÙ lỗi block trước)

FIX-PACK (sửa các lỗi audit từ exp_h/i):
  1. Scale Lloyd trên SURVIVOR (thay absmean cả group gồm cả phần tử bị mask=0 -> scale quá bé).
  2. Mask N:M theo Wanda: importance = |W| * ||X_channel|| (thay |W| thuần — mù activation).
  3. Nhiều bước hơn (100 TF / 70 seq) + calib 60 câu (thay 40).
KẾ TOÁN BIT TRUNG THỰC (sửa lỗi exp_i thiếu scale):
  ternary 2:4 = 0.5*1.58 (payload) + log2(C(4,2))/4 (mask) + 16/32 (scale f16/g32) = 1.94 bpw
  dense       = 1.58 + 0.5 = 2.08 bpw
"""
import glob
import io
import json
import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(5)

MODEL_DIR = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*")[0]
DEV_VI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEV_JA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_JSON = r"e:\Bit-Translate\eval\lowbit_ptq\exp_k_results.json"
GROUP = 32
TF_STEPS = 100
TF_LR = 1.5e-3
SEQ_STEPS = 70
SEQ_LR = 1e-3
SEQ_BATCH = 6
N_CALIB = 60
N_EVAL = 16
MAX_TOK = 96

LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(ln)
            if len(out) >= n:
                break
    return out


def grp_pad(T):
    R, C = T.shape
    pad = (GROUP - C % GROUP) % GROUP
    Tp = F.pad(T, (0, pad)) if pad else T
    return Tp.view(R, -1, GROUP), C


def lloyd_masked_ternary(W, mask):
    """Ternary {-1,0,+1}*s per group-32, mask cố định (True=giữ), scale Lloyd trên survivor."""
    Wg, C = grp_pad(W)
    if mask is None:
        Mg = torch.ones_like(Wg)
    else:
        Mg, _ = grp_pad(mask.float())
    Wm = Wg * Mg
    cnt = Mg.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mg
        num = (Wm * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    t = torch.round(Wm / s).clamp(-1, 1) * Mg
    return (t * s).view(W.shape[0], -1)[:, :C]


def wanda_nm_mask(W, xnorm, N=2, M=4):
    """mask N:M theo importance |W|*xnorm (Wanda), nhóm M theo chiều input."""
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def bpw(mask_keep, N=None, M=None, scale_bits=16.0 / GROUP):
    payload = mask_keep * 1.58
    mask_cost = math.log2(math.comb(M, N)) / M if N else 0.0
    return payload + mask_cost + scale_bits


@torch.no_grad()
def eval_ppl(model, tok, lines):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
        if ids.shape[1] < 2:
            continue
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return float(np.exp(nll / max(ntok, 1)))


def tf_reconstruct(W0, X, mask):
    target = (X @ W0.t()).detach()
    Wfp = W0.clone().requires_grad_(True)
    opt = torch.optim.Adam([Wfp], lr=TF_LR)
    best, bestW = float("inf"), None
    for _ in range(TF_STEPS):
        opt.zero_grad()
        with torch.no_grad():
            q = lloyd_masked_ternary(Wfp, mask)
        Wq = Wfp + (q - Wfp).detach()
        loss = (X @ Wq.t() - target).pow(2).mean()
        if loss.item() < best:
            best = loss.item()
            bestW = q.detach().clone()
        loss.backward()
        opt.step()
    return bestW if bestW is not None else lloyd_masked_ternary(W0, mask)


class STELinear(nn.Module):
    def __init__(self, lin, mask):
        super().__init__()
        self.Wfp = nn.Parameter(lin.weight.data.clone())
        self.mask = mask

    def q(self, W=None):
        with torch.no_grad():
            return lloyd_masked_ternary(self.Wfp if W is None else W, self.mask)

    def forward(self, x):
        q = self.q()
        Wq = self.Wfp + (q - self.Wfp).detach()
        return F.linear(x, Wq)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Nạp Qwen3-0.6B FP32")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]

    vi = read_lines(DEV_VI, N_CALIB // 2)
    ja = read_lines(DEV_JA, N_CALIB // 2)
    calib = [x for pr in zip(vi, ja) for x in pr]
    eval_vi = read_lines(DEV_VI, 400)[-N_EVAL:]
    eval_ja = read_lines(DEV_JA, 400)[-N_EVAL:]
    ppl_fp = (eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja))
    log(f"FP32: vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")
    results = {"fp32": {"ppl_vi": ppl_fp[0], "ppl_ja": ppl_fp[1]},
               "config": {"tf_steps": TF_STEPS, "seq_steps": SEQ_STEPS, "n_calib": N_CALIB}}

    # ---- thu activation TF + xnorm ----
    log("Thu activation calibration (60 câu)...")
    acts = {n: [] for n, _ in linears}
    handles = []
    def mk(nm):
        def h(mod, inp):
            acts[nm].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float16))
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=MAX_TOK).input_ids
            if ids.shape[1] < 4:
                continue
            calib_ids.append(ids)
            model(ids)
    for h in handles:
        h.remove()
    X_all, xnorm = {}, {}
    for n in acts:
        X = torch.cat(acts[n], 0)
        X_all[n] = X
        xnorm[n] = X.float().pow(2).mean(0).sqrt()
        acts[n] = None
    orig = {n: m.weight.data.clone() for n, m in linears}
    masks = {n: wanda_nm_mask(orig[n], xnorm[n], 2, 4) for n, _ in linears}
    log(f"Đã dựng mask Wanda 2:4 cho {len(masks)} ma trận. bpw ternary2:4={bpw(0.5,2,4):.2f}, dense={bpw(1.0):.2f}")

    def restore():
        with torch.no_grad():
            for n, m in linears:
                m.weight.data = orig[n].clone()

    def run_tf_arm(tag, use_mask):
        log(f"=== ARM {tag} (teacher-forcing) ===")
        t0 = time.time()
        for i, (n, m) in enumerate(linears):
            X = X_all[n].to(torch.float32)
            m.weight.data = tf_reconstruct(orig[n], X, masks[n] if use_mask else None)
            if (i + 1) % 49 == 0:
                log(f"    {i+1}/196 ({time.time()-t0:.0f}s)")
        pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
        log(f"    => PPL vi {pv:.1f} / ja {pj:.1f}")
        results[tag] = {"ppl_vi": pv, "ppl_ja": pj}
        with io.open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        restore()

    run_tf_arm("arm1b_dense_fixpack_2.08bpw", use_mask=False)
    run_tf_arm("arm1_t24_fixpack_TF_1.94bpw", use_mask=True)

    # ---- ARM 2: SEQUENTIAL block-wise (BRECQ-lite) ----
    log("=== ARM 2: ternary 2:4 fix-pack, SEQUENTIAL block-wise ===")
    # quỹ đạo FP: H_fp[b][s] = hidden vào block b (b=0..28, 28 = sau block cuối)
    layers = model.model.layers
    NB = len(layers)
    ropes, H_fp = [], [[] for _ in range(NB + 1)]
    with torch.no_grad():
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            pos = torch.arange(ids.shape[1])[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                H_fp[b].append(h.clone())
                h = blk(h, position_embeddings=(cos, sin))
            H_fp[NB].append(h.clone())
    NC = len(calib_ids)
    H_q = [H_fp[0][s].clone() for s in range(NC)]
    eval_sub = list(range(0, NC, max(1, NC // 12)))[:12]
    t0 = time.time()
    for b, blk in enumerate(layers):
        stes = []
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            sm = STELinear(lin, masks[key])
            setattr(parent, name, sm)
            stes.append((parent, name, lin, sm))
        params = [sm.Wfp for _, _, _, sm in stes]
        opt = torch.optim.Adam(params, lr=SEQ_LR)
        best, best_state = float("inf"), None

        def eval_block():
            with torch.no_grad():
                v = 0.0
                for s in eval_sub:
                    out = blk(H_q[s], position_embeddings=ropes[s])
                    v += F.mse_loss(out, H_fp[b + 1][s]).item()
            return v

        e0 = eval_block()
        for step in range(SEQ_STEPS):
            idx = random.sample(range(NC), min(SEQ_BATCH, NC))
            opt.zero_grad()
            loss = 0.0
            for s in idx:
                out = blk(H_q[s], position_embeddings=ropes[s])
                loss = loss + F.mse_loss(out, H_fp[b + 1][s])
            (loss / len(idx)).backward()
            opt.step()
            if step % 10 == 9 or step == SEQ_STEPS - 1:
                v = eval_block()
                if v < best:
                    best = v
                    best_state = [sm.Wfp.detach().clone() for _, _, _, sm in stes]
        # bake best
        for (parent, name, lin, sm), Wb in zip(stes, best_state if best_state else [sm.Wfp.detach() for _, _, _, sm in stes]):
            lin.weight.data = sm.q(Wb)
            setattr(parent, name, lin)
        with torch.no_grad():
            for s in range(NC):
                H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"    block {b:2d}/28: mse {e0:.4f} -> {best:.4f}  ({time.time()-t0:.0f}s)")

    pv, pj = eval_ppl(model, tok, eval_vi), eval_ppl(model, tok, eval_ja)
    log(f"    => ARM2 PPL vi {pv:.1f} / ja {pj:.1f}")
    results["arm2_t24_fixpack_SEQ_1.94bpw"] = {"ppl_vi": pv, "ppl_ja": pj}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print("EXP K — FIX-PACK + SEQUENTIAL (kế toán bit trung thực)")
    print("=" * 76)
    print(f"{'arm':40s}{'PPL vi':>12s}{'PPL ja':>12s}")
    for k, v in results.items():
        if k in ("fp32", "config"):
            continue
        print(f"{k:40s}{v['ppl_vi']:12.1f}{v['ppl_ja']:12.1f}")
    print(f"{'fp32':40s}{ppl_fp[0]:12.1f}{ppl_fp[1]:12.1f}")
    print("Mốc exp_i (80 step, mask |W|, scale absmean-all): t2:4 TF = vi 6832.8 / ja 21469.8")
    print("=" * 76)


if __name__ == "__main__":
    main()
