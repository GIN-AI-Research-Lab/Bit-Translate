# -*- coding: utf-8 -*-
"""
Buoc quyet dinh: do PPL THAT (khong phai sai so tai tao trong so) tren model DAY DU 16 layer.
SUA sau crash het RAM (may nay thuc te chi 34GB/28.8GB available, KHONG phai 48GB nhu
CLAUDE.md ghi - xem memory project-machine-a-environment): dung fp16 (giam nua bo nho) +
KHONG giu du ban goc tung layer (bug cu: giu ca 16 layer x 536MB = 8.6GB thua) + tach 3 CHE
DO chay RIENG (--mode baseline|aqlm|ternary) qua 3 lan invoke process moi, tranh cong don bo
nho giua cac pha trong CUNG 1 process.

Chay:
  python eval/lowbit_ptq/exp_ak_ppl_real.py --mode baseline
  python eval/lowbit_ptq/exp_ak_ppl_real.py --mode aqlm
  python eval/lowbit_ptq/exp_ak_ppl_real.py --mode ternary
"""
import argparse
import gc
import io
import json
import math
import os
import sys
import time

import gguf
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
N_LAYERS = 16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ak_results.json")
DTYPE = torch.float16

PPL_TEXTS = [
    "The two nations signed a trade agreement on Friday that will gradually eliminate tariffs "
    "on agricultural goods over the next five years, according to officials present at the signing.",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n"
    "        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n"
    "    return -1",
    "Astronomers announced the discovery of a distant exoplanet whose atmosphere appears to "
    "contain water vapor, raising new questions about the conditions required for habitability.",
    "The company's quarterly earnings report exceeded analyst expectations, driven largely by "
    "strong sales of its cloud computing division despite a slowdown in hardware revenue.",
    "After a lengthy renovation, the old train station reopened to the public with a new cafe, "
    "expanded waiting areas, and accessibility ramps that were absent from the original design.",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def dq(byname, name, dtype=torch.float32):
    t = byname[name]
    arr = gguf.quants.dequantize(t.data, t.tensor_type)
    out = torch.from_numpy(arr.copy()).to(dtype)
    del arr
    return out


def kmeans(X, K, iters=25, seed=0):
    X = X.float()
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
    X = X.float()
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


def aqlm_quantize(W_all_f16, M=2, K=64, g=8):
    shp = W_all_f16.shape
    Xg = W_all_f16.reshape(-1, g).float()
    C1 = kmeans(Xg, K, seed=0)
    resid = Xg - C1[torch.cdist(Xg, C1).argmin(1)]
    C2 = kmeans(resid, K, seed=1)
    a1, a2 = beam_assign(Xg, C1, C2, beam=4)
    rec = (C1[a1] + C2[a2]).reshape(shp)
    return rec.to(DTYPE)


def ternary_nm_quantize(W_all_f16, n=2, m=4, sgroup=64):
    out = torch.empty_like(W_all_f16)
    for e in range(W_all_f16.shape[0]):
        W = W_all_f16[e].float()
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
        out[e] = (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C).to(DTYPE)
    return out


@torch.no_grad()
def compute_ppl(model, tok, texts):
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        logits = model(ids).logits[0, :-1].float()
        target = ids[0, 1:]
        loss = F.cross_entropy(logits, target)
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses)), losses


def build_model(byname):
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=N_LAYERS, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5,
                      num_experts_per_tok=8, num_experts=64, norm_topk_prob=False,
                      tie_word_embeddings=False, attention_bias=False)
    log("  tao model rong (meta device - KHONG cap phat random-init day du, tranh giu 2 ban)")
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(DTYPE)
    # QUAN TRONG: to_empty() de lai buffer KHONG hoc duoc (RoPE inv_freq) la rac chua khoi
    # tao - PHAI gan lai dung cong thuc (da doi chieu KHOP TUNG BIT voi model khoi tao thuong
    # truoc khi tin, xem log kiem tra rieng 04/08). Khong lam buoc nay se hong attention AM
    # THAM, khong crash, chi sai so.
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    model.model.rotary_emb.inv_freq.copy_(inv_freq)
    model.model.rotary_emb.original_inv_freq.copy_(inv_freq)
    sd = model.state_dict()          # THAM CHIEU truc tiep toi tensor cua model, khong phai ban rieng

    def put(hf_name, gguf_name):
        sd[hf_name].copy_(dq(byname, gguf_name, DTYPE))     # ghi TRUC TIEP vao cho model, khong tich luy

    put("model.embed_tokens.weight", "token_embd.weight")
    put("model.norm.weight", "output_norm.weight")
    put("lm_head.weight", "output.weight")
    for L in range(N_LAYERS):
        put(f"model.layers.{L}.self_attn.q_proj.weight", f"blk.{L}.attn_q.weight")
        put(f"model.layers.{L}.self_attn.k_proj.weight", f"blk.{L}.attn_k.weight")
        put(f"model.layers.{L}.self_attn.v_proj.weight", f"blk.{L}.attn_v.weight")
        put(f"model.layers.{L}.self_attn.o_proj.weight", f"blk.{L}.attn_output.weight")
        put(f"model.layers.{L}.self_attn.q_norm.weight", f"blk.{L}.attn_q_norm.weight")
        put(f"model.layers.{L}.self_attn.k_norm.weight", f"blk.{L}.attn_k_norm.weight")
        put(f"model.layers.{L}.input_layernorm.weight", f"blk.{L}.attn_norm.weight")
        put(f"model.layers.{L}.post_attention_layernorm.weight", f"blk.{L}.ffn_norm.weight")
        put(f"model.layers.{L}.mlp.gate.weight", f"blk.{L}.ffn_gate_inp.weight")
        gate_all = dq(byname, f"blk.{L}.ffn_gate_exps.weight", DTYPE)
        up_all = dq(byname, f"blk.{L}.ffn_up_exps.weight", DTYPE)
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"].copy_(torch.cat([gate_all, up_all], dim=1))
        del gate_all, up_all
        sd[f"model.layers.{L}.mlp.experts.down_proj"].copy_(dq(byname, f"blk.{L}.ffn_down_exps.weight", DTYPE))
        if L % 4 == 0:
            log(f"  layer {L} xong")
    gc.collect()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "aqlm", "ternary"], required=True)
    args = ap.parse_args()

    log(f"doc GGUF header: {GGUF_PATH}")
    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}

    log(f"dung model {N_LAYERS} layer THAT (dtype={DTYPE}, khong giu du ban goc tung layer)...")
    model = build_model(byname)
    log("load xong.")
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    if args.mode != "baseline":
        log(f"nen '{args.mode}' tung layer (dequant TUOI moi layer tu GGUF, khong giu du)...")
        for L in range(N_LAYERS):
            tL = time.time()
            down_fresh = dq(byname, f"blk.{L}.ffn_down_exps.weight", DTYPE)   # tuoi, chi 1 layer
            if args.mode == "aqlm":
                rec = aqlm_quantize(down_fresh, M=2, K=64, g=8)
            else:
                rec = ternary_nm_quantize(down_fresh, n=2, m=4)
            model.model.layers[L].mlp.experts.down_proj.data.copy_(rec)
            del down_fresh, rec
            log(f"  layer {L} nen xong ({time.time() - tL:.0f}s)")
        gc.collect()

    if args.mode == "baseline":
        log("sanity check: generate 20 token...")
        ids = tok("The capital of France is", return_tensors="pt").input_ids
        out = model.generate(ids, max_new_tokens=20, do_sample=False)
        log(f"  '{tok.decode(out[0], skip_special_tokens=True)}'")

    ppl, losses = compute_ppl(model, tok, PPL_TEXTS)
    log(f"PPL ({args.mode}) = {ppl:.3f} | per-cau: {[round(x, 3) for x in losses]}")

    old = {}
    if os.path.exists(OUT_JSON):
        try:
            old = json.load(io.open(OUT_JSON, encoding="utf-8"))
        except Exception:
            old = {}
    old[args.mode] = {"ppl": round(ppl, 4), "losses": [round(x, 4) for x in losses]}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON} (key={args.mode})")


if __name__ == "__main__":
    main()
