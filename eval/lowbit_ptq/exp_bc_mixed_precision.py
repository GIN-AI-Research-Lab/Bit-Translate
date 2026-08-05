# -*- coding: utf-8 -*-
"""
Huong 1 ban thong minh: phan bo bit THEO DO NHAY (dung phat hien exp_as - down chiu ternary,
gate_up vo o ternary). Thay vi dong deu, giu down=ternary 2:4 (re, da chung minh ×1.35) va
QUET gate_up qua fp16/int8/int4/int3/int2/ternary de tim NGUONG gate_up chiu duoc bao nhieu
bit. Cau tra loi: diem ngot bpw/chat luong cho model low-bit thuc te.

Attention giu fp16 (co lap gate_up). down=ternary co dinh. Naive (khong recon) - so cong bang
voi cac config truoc.

Chay: python eval/lowbit_ptq/exp_av_mixed_precision.py
"""
import gc
import io
import json
import math
import os
import sys
import time

import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
N_LAYERS = 16
DTYPE = torch.float16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bc_results.json")

# down luon ternary; quet gate_up qua cac muc nay
GATE_UP_CONFIGS = ["fp16", "int8", "int4", "int3", "int2", "ternary"]

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
    return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).to(dtype)


def ternary_nm_2d(W_f16, n=2, m=4, sgroup=64):
    W = W_f16.float()
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
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C).to(DTYPE)


def intN_2d(W_f16, nbits, g=128):
    W = W_f16.float()
    R, C = W.shape
    Wg = W.view(R, -1, g)
    qmax = 2 ** (nbits - 1) - 1
    scale = (Wg.abs().amax(2, keepdim=True) / max(qmax, 1)).clamp(min=1e-8)
    q = torch.round(Wg / scale).clamp(-qmax - 1, qmax)
    return (q * scale).view(R, C).to(DTYPE)


def quant_3d(W_f16, kind):
    out = torch.empty_like(W_f16)
    for e in range(W_f16.shape[0]):
        if kind == "ternary":
            out[e] = ternary_nm_2d(W_f16[e])
        elif kind.startswith("int"):
            out[e] = intN_2d(W_f16[e], int(kind[3:]))
    return out


def eff_bpw(gate_up_kind):
    """bpw hieu dung cua khoi EXPERT (down 1/3 = ternary 1.56, gate_up 2/3 = kind)."""
    down_bpw = 1.564
    gk = {"fp16": 16.0, "int8": 8.0 + 16.0 / 128, "int4": 4.0 + 16.0 / 128,
          "int3": 3.0 + 16.0 / 128, "int2": 2.0 + 16.0 / 128, "ternary": 1.564}[gate_up_kind]
    return (1 * down_bpw + 2 * gk) / 3


@torch.no_grad()
def compute_ppl(model, tok, texts):
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        losses.append(F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def build_model(byname):
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=N_LAYERS, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5, num_experts_per_tok=8,
                      num_experts=64, norm_topk_prob=False, tie_word_embeddings=False, attention_bias=False)
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(DTYPE)
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    model.model.rotary_emb.inv_freq.copy_(inv)
    model.model.rotary_emb.original_inv_freq.copy_(inv)
    sd = model.state_dict()

    def put(h, g):
        sd[h].copy_(dq(byname, g, DTYPE))

    put("model.embed_tokens.weight", "token_embd.weight")
    put("model.norm.weight", "output_norm.weight")
    put("lm_head.weight", "output.weight")
    for L in range(N_LAYERS):
        for hn, gn in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                       ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output"),
                       ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
                       ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                       ("mlp.gate", "ffn_gate_inp")]:
            put(f"model.layers.{L}.{hn}.weight", f"blk.{L}.{gn}.weight")
        g_ = dq(byname, f"blk.{L}.ffn_gate_exps.weight", DTYPE)
        u_ = dq(byname, f"blk.{L}.ffn_up_exps.weight", DTYPE)
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"].copy_(torch.cat([g_, u_], dim=1))
        del g_, u_
        sd[f"model.layers.{L}.mlp.experts.down_proj"].copy_(dq(byname, f"blk.{L}.ffn_down_exps.weight", DTYPE))
        if L % 4 == 0:
            log(f"  layer {L} load xong")
    gc.collect()
    model.eval()
    return model


def main():
    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}
    log("build model...")
    model = build_model(byname)
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    base_ppl = compute_ppl(model, tok, PPL_TEXTS)
    log(f"baseline (Q4_K_M dequant): PPL {base_ppl:.3f}")

    # down = ternary CO DINH (ap 1 lan, giu suot)
    log("ap down_proj = ternary 2:4 (co dinh) cho moi layer...")
    for L in range(N_LAYERS):
        w = model.model.layers[L].mlp.experts.down_proj
        w.data.copy_(quant_3d(w.data, "ternary"))
    gc.collect()

    results = {"baseline_q4km": round(base_ppl, 3)}
    for gu in GATE_UP_CONFIGS:
        # re-dequant gate_up TUOI moi config (down giu ternary)
        log(f"=== gate_up = {gu} (down=ternary co dinh) ===")
        for L in range(N_LAYERS):
            gate = dq(byname, f"blk.{L}.ffn_gate_exps.weight", DTYPE)
            up = dq(byname, f"blk.{L}.ffn_up_exps.weight", DTYPE)
            gu_w = torch.cat([gate, up], dim=1)
            del gate, up
            if gu != "fp16":
                gu_w = quant_3d(gu_w, gu)
            model.model.layers[L].mlp.experts.gate_up_proj.data.copy_(gu_w)
            del gu_w
        gc.collect()
        ppl = compute_ppl(model, tok, PPL_TEXTS)
        bpw = eff_bpw(gu)
        results[f"down_tern_gateup_{gu}"] = {"ppl": round(ppl, 3), "x_base": round(ppl / base_ppl, 2),
                                             "expert_bpw": round(bpw, 2)}
        log(f"  PPL {ppl:.3f} (x{ppl/base_ppl:.2f}) | expert bpw ~{bpw:.2f}")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== FRONTIER: down=ternary co dinh, quet gate_up ===")
    log(f"  {'gate_up':10s} {'PPL':>10s} {'xbase':>7s} {'expert_bpw':>11s}")
    for gu in GATE_UP_CONFIGS:
        r_ = results[f"down_tern_gateup_{gu}"]
        log(f"  {gu:10s} {r_['ppl']:>10.1f} {r_['x_base']:>6.2f}x {r_['expert_bpw']:>10.2f}")
    log("\n  Diem ngot = bpw thap nhat ma xbase con chap nhan duoc (vd <2x).")


if __name__ == "__main__":
    main()
