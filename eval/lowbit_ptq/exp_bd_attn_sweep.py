# -*- coding: utf-8 -*-
"""
Tiep noi exp_bc: hoan thien phan bo bit toan model. Nen exp_bc: down=ternary + gate_up=int3
la diem ngot expert (2.60bpw @ x1.38). O day GIU nen do, QUET attention (q/k/v/o) qua cac
muc bit de tim nguong attention chiu duoc + XAC NHAN bang BATTERY (exp_bc chi co PPL).

Ky vong (lab): attention NHAY hon gate_up -> can bit cao hon. Do de biet chinh xac.
Config cuoi = down=ternary + gate_up=int3 + attention=<diem ngot> = model nen day du PTQ thuan.

Chay: python eval/lowbit_ptq/exp_bd_attn_sweep.py
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
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bd_results.json")
ATTN_CONFIGS = ["fp16", "int8", "int4", "int3", "int2", "ternary"]
ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]

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
BATTERY = [
    ("math", "Q: 7 + 15 = ?\nA:", "number", 22), ("math", "Q: 12 * 3 = ?\nA:", "number", 36),
    ("math", "Q: 100 - 37 = ?\nA:", "number", 63),
    ("math", "Q: A box has 12 apples. Tom takes 5. How many apples are left?\nA:", "number", 7),
    ("fact", "Q: What is the capital of France?\nA:", "contains", ["paris"]),
    ("fact", "Q: How many days are in one week?\nA:", "contains", ["7", "seven"]),
    ("fact", "Q: What color is the sky on a clear day?\nA:", "contains", ["blue"]),
    ("fact", "Q: Who wrote Romeo and Juliet?\nA:", "contains", ["shakespeare"]),
    ("code", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return", "code_eval", ("add", (2, 3), 5)),
    ("code", "def square(x):\n    \"\"\"Return x squared.\"\"\"\n    return", "code_eval", ("square", (4,), 16)),
    ("code", "def double(x):\n    \"\"\"Return 2 times x.\"\"\"\n    return", "code_eval", ("double", (5,), 10)),
    ("inst", "Complete the sequence: one, two, three, four,", "contains", ["five"]),
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


def quant_2d(W_f16, kind):
    if kind == "ternary":
        return ternary_nm_2d(W_f16)
    if kind.startswith("int"):
        return intN_2d(W_f16, int(kind[3:]))
    return W_f16


def quant_3d(W_f16, kind):
    out = torch.empty_like(W_f16)
    for e in range(W_f16.shape[0]):
        out[e] = quant_2d(W_f16[e], kind)
    return out


def attn_bpw(kind):
    return {"fp16": 16.0, "int8": 8.0 + 16.0 / 128, "int4": 4.0 + 16.0 / 128,
            "int3": 3.0 + 16.0 / 128, "int2": 2.0 + 16.0 / 128, "ternary": 1.564}[kind]


def _first_number(txt):
    num = ""
    for ch in txt:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return int(num) if num else None


def _validate(kind, expect, prompt, out):
    low = out.lower()
    try:
        if kind == "number":
            return _first_number(out) == expect
        if kind == "contains":
            return any(e.lower() in low for e in expect)
        if kind == "code_eval":
            fname, fargs, fret = expect
            line = out.split("\n")[0].strip()
            src = prompt + line if prompt.rstrip().endswith(":") else prompt + " " + line
            ns = {}
            exec(compile(src, "<b>", "exec"), {"__builtins__": {}}, ns)   # noqa: S102
            return ns[fname](*fargs) == fret
    except Exception:
        return False
    return False


@torch.no_grad()
def compute_ppl(model, tok, texts):
    losses = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        losses.append(F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def battery(model, tok):
    torch.manual_seed(0)
    n_pass = 0
    for dom, prompt, kind, expect in BATTERY:
        ids = tok(prompt, return_tensors="pt").input_ids
        o = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)[0][ids.shape[1]:]
        n_pass += _validate(kind, expect, prompt, tok.decode(o, skip_special_tokens=True))
    return n_pass


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
        if L % 8 == 0:
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
    base_bat = battery(model, tok)
    log(f"baseline: PPL {base_ppl:.3f} | battery {base_bat}/{len(BATTERY)}")

    # NEN CO DINH: down=ternary + gate_up=int3 (diem ngot exp_bc)
    log("ap nen co dinh: down=ternary + gate_up=int3 cho moi layer...")
    for L in range(N_LAYERS):
        wd = model.model.layers[L].mlp.experts.down_proj
        wd.data.copy_(quant_3d(wd.data, "ternary"))
        wg = model.model.layers[L].mlp.experts.gate_up_proj
        wg.data.copy_(quant_3d(wg.data, "int3"))
    gc.collect()
    ppl_nofq = compute_ppl(model, tok, PPL_TEXTS)
    bat_nofq = battery(model, tok)
    log(f"down=tern+gateup=int3 (attn=fp16): PPL {ppl_nofq:.3f} (x{ppl_nofq/base_ppl:.2f}) | battery {bat_nofq}/{len(BATTERY)}")

    results = {"baseline": {"ppl": round(base_ppl, 3), "battery": f"{base_bat}/{len(BATTERY)}"}}
    for ac in ATTN_CONFIGS:
        log(f"=== attention = {ac} (down=tern, gate_up=int3) ===")
        for L in range(N_LAYERS):
            for proj in ATTN_PROJS:
                w = getattr(model.model.layers[L].self_attn, proj).weight
                orig = dq(byname, f"blk.{L}.attn_{'output' if proj == 'o_proj' else proj[0]}.weight", DTYPE)
                w.data.copy_(quant_2d(orig, ac) if ac != "fp16" else orig)
                del orig
        gc.collect()
        ppl = compute_ppl(model, tok, PPL_TEXTS)
        bat = battery(model, tok)
        results[f"attn_{ac}"] = {"ppl": round(ppl, 3), "x_base": round(ppl / base_ppl, 2),
                                 "battery": f"{bat}/{len(BATTERY)}", "attn_bpw": round(attn_bpw(ac), 2)}
        log(f"  PPL {ppl:.3f} (x{ppl/base_ppl:.2f}) | battery {bat}/{len(BATTERY)} | attn bpw ~{attn_bpw(ac):.2f}")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== FRONTIER attention (nen down=tern + gate_up=int3) ===")
    log(f"  {'attn':10s} {'PPL':>10s} {'xbase':>7s} {'battery':>9s} {'attn_bpw':>9s}")
    for ac in ATTN_CONFIGS:
        r_ = results[f"attn_{ac}"]
        log(f"  {ac:10s} {r_['ppl']:>10.1f} {r_['x_base']:>6.2f}x {r_['battery']:>9s} {r_['attn_bpw']:>8.2f}")


if __name__ == "__main__":
    main()
