# -*- coding: utf-8 -*-
"""
Huong 1: nen ternary N:M 2:4 DAY DU dan (khong chi down_proj 1/3 nhu exp_ak). Tra loi cau
hoi user: "nen day du co hong khong?". Ablation TANG DAN de co lap phan nao chiu/khong:
  stage 0: baseline (Q4_K_M dequant, chua nen them)
  stage 1: + down_proj  (1/3 expert - da biet tu exp_ak: x1.35)
  stage 2: + gate_up_proj (2/3 con lai cua expert -> TOAN BO expert ternary)
  stage 3: + attention q/k/v/o (gan nhu TOAN model ternary, chi con embed/lm_head/router/norm fp16)
Do PPL + battery da mien MOI stage.

Giu fp16 (khong nen): embed_tokens, lm_head, router (ffn_gate_inp), moi norm (gom q/k-norm).
Day la chuan (embed/lm_head/router luon giu bit cao - xem comment runner 30B, README 0.6B).

Chay: python eval/lowbit_ptq/exp_as_full_quant.py
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
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
N_LAYERS = 16
DTYPE = torch.float16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_az_results.json")

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
    ("math", "Q: What is half of 26?\nA:", "number", 13), ("math", "Q: 3 * 4 + 2 = ?\nA:", "number", 14),
    ("fact", "Q: What is the capital of France?\nA:", "contains", ["paris"]),
    ("fact", "Q: How many days are in one week?\nA:", "contains", ["7", "seven"]),
    ("fact", "Q: What color is the sky on a clear day?\nA:", "contains", ["blue"]),
    ("fact", "Q: Who wrote Romeo and Juliet?\nA:", "contains", ["shakespeare"]),
    ("code", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return", "code_eval", ("add", (2, 3), 5)),
    ("code", "def square(x):\n    \"\"\"Return x squared.\"\"\"\n    return", "code_eval", ("square", (4,), 16)),
    ("code", "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n    return", "code_eval", ("is_even", (4,), True)),
    ("code", "def double(x):\n    \"\"\"Return 2 times x.\"\"\"\n    return", "code_eval", ("double", (5,), 10)),
    ("inst", "List exactly three fruits, one per line:\n1.", "lines3", None),
    ("inst", "Reply with one word only. The opposite of hot is:", "contains", ["cold"]),
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


def ternary_nm_3d(W_f16):
    out = torch.empty_like(W_f16)
    for e in range(W_f16.shape[0]):
        out[e] = ternary_nm_2d(W_f16[e])
    return out


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
        if kind == "lines3":
            items = [ln for ln in ("1." + out.split("\n\n")[0]).split("\n") if ln.strip()]
            return 2 <= len([x for x in items if any(c.isalpha() for c in x)]) <= 4
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
        logits = model(ids).logits[0, :-1].float()
        losses.append(F.cross_entropy(logits, ids[0, 1:]).item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def battery(model, tok):
    torch.manual_seed(0)
    n_pass, per_dom = 0, {}
    for dom, prompt, kind, expect in BATTERY:
        ids = tok(prompt, return_tensors="pt").input_ids
        o = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)[0][ids.shape[1]:]
        ok = _validate(kind, expect, prompt, tok.decode(o, skip_special_tokens=True))
        n_pass += ok
        d = per_dom.setdefault(dom, [0, 0])
        d[0] += ok
        d[1] += 1
    return n_pass, {k: f"{v[0]}/{v[1]}" for k, v in per_dom.items()}


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
    log("build model 16 layer...")
    model = build_model(byname)
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    results = {}

    def measure(tag):
        ppl = compute_ppl(model, tok, PPL_TEXTS)
        npass, dom = battery(model, tok)
        results[tag] = {"ppl": round(ppl, 3), "battery": f"{npass}/{len(BATTERY)}", "per_dom": dom}
        base = results.get("s0_baseline", {}).get("ppl", ppl)
        log(f"  [{tag}] PPL {ppl:.3f} (x{ppl/base:.2f}) | battery {npass}/{len(BATTERY)} | {dom}")

    log("=== stage 0: baseline ===")
    measure("s0_baseline")

    log("=== stage 1: + down_proj ternary (16 layer x 64 expert) ===")
    for L in range(N_LAYERS):
        w = model.model.layers[L].mlp.experts.down_proj
        w.data.copy_(ternary_nm_3d(w.data))
    gc.collect()
    measure("s1_down")

    log("=== stage 2: + gate_up_proj ternary (TOAN BO expert) ===")
    for L in range(N_LAYERS):
        w = model.model.layers[L].mlp.experts.gate_up_proj
        w.data.copy_(ternary_nm_3d(w.data))
        if L % 4 == 0:
            log(f"    gate_up layer {L} xong")
    gc.collect()
    measure("s2_experts_full")

    log("=== stage 3: + attention q/k/v/o ternary (gan TOAN model) ===")
    for L in range(N_LAYERS):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = getattr(model.model.layers[L].self_attn, proj).weight
            w.data.copy_(ternary_nm_2d(w.data))
    gc.collect()
    measure("s3_all")

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== TOM TAT ABLATION TANG DAN ===")
    b = results["s0_baseline"]["ppl"]
    for tag in ["s0_baseline", "s1_down", "s2_experts_full", "s3_all"]:
        r_ = results[tag]
        log(f"  {tag:18s}: PPL {r_['ppl']:8.2f} (x{r_['ppl']/b:.2f}) | battery {r_['battery']}")


if __name__ == "__main__":
    main()
