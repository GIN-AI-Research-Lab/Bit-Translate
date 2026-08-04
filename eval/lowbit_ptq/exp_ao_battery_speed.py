# -*- coding: utf-8 -*-
"""
Theo yeu cau user: (1) do tok/s baseline vs ternary (CHUNG MINH khong khac biet vi chua dong
goi packed-format, chi thay GIA TRI trong so, van luu fp16 day dac - toc do thuc can kernel
rieng, xem RESEARCH_MOE_SPEED_PTQ.md); (2) battery da mien (toan/kien thuc/code/lam-theo-chi-
dan, validator tu dong - tai dung dung BATTERY cua exp_ab_subbit_curriculum.py) tren model
DAY DU 16 layer, baseline vs ternary N:M 2:4 (khong chay lai AQLM - da biet sup PPL x703,
chay lai ton ~57 phut cho ket qua da ro).

Chay:
  python eval/lowbit_ptq/exp_ao_battery_speed.py --mode baseline
  python eval/lowbit_ptq/exp_ao_battery_speed.py --mode ternary
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
DTYPE = torch.float16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_ao_results.json")

# BATTERY: copy tu exp_ab_subbit_curriculum.py (khong import - tranh side-effect top-level
# cua cac file script khac, dung pattern da thiet lap ca session).
BATTERY = [
    ("math", "Q: 7 + 15 = ?\nA:", "number", 22),
    ("math", "Q: 12 * 3 = ?\nA:", "number", 36),
    ("math", "Q: 100 - 37 = ?\nA:", "number", 63),
    ("math", "Q: A box has 12 apples. Tom takes 5. How many apples are left?\nA:", "number", 7),
    ("math", "Q: What is half of 26?\nA:", "number", 13),
    ("math", "Q: 3 * 4 + 2 = ?\nA:", "number", 14),
    ("fact", "Q: What is the capital of France?\nA:", "contains", ["paris"]),
    ("fact", "Q: How many days are in one week?\nA:", "contains", ["7", "seven"]),
    ("fact", "Q: What color is the sky on a clear day?\nA:", "contains", ["blue"]),
    ("fact", "Q: Who wrote Romeo and Juliet?\nA:", "contains", ["shakespeare"]),
    ("code", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return",
     "code_eval", ("add", (2, 3), 5)),
    ("code", "def square(x):\n    \"\"\"Return x squared.\"\"\"\n    return",
     "code_eval", ("square", (4,), 16)),
    ("code", "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n    return",
     "code_eval", ("is_even", (4,), True)),
    ("code", "def double(x):\n    \"\"\"Return 2 times x.\"\"\"\n    return",
     "code_eval", ("double", (5,), 10)),
    ("inst", "List exactly three fruits, one per line:\n1.", "lines3", None),
    ("inst", "Reply with one word only. The opposite of hot is:", "contains", ["cold"]),
    ("inst", "Complete the sequence: one, two, three, four,", "contains", ["five"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
            body = out.split("\n\n")[0]
            items = [ln for ln in ("1." + body).split("\n") if ln.strip()]
            return 2 <= len([x for x in items if any(c.isalpha() for c in x)]) <= 4
        if kind == "code_eval":
            fname, fargs, fret = expect
            line = out.split("\n")[0].strip()
            src = prompt + " " + line if not prompt.rstrip().endswith(":") else prompt + line
            ns = {}
            exec(compile(src, "<bat>", "exec"), {"__builtins__": {}}, ns)   # noqa: S102
            return ns[fname](*fargs) == fret
    except Exception:
        return False
    return False


def dq(byname, name, dtype=torch.float32):
    t = byname[name]
    arr = gguf.quants.dequantize(t.data, t.tensor_type)
    out = torch.from_numpy(arr.copy()).to(dtype)
    del arr
    return out


def to_f8(s):
    sign = torch.sign(s)
    a = s.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a))
    m = a / (2 ** e)
    return sign * (2 ** e) * (torch.round(m * 8) / 8)


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


def build_model(byname):
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=N_LAYERS, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5,
                      num_experts_per_tok=8, num_experts=64, norm_topk_prob=False,
                      tie_word_embeddings=False, attention_bias=False)
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(DTYPE)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    model.model.rotary_emb.inv_freq.copy_(inv_freq)
    model.model.rotary_emb.original_inv_freq.copy_(inv_freq)
    sd = model.state_dict()

    def put(hf_name, gguf_name):
        sd[hf_name].copy_(dq(byname, gguf_name, DTYPE))

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


@torch.no_grad()
def run_battery(model, tok, temps=(0.0,), max_new=24):
    res = {}
    for temp in temps:
        torch.manual_seed(0)
        n_pass, per_dom, transcripts = 0, {}, []
        for dom, prompt, kind, expect in BATTERY:
            ids = tok(prompt, return_tensors="pt").input_ids
            kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            if temp > 0:
                kw.update(do_sample=True, temperature=temp, top_p=0.9)
            else:
                kw.update(do_sample=False)
            o = model.generate(ids, **kw)[0][ids.shape[1]:]
            txt = tok.decode(o, skip_special_tokens=True)
            ok = _validate(kind, expect, prompt, txt)
            n_pass += ok
            d = per_dom.setdefault(dom, [0, 0])
            d[0] += ok
            d[1] += 1
            transcripts.append({"dom": dom, "ok": bool(ok), "prompt": prompt[:50], "out": txt[:80]})
        res[f"t{temp}"] = {"pass_rate": round(n_pass / len(BATTERY), 3),
                           "per_dom": {k: f"{v[0]}/{v[1]}" for k, v in per_dom.items()},
                           "transcripts": transcripts}
        log(f"  battery t={temp}: {n_pass}/{len(BATTERY)} | {res[f't{temp}']['per_dom']}")
        for t in transcripts:
            log(f"    [{t['dom']}] {'OK ' if t['ok'] else 'SAI'} '{t['prompt']}' -> '{t['out']}'")
    return res


@torch.no_grad()
def measure_tokps(model, tok, n_tokens=40, n_trials=3):
    ids = tok("The history of the internet began with", return_tensors="pt").input_ids
    times = []
    for _ in range(n_trials):
        t0 = time.time()
        model.generate(ids, max_new_tokens=n_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        times.append(time.time() - t0)
    best = min(times)
    return n_tokens / best, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "ternary"], required=True)
    args = ap.parse_args()

    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}
    log(f"dung model ({args.mode})...")
    model = build_model(byname)
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    if args.mode == "ternary":
        log("nen ternary 2:4 tung layer...")
        for L in range(N_LAYERS):
            down_fresh = dq(byname, f"blk.{L}.ffn_down_exps.weight", DTYPE)
            rec = ternary_nm_quantize(down_fresh, n=2, m=4)
            model.model.layers[L].mlp.experts.down_proj.data.copy_(rec)
            del down_fresh, rec
        gc.collect()

    log("do tok/s (5 luot warm-up + do, lay best cua 3 lan)...")
    tokps, times = measure_tokps(model, tok)
    log(f"  tok/s = {tokps:.2f} (thoi gian 3 lan: {[round(t,2) for t in times]})")

    log("chay battery da mien...")
    battery = run_battery(model, tok)

    old = {}
    if os.path.exists(OUT_JSON):
        try:
            old = json.load(io.open(OUT_JSON, encoding="utf-8"))
        except Exception:
            old = {}
    old[args.mode] = {"tokps": round(tokps, 2), "battery": battery}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON} (key={args.mode})")


if __name__ == "__main__":
    main()
