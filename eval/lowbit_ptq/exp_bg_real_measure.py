# -*- coding: utf-8 -*-
"""
Do THUC TE cho config nen OLMoE (down=ternary + gate_up=int3 + attn=int4 + emb/head=int4):
  1. Do chinh xac THAT: sinh van ban mau (nhin mat) + battery.
  2. tok/s: do THAT (nhung LUU Y: van fp16 day dac -> giong ban goc, CHUA co kernel doc packed).
  3. RAM su dung: process RSS THAT (cung LUU Y nhu tren - la fp16 size, khong phai size nen).
  4. Dung luong file sau nen: TINH CHINH XAC tu bit-width + scale (bit-packing xac dinh).

Trung thuc: (1)(4) la so THAT ngay. (2)(3) chi la trang thai HIEN TAI (chua packed) - loi ich
toc do/RAM that CAN kernel doc packed-format (viec engineering lon, chua lam - RESEARCH_MOE_SPEED_PTQ).

Chay: python eval/lowbit_ptq/exp_bg_real_measure.py
"""
import gc
import io
import json
import math
import os
import sys
import time

import gguf
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GGUF_PATH = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q4_k_m.gguf"
N_LAYERS = 16
DTYPE = torch.float16
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bg_results.json")

GEN_PROMPTS = [
    "The most important invention of the 20th century was",
    "Question: Why is the sky blue?\nAnswer:",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Vietnam is a country in Southeast Asia known for",
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

# so tham so tung khoi
P_DOWN, P_GATEUP, P_ATTN, P_EMBHEAD, P_MISC = (64*2048*1024*16, 64*2048*2048*16,
                                               4*2048*2048*16, 2*50304*2048, 64*2048*16)
CFG_BPW = {"down": 1.564, "gateup": 3.125, "attn": 4.125, "embhead": 4.125, "misc": 16.0}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def dq(byname, name, dtype=torch.float32):
    t = byname[name]
    return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).to(dtype)


def ternary_2d(W, sgroup=64):
    W = W.float(); R, C = W.shape
    Wg = W.view(R, -1, 4); idx = Wg.abs().topk(2, dim=2).indices
    mask = torch.zeros_like(Wg).scatter_(2, idx, 1.0).view(R, C)
    Wv, Mv = W.view(R, -1, sgroup), mask.view(R, -1, sgroup); Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1); s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C).to(DTYPE)


def intN_2d(W, nbits, g=128):
    W = W.float(); R, C = W.shape; Wg = W.view(R, -1, g)
    qmax = 2 ** (nbits - 1) - 1
    scale = (Wg.abs().amax(2, keepdim=True) / max(qmax, 1)).clamp(min=1e-8)
    q = torch.round(Wg / scale).clamp(-qmax - 1, qmax)
    return (q * scale).view(R, C).to(DTYPE)


def q3(W, kind):
    out = torch.empty_like(W)
    for e in range(W.shape[0]):
        out[e] = ternary_2d(W[e]) if kind == "ternary" else intN_2d(W[e], int(kind[3:]))
    return out


def _first_number(txt):
    num = ""
    for ch in txt:
        if ch.isdigit(): num += ch
        elif num: break
    return int(num) if num else None


def _validate(kind, expect, prompt, out):
    low = out.lower()
    try:
        if kind == "number": return _first_number(out) == expect
        if kind == "contains": return any(e.lower() in low for e in expect)
        if kind == "code_eval":
            fname, fargs, fret = expect; line = out.split("\n")[0].strip()
            src = prompt + line if prompt.rstrip().endswith(":") else prompt + " " + line
            ns = {}; exec(compile(src, "<b>", "exec"), {"__builtins__": {}}, ns)   # noqa: S102
            return ns[fname](*fargs) == fret
    except Exception:
        return False
    return False


@torch.no_grad()
def battery(model, tok):
    torch.manual_seed(0); n = 0
    for dom, prompt, kind, expect in BATTERY:
        ids = tok(prompt, return_tensors="pt").input_ids
        o = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)[0][ids.shape[1]:]
        n += _validate(kind, expect, prompt, tok.decode(o, skip_special_tokens=True))
    return n


@torch.no_grad()
def build_and_compress(byname):
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=N_LAYERS, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5, num_experts_per_tok=8,
                      num_experts=64, norm_topk_prob=False, tie_word_embeddings=False, attention_bias=False)
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(DTYPE)
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    model.model.rotary_emb.inv_freq.copy_(inv); model.model.rotary_emb.original_inv_freq.copy_(inv)
    sd = model.state_dict()

    def put(h, g): sd[h].copy_(dq(byname, g, DTYPE))

    # embed/head = int4
    e0 = dq(byname, "token_embd.weight", DTYPE); sd["model.embed_tokens.weight"].copy_(intN_2d(e0, 4))
    h0 = dq(byname, "output.weight", DTYPE); sd["lm_head.weight"].copy_(intN_2d(h0, 4))
    del e0, h0
    sd["model.norm.weight"].copy_(dq(byname, "output_norm.weight", DTYPE))
    for L in range(N_LAYERS):
        for hn, gn in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                       ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_output")]:
            sd[f"model.layers.{L}.{hn}.weight"].copy_(intN_2d(dq(byname, f"blk.{L}.{gn}.weight", DTYPE), 4))  # attn=int4
        for hn, gn in [("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"),
                       ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                       ("mlp.gate", "ffn_gate_inp")]:
            put(f"model.layers.{L}.{hn}.weight", f"blk.{L}.{gn}.weight")
        g_ = dq(byname, f"blk.{L}.ffn_gate_exps.weight", DTYPE); u_ = dq(byname, f"blk.{L}.ffn_up_exps.weight", DTYPE)
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"].copy_(q3(torch.cat([g_, u_], dim=1), "int3"))  # gate_up=int3
        del g_, u_
        sd[f"model.layers.{L}.mlp.experts.down_proj"].copy_(q3(dq(byname, f"blk.{L}.ffn_down_exps.weight", DTYPE), "ternary"))  # down=ternary
        if L % 8 == 0:
            log(f"  layer {L} nen xong")
    gc.collect()
    model.eval()
    return model


def compressed_file_gb():
    bits = (P_DOWN * CFG_BPW["down"] + P_GATEUP * CFG_BPW["gateup"] + P_ATTN * CFG_BPW["attn"]
            + P_EMBHEAD * CFG_BPW["embhead"] + P_MISC * CFG_BPW["misc"])
    return bits / 8 / 1e9


def main():
    r = gguf.GGUFReader(GGUF_PATH)
    byname = {t.name: t for t in r.tensors}
    rss0 = rss_gb()
    log(f"RSS truoc load: {rss0:.2f}GB")
    log("build + nen (down=ternary + gate_up=int3 + attn=int4 + emb/head=int4)...")
    model = build_and_compress(byname)
    rss1 = rss_gb()
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    log(f"RSS sau load model (fp16 day dac): {rss1:.2f}GB")

    # 1. sinh van ban mau
    log("\n=== 1. SINH VAN BAN MAU (greedy, chat luong THAT) ===")
    samples = []
    for p in GEN_PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        o = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)[0][ids.shape[1]:]
        txt = tok.decode(o, skip_special_tokens=True)
        samples.append({"prompt": p[:45], "out": txt[:150]})
        log(f"  '{p[:40]}...' -> '{txt[:110]}'")

    # battery
    bat = battery(model, tok)
    log(f"\n=== battery: {bat}/{len(BATTERY)} ===")

    # 2. tok/s THAT (fp16 day dac - CHUA packed)
    log("\n=== 2. tok/s (fp16 day dac, CHUA packed) ===")
    ids = tok("The history of science began when", return_tensors="pt").input_ids
    times = []
    for _ in range(3):
        t0 = time.time()
        model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        times.append(time.time() - t0)
    tokps = 48 / min(times)
    log(f"  tok/s = {tokps:.2f} (LUU Y: giong ban goc, chua co kernel doc packed)")

    # 3. RAM
    rss2 = rss_gb()
    log(f"\n=== 3. RAM su dung (RSS): {rss2:.2f}GB (fp16 day dac - CHUA packed) ===")

    # 4. dung luong file sau nen (tinh chinh xac)
    comp_gb = compressed_file_gb()
    fp16_gb = (P_DOWN + P_GATEUP + P_ATTN + P_EMBHEAD + P_MISC) * 16 / 8 / 1e9
    log(f"\n=== 4. DUNG LUONG FILE SAU NEN (tinh chinh xac tu bit-width) ===")
    log(f"  fp16 goc:  {fp16_gb:.2f}GB")
    log(f"  sau nen:   {comp_gb:.2f}GB  ({fp16_gb/comp_gb:.1f}x nho hon)")
    log(f"  (down=ternary 1.56 + gate_up=int3 3.12 + attn/emb/head=int4 4.12 + misc fp16)")

    out = {"accuracy": {"battery": f"{bat}/{len(BATTERY)}", "note": "PPL x1.53 tu exp_be",
                        "samples": samples},
           "tokps_current_unpacked": round(tokps, 2),
           "ram_rss_gb_unpacked": round(rss2, 2),
           "file_size": {"fp16_gb": round(fp16_gb, 2), "compressed_gb": round(comp_gb, 2),
                         "ratio": round(fp16_gb / comp_gb, 1)},
           "HONEST_NOTE": "accuracy + file_size la THAT ngay; tok/s + RAM la trang thai HIEN "
                          "TAI (fp16 day dac, chua packed) - loi ich toc do/RAM that CAN kernel "
                          "doc packed-format (chua lam)."}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\nda ghi {OUT_JSON}")

    log("\n=== TOM TAT TRUNG THUC ===")
    log(f"  [THAT ngay] Do chinh xac: battery {bat}/12, PPL x1.53 | File nen: {comp_gb:.2f}GB "
        f"({fp16_gb/comp_gb:.1f}x nho hon fp16 {fp16_gb:.1f}GB)")
    log(f"  [CHUA that] tok/s {tokps:.1f} + RAM {rss2:.1f}GB = trang thai fp16-day-dac, KHONG "
        f"phai loi ich nen - can kernel packed (RESEARCH_MOE_SPEED_PTQ) moi hien loi ich toc do/RAM.")


if __name__ == "__main__":
    main()
