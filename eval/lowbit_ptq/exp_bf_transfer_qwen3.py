# -*- coding: utf-8 -*-
"""
Kiem chung CHUYEN GIAO: thu hang do nhay (down<gate_up<attention<lm_head) tim tren OLMoE-1B-7B
(MoE, MHA, untied) co giu tren Qwen3-0.6B (DENSE, GQA 16q/8kv, TIED embedding) khong? Kien
truc khac han -> neu ranking giu -> chuyen giao that (bat nguon tu kien truc, khong phai model
cu the). Claim da noi voi user: "ranking chuyen giao, chi do lai NGUONG".

2 phan:
  A. RANKING PROBE: moi thanh phan ternary RIENG LE (con lai fp16) -> PPL. Cai nao PPL cao
     nhat = nhay nhat. So thu tu voi OLMoE (down ben nhat, attention/lm_head nhay nhat).
  B. TRANSFER TRUC TIEP: ap config thang cua OLMoE (down=ternary + gate/up=int3 + attn=int4
     + emb/head=int4) len Qwen3-0.6B -> PPL + battery. Co ra ~x1.5 nhu OLMoE khong?

Reload model moi cau hinh (0.6B nho, ~10s) de khong ro ri trang thai.
Chay: python eval/lowbit_ptq/exp_bf_transfer_qwen3.py
"""
import gc
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = ("F:/Project Ai/Quantazation/hf_cache/hub/models--Qwen--Qwen3-0.6B/"
             "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bf_results.json")

PPL_TEXTS = [
    "The two nations signed a trade agreement that will gradually eliminate tariffs over five years.",
    "def binary_search(a, t):\n    lo, hi = 0, len(a)-1\n    while lo <= hi:\n        m=(lo+hi)//2\n        if a[m]==t: return m\n        elif a[m]<t: lo=m+1\n        else: hi=m-1\n    return -1",
    "Astronomers announced the discovery of a distant exoplanet whose atmosphere contains water vapor.",
    "Chính phủ công bố kế hoạch đầu tư lớn cho hệ thống giao thông công cộng trong thập kỷ tới.",
    "この新しい技術は医療分野での幅広い応用が期待されており、多くの研究が進行中です。",
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


def ternary_2d(W, sgroup=64):
    W = W.float()
    R, C = W.shape
    Wg = W.view(R, -1, 4)
    idx = Wg.abs().topk(2, dim=2).indices
    mask = torch.zeros_like(Wg).scatter_(2, idx, 1.0).view(R, C)
    Wv, Mv = W.view(R, -1, sgroup), mask.view(R, -1, sgroup)
    Wm = Wv * Mv
    cnt = Mv.sum(2, keepdim=True).clamp(min=1)
    s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-8)
    for _ in range(3):
        t = torch.round(Wm / s).clamp(-1, 1) * Mv
        num, den = (Wm * t).sum(2, keepdim=True), (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-8)
    return (torch.round(Wm / s.clamp(min=1e-8)).clamp(-1, 1) * s * Mv).view(R, C)


def intN_2d(W, nbits, g=128):
    W = W.float()
    R, C = W.shape
    if C % g:
        g = 64 if C % 64 == 0 else 32
    Wg = W.view(R, -1, g)
    qmax = 2 ** (nbits - 1) - 1
    scale = (Wg.abs().amax(2, keepdim=True) / max(qmax, 1)).clamp(min=1e-8)
    q = torch.round(Wg / scale).clamp(-qmax - 1, qmax)
    return (q * scale).view(R, C)


def quant(W, kind):
    if kind in (None, "fp16"):
        return W
    if kind == "ternary":
        return ternary_2d(W).to(W.dtype)
    return intN_2d(W, int(kind[3:])).to(W.dtype)


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
def compute_ppl(model, tok):
    losses = []
    for t in PPL_TEXTS:
        ids = tok(t, return_tensors="pt").input_ids
        losses.append(F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item())
    return math.exp(sum(losses) / len(losses))


@torch.no_grad()
def battery(model, tok):
    torch.manual_seed(0)
    n = 0
    for dom, prompt, kind, expect in BATTERY:
        ids = tok(prompt, return_tensors="pt").input_ids
        o = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)[0][ids.shape[1]:]
        n += _validate(kind, expect, prompt, tok.decode(o, skip_special_tokens=True))
    return n


@torch.no_grad()
def load_and_apply(cfg):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    for blk in model.model.layers:
        if cfg.get("down"):
            blk.mlp.down_proj.weight.data.copy_(quant(blk.mlp.down_proj.weight.data, cfg["down"]))
        if cfg.get("gateup"):
            blk.mlp.gate_proj.weight.data.copy_(quant(blk.mlp.gate_proj.weight.data, cfg["gateup"]))
            blk.mlp.up_proj.weight.data.copy_(quant(blk.mlp.up_proj.weight.data, cfg["gateup"]))
        if cfg.get("attn"):
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                w = getattr(blk.self_attn, p).weight
                w.data.copy_(quant(w.data, cfg["attn"]))
    if cfg.get("embhead"):     # tied: quantize embed_tokens (lm_head shared)
        model.model.embed_tokens.weight.data.copy_(quant(model.model.embed_tokens.weight.data, cfg["embhead"]))
    return model


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    log("=== baseline FP ===")
    m = load_and_apply({})
    base = compute_ppl(m, tok)
    base_bat = battery(m, tok)
    log(f"  baseline PPL {base:.3f} | battery {base_bat}/{len(BATTERY)}")
    del m
    gc.collect()

    results = {"baseline": {"ppl": round(base, 3), "battery": f"{base_bat}/{len(BATTERY)}"}}

    log("\n=== A. RANKING PROBE (moi thanh phan ternary RIENG LE) ===")
    for comp in ["down", "gateup", "attn"]:
        m = load_and_apply({comp: "ternary"})
        ppl = compute_ppl(m, tok)
        results[f"probe_{comp}_ternary"] = {"ppl": round(ppl, 3), "x_base": round(ppl / base, 2)}
        log(f"  chi {comp}=ternary: PPL {ppl:.1f} (x{ppl/base:.1f})")
        del m
        gc.collect()

    log("\n=== B. TRANSFER config thang cua OLMoE len Qwen3-0.6B ===")
    cfg_win = {"down": "ternary", "gateup": "int3", "attn": "int4", "embhead": "int4"}
    m = load_and_apply(cfg_win)
    ppl = compute_ppl(m, tok)
    bat = battery(m, tok)
    results["transfer_olmoe_config"] = {"config": cfg_win, "ppl": round(ppl, 3),
                                        "x_base": round(ppl / base, 2), "battery": f"{bat}/{len(BATTERY)}"}
    log(f"  down=tern+gate/up=int3+attn=int4+emb/head=int4: PPL {ppl:.1f} (x{ppl/base:.2f}) | battery {bat}/{len(BATTERY)}")
    del m
    gc.collect()

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== KET LUAN CHUYEN GIAO ===")
    rk = {c: results[f"probe_{c}_ternary"]["x_base"] for c in ["down", "gateup", "attn"]}
    order = sorted(rk, key=rk.get)
    log(f"  Thu hang do nhay Qwen3 (ternary rieng le, x_base): "
        + " < ".join(f"{c}({rk[c]}x)" for c in order))
    log(f"  OLMoE thu hang: down < gate_up < attention (down ben nhat).")
    match = order[0] == "down"
    log(f"  down co la BEN NHAT tren Qwen3? {'CO' if match else 'KHONG'} "
        f"-> ranking {'chuyen giao' if match else 'KHAC - can xem lai'}")
    tr = results["transfer_olmoe_config"]["x_base"]
    log(f"  Transfer config OLMoE truc tiep: x{tr} (OLMoE goc x1.53). "
        f"{'Gan tuong duong -> transfer duoc' if tr < 3 else 'Lech nhieu -> can do lai nguong rieng Qwen3'}")


if __name__ == "__main__":
    main()
