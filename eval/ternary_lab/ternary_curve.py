# -*- coding: utf-8 -*-
"""
THI NGHIEM: ternary hoa (BitNet-style, data-free) Qwen2.5-0.5B-Instruct FP32.
Do duong cong suy giam A1 (TWN per-tensor) -> A2 (TWN per-row) -> A3 (per-row + alpha least-squares).
Chi ap len 7 loai Linear trong block (q,k,v,o,gate,up,down). Khong dung embed/lm_head/norm/bias.
Ket qua ghi incremental vao results.json.
"""
import os
os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

import json
import math
import sys
import time

import torch

torch.set_num_threads(5)
torch.manual_seed(0)

LAB = "D:/Bit-Translate-data/ternary_lab"
RESULTS_PATH = os.path.join(LAB, "results.json")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEV_JA = "D:/Bit-Translate-data/clean_v6/dev.ja"
DEV_VI = "D:/Bit-Translate-data/clean_v6/dev.vi"
N_SENT = 60
MIN_TOKENS = 5

JA_TEST = [
    "来週の会議は資料が間に合わないので、日程を変更したいと思います。",
    "今日は天気がいいですね。",
    "彼は毎朝コーヒーを飲みながら新聞を読みます。",
]

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

results = {"model": MODEL_ID, "variants": {}}


def save_results():
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ load model
log("Loading tokenizer + model (FP32)...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
tied = (model.lm_head.weight is model.model.embed_tokens.weight) or (
    model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
)
log(f"Total params: {n_params:,}  | lm_head tied to embed_tokens: {tied}")

# collect target linears
target_linears = {}  # name -> module
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear) and ".layers." in name and name.split(".")[-1] in TARGET_SUFFIXES:
        target_linears[name] = mod

suffix_counts = {}
for name in target_linears:
    sfx = name.split(".")[-1]
    suffix_counts[sfx] = suffix_counts.get(sfx, 0) + 1
target_param_count = sum(m.weight.numel() for m in target_linears.values())
log(f"Target Linear modules: {len(target_linears)} ({suffix_counts}); weight params: {target_param_count:,}")

layer0 = [n for n in target_linears if ".layers.0." in n]
log(f"Layer 0 linears: {sorted(layer0)}")

results["meta"] = {
    "n_params": n_params,
    "lm_head_tied": bool(tied),
    "n_target_linears": len(target_linears),
    "suffix_counts": suffix_counts,
    "target_weight_params": target_param_count,
    "torch_threads": torch.get_num_threads(),
}
save_results()

# ------------------------------------------------------------------ eval data
def load_sents(path, n):
    sents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
            if len(sents) >= n:
                break
    return sents

ja_sents_raw = load_sents(DEV_JA, N_SENT)
vi_sents_raw = load_sents(DEV_VI, N_SENT)


def filter_short(sents):
    keep = []
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] >= MIN_TOKENS:
            keep.append(s)
    return keep

ja_sents = filter_short(ja_sents_raw)
vi_sents = filter_short(vi_sents_raw)
log(f"JA sentences kept: {len(ja_sents)}/{len(ja_sents_raw)}  VI kept: {len(vi_sents)}/{len(vi_sents_raw)}")
results["meta"]["ja_sents_used"] = len(ja_sents)
results["meta"]["vi_sents_used"] = len(vi_sents)
save_results()


@torch.no_grad()
def perplexity(sents):
    """Per-token ppl: tong CE tren tat ca token label / tong token -> exp."""
    total_nll, total_tok = 0.0, 0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids
        out = model(input_ids=ids, labels=ids)
        n_lab = ids.shape[1] - 1  # so token duoc cham diem
        total_nll += out.loss.item() * n_lab
        total_tok += n_lab
    return math.exp(total_nll / total_tok)


@torch.no_grad()
def translate(ja):
    msgs = [{"role": "user", "content": f"Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: {ja}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    out = model.generate(
        ids,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def evaluate(tag):
    t0 = time.time()
    ppl_ja = perplexity(ja_sents)
    log(f"[{tag}] ppl_JA = {ppl_ja:.3f} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    ppl_vi = perplexity(vi_sents)
    log(f"[{tag}] ppl_VI = {ppl_vi:.3f} ({time.time()-t0:.0f}s)")
    trans = []
    for ja in JA_TEST:
        t0 = time.time()
        hyp = translate(ja)
        log(f"[{tag}] JA: {ja}\n          -> {hyp}  ({time.time()-t0:.0f}s)")
        trans.append(hyp)
    return {"ppl_ja": ppl_ja, "ppl_vi": ppl_vi, "translations": trans}


# ------------------------------------------------------------------ baseline
log("=== BASELINE FP32 ===")
base = evaluate("baseline")
results["variants"]["baseline"] = base
save_results()

# ------------------------------------------------------------------ backup originals
log("Backing up original weights of target linears...")
orig = {name: mod.weight.detach().clone() for name, mod in target_linears.items()}


def restore():
    with torch.no_grad():
        for name, mod in target_linears.items():
            mod.weight.copy_(orig[name])


# ------------------------------------------------------------------ ternary variants
@torch.no_grad()
def ternarize_A1(w):
    """TWN per-tensor: delta=0.7*mean|W|; alpha = mean(|w| : |w|>=delta)."""
    absw = w.abs()
    delta = 0.7 * absw.mean()
    mask = absw >= delta
    t = torch.sign(w) * mask
    alpha = absw[mask].mean() if mask.any() else torch.tensor(0.0)
    return alpha * t, (t == 0).sum().item()


@torch.no_grad()
def ternarize_A2(w):
    """TWN per-row: delta, alpha rieng tung hang (dim dau ra)."""
    absw = w.abs()
    delta = 0.7 * absw.mean(dim=1, keepdim=True)
    mask = absw >= delta
    t = torch.sign(w) * mask
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
    alpha = (absw * mask).sum(dim=1, keepdim=True) / cnt
    alpha = alpha * mask.any(dim=1, keepdim=True)  # hang rong -> alpha 0
    return alpha * t, (t == 0).sum().item()


@torch.no_grad()
def ternarize_A3(w):
    """Per-row + alpha least-squares: alpha* = <W_row,T_row>/<T_row,T_row>."""
    absw = w.abs()
    delta = 0.7 * absw.mean(dim=1, keepdim=True)
    mask = absw >= delta
    t = (torch.sign(w) * mask).to(w.dtype)
    num = (w * t).sum(dim=1, keepdim=True)
    den = (t * t).sum(dim=1, keepdim=True).clamp(min=1e-12)
    alpha = num / den
    alpha = alpha * mask.any(dim=1, keepdim=True)
    return alpha * t, (t == 0).sum().item()


VARIANTS = [("A1_per_tensor", ternarize_A1), ("A2_per_row", ternarize_A2), ("A3_per_row_ls", ternarize_A3)]

for vtag, fn in VARIANTS:
    log(f"=== {vtag} ===")
    restore()
    zeros, total = 0, 0
    with torch.no_grad():
        for name, mod in target_linears.items():
            new_w, nz = fn(orig[name])
            mod.weight.copy_(new_w)
            zeros += nz
            total += new_w.numel()
    pct_zero = 100.0 * zeros / total
    log(f"[{vtag}] converted {len(target_linears)} linears, zero ratio = {pct_zero:.2f}%")
    res = evaluate(vtag)
    res["pct_zero"] = pct_zero
    res["ppl_ja_x_baseline"] = res["ppl_ja"] / base["ppl_ja"]
    res["ppl_vi_x_baseline"] = res["ppl_vi"] / base["ppl_vi"]
    results["variants"][vtag] = res
    save_results()

restore()
log("DONE. Results at " + RESULTS_PATH)
