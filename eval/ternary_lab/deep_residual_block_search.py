# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM SÂU: MULTI-STEP RESIDUAL & MULTI-BLOCK CODEBOOK MIGRATION (1 TO 4 STEPS).
Đo Perplexity (EN, JA, VI), Dung lượng i2_s (MB) và Tốc độ suy luận ước tính (tok/s).
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

torch.set_num_threads(6)
torch.manual_seed(42)

MODEL_ID = "Qwen/Qwen3-0.6B"
RESULTS_PATH = "eval/ternary_lab/deep_residual_results.json"
DEV_JA_PATH = "D:/Bit-Translate-data/clean_v6/dev.ja"
DEV_VI_PATH = "D:/Bit-Translate-data/clean_v6/dev.vi"

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

CALIB_EN_SENTS = [
    "Artificial intelligence and deep learning models are advancing rapidly across many domain areas.",
    "The translation quality between Japanese and Vietnamese requires context-aware neural architectures.",
    "Post-training quantization reduces memory footprint while attempting to preserve language modeling quality.",
    "Edge deployment on resource constrained hardware requires lightweight matrix operations and binary execution."
]

TEST_PROMPTS = [
    "Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: 来週の会議は資料が間に合わないので、日程を変更したいと思います。",
    "Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: 今日は天気がいいですね。"
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_sentences(path, max_count=40):
    sents = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line)
                if len(sents) >= max_count:
                    break
    return sents

log(f"Loading model {MODEL_ID}...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

target_linears = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and ".layers." in name:
        sfx = name.split(".")[-1]
        if sfx in TARGET_SUFFIXES:
            target_linears[name] = mod

log(f"Target linear layers found: {len(target_linears)}")
orig_weights = {name: mod.weight.detach().clone() for name, mod in target_linears.items()}

def restore_weights():
    with torch.no_grad():
        for name, mod in target_linears.items():
            mod.weight.copy_(orig_weights[name])

ja_sents = load_sentences(DEV_JA_PATH, 40)
vi_sents = load_sentences(DEV_VI_PATH, 40)
en_sents = CALIB_EN_SENTS * 8

total_params = sum(p.numel() for p in model.parameters())
linear_params = sum(m.weight.numel() for m in target_linears.values())
non_linear_params = total_params - linear_params

def calculate_metrics(steps):
    """Tính toán dung lượng MB và tốc độ tok/s ước tính trên CPU (băng thông ~55GB/s)."""
    # Non-linear params: FP16 = 2 bytes/param
    non_linear_bytes = non_linear_params * 2
    # Linear params: 1-step = 2-bit (0.25 bytes/param per step)
    linear_bytes = linear_params * (0.25 * steps)
    
    total_bytes = non_linear_bytes + linear_bytes
    size_mb = total_bytes / (1024 * 1024)
    
    # Ước tính throughput CPU laptop
    bandwidth_mb_s = 55.0 * 1024.0 # 56,320 MB/s
    tok_per_sec = bandwidth_mb_s / size_mb
    
    return round(size_mb, 1), round(tok_per_sec, 1)

@torch.no_grad()
def calculate_perplexity(sents):
    if not sents:
        return 0.0
    total_nll, total_tok = 0.0, 0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            continue
        mask = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        n_lab = ids.shape[1] - 1
        total_nll += out.loss.item() * n_lab
        total_tok += n_lab
    return math.exp(total_nll / total_tok) if total_tok > 0 else 0.0

@torch.no_grad()
def generate_samples():
    samples = []
    for prompt in TEST_PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids
        mask = torch.ones_like(ids)
        out = model.generate(ids, attention_mask=mask, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        samples.append(gen_text)
    return samples

def eval_current_state(tag, steps=1):
    log(f"--- Evaluating [{tag}] (Steps={steps}) ---")
    size_mb, tok_s = calculate_metrics(steps)
    t0 = time.time()
    ppl_en = calculate_perplexity(en_sents)
    ppl_ja = calculate_perplexity(ja_sents)
    ppl_vi = calculate_perplexity(vi_sents)
    elapsed = time.time() - t0
    log(f"[{tag}] PPL EN: {ppl_en:.2f} | JA: {ppl_ja:.2f} | VI: {ppl_vi:.2f} | Size: {size_mb}MB | Speed: {tok_s}tok/s (Time: {elapsed:.1f}s)")
    samples = generate_samples()
    for idx, s in enumerate(samples):
        log(f"[{tag}] Sample {idx+1}: {s[:80]}...")
    return {
        "ppl_en": ppl_en,
        "ppl_ja": ppl_ja,
        "ppl_vi": ppl_vi,
        "size_mb": size_mb,
        "speed_tok_s": tok_s,
        "samples": samples
    }

results = {
    "model": MODEL_ID,
    "total_params": total_params,
    "target_linear_params": linear_params,
    "experiments": {}
}

# --- Core Vectorized Block Quantizer ---
def block_quantize_tensor(w, bh=4, bw=4):
    out_f, in_f = w.shape
    pad_out = (bh - out_f % bh) % bh
    pad_in = (bw - in_f % bw) % bw
    
    if pad_out > 0 or pad_in > 0:
        w_pad = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    else:
        w_pad = w
        
    H, W = w_pad.shape
    blocks = w_pad.view(H // bh, bh, W // bw, bw).permute(0, 2, 1, 3).reshape(-1, bh * bw)
    
    abs_b = blocks.abs()
    delta = 0.7 * abs_b.mean(dim=1, keepdim=True)
    mask = abs_b >= delta
    t_b = torch.sign(blocks) * mask
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
    alpha = (abs_b * mask).sum(dim=1, keepdim=True) / cnt
    
    q_blocks = (alpha * t_b).view(H // bh, W // bw, bh, bw).permute(0, 2, 1, 3).reshape(H, W)
    return q_blocks[:out_f, :in_f]

def multi_step_residual_quantize(w, bh=4, bw=4, steps=1):
    res = w.clone()
    w_out = torch.zeros_like(w)
    for _ in range(steps):
        q_step = block_quantize_tensor(res, bh=bh, bw=bw)
        w_out += q_step
        res -= q_step
    return w_out

# =========================================================================
# EXPERIMENT GROUP 1: Block 4x4 Scaling (1 to 4 Steps)
# =========================================================================
for steps in [1, 2, 3, 4]:
    tag = f"Block4x4_Step_{steps}"
    log(f"=== Running {tag} ===")
    restore_weights()
    with torch.no_grad():
        for name, mod in target_linears.items():
            w_q = multi_step_residual_quantize(orig_weights[name], bh=4, bw=4, steps=steps)
            mod.weight.copy_(w_q)
    results["experiments"][tag] = eval_current_state(tag, steps=steps)

# =========================================================================
# EXPERIMENT GROUP 2: Block 2x2 Scaling (1 to 3 Steps)
# =========================================================================
for steps in [1, 2, 3]:
    tag = f"Block2x2_Step_{steps}"
    log(f"=== Running {tag} ===")
    restore_weights()
    with torch.no_grad():
        for name, mod in target_linears.items():
            w_q = multi_step_residual_quantize(orig_weights[name], bh=2, bw=2, steps=steps)
            mod.weight.copy_(w_q)
    results["experiments"][tag] = eval_current_state(tag, steps=steps)

# =========================================================================
# EXPERIMENT GROUP 3: Hybrid Orthogonal Block Residual (Row 1x4 -> Col 4x1 -> Block 2x2)
# =========================================================================
tag = "Hybrid_Orthogonal_3Step_Block"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        # Step 1: Row 1x4
        w1 = block_quantize_tensor(w, bh=1, bw=4)
        res1 = w - w1
        # Step 2: Col 4x1
        w2 = block_quantize_tensor(res1, bh=4, bw=1)
        res2 = res1 - w2
        # Step 3: Block 2x2
        w3 = block_quantize_tensor(res2, bh=2, bw=2)
        mod.weight.copy_(w1 + w2 + w3)
results["experiments"][tag] = eval_current_state(tag, steps=3)

restore_weights()

os.makedirs("eval/ternary_lab", exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log(f"Deep residual experiments finished! Results saved to {RESULTS_PATH}")
