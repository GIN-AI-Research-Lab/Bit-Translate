# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM TỐI ƯU DUNG LƯỢNG TỐI GIẢN (2.6 - 3.0 BITS) & MÔ PHỎNG HIỆU NĂNG MOE.
Thử nghiệm:
  1. Sparse Residual Block 50% (Tỉa 50% khối W2 yếu).
  2. Sparse Residual Block 70% (Tỉa 70% khối W2 yếu).
  3. Middle-Layer Selective 2-Step (Tầng 6-18 dùng 2-Step, các tầng khác dùng 1-Step).
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
RESULTS_PATH = "eval/ternary_lab/ultra_moe_results.json"
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

log(f"Target linears found: {len(target_linears)}")
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

def calculate_custom_metrics(linear_steps_dict):
    non_linear_bytes = non_linear_params * 2 # FP16
    total_linear_bytes = 0.0
    total_linear_bits = 0.0
    
    for name, effective_steps in linear_steps_dict.items():
        n_p = target_linears[name].weight.numel()
        b_p = 0.25 * effective_steps
        total_linear_bytes += n_p * b_p
        total_linear_bits += n_p * (2.0 * effective_steps)
        
    avg_bits = total_linear_bits / linear_params
    total_bytes = non_linear_bytes + total_linear_bytes
    size_mb = total_bytes / (1024 * 1024)
    
    bandwidth_mb_s = 55.0 * 1024.0
    tok_per_sec = bandwidth_mb_s / size_mb
    
    return round(avg_bits, 2), round(size_mb, 1), round(tok_per_sec, 1)

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

def eval_current_state(tag, linear_steps_dict):
    log(f"--- Evaluating [{tag}] ---")
    avg_bits, size_mb, tok_s = calculate_custom_metrics(linear_steps_dict)
    t0 = time.time()
    ppl_en = calculate_perplexity(en_sents)
    ppl_ja = calculate_perplexity(ja_sents)
    ppl_vi = calculate_perplexity(vi_sents)
    elapsed = time.time() - t0
    log(f"[{tag}] Bits: {avg_bits}b | Size: {size_mb}MB | Speed: {tok_s}tok/s | PPL EN: {ppl_en:.2f} | JA: {ppl_ja:.2f} | VI: {ppl_vi:.2f} (Time: {elapsed:.1f}s)")
    samples = generate_samples()
    for idx, s in enumerate(samples):
        log(f"[{tag}] Sample {idx+1}: {s[:80]}...")
    return {
        "avg_bits": avg_bits,
        "size_mb": size_mb,
        "speed_tok_s": tok_s,
        "ppl_en": ppl_en,
        "ppl_ja": ppl_ja,
        "ppl_vi": ppl_vi,
        "samples": samples
    }

results = {
    "model": MODEL_ID,
    "experiments": {}
}

def block_quantize_tensor(w, bh=2, bw=2):
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

def sparse_step2_quantize(w, bh=2, bw=2, keep_ratio=0.5):
    """Nén W1 chuẩn + Nén W2 tỉa bỏ (1 - keep_ratio) số khối yếu."""
    w1 = block_quantize_tensor(w, bh=bh, bw=bw)
    res1 = w - w1
    
    # Vectorized block norm calculation for W2
    out_f, in_f = res1.shape
    pad_out = (bh - out_f % bh) % bh
    pad_in = (bw - in_f % bw) % bw
    res1_pad = torch.nn.functional.pad(res1, (0, pad_in, 0, pad_out))
    H, W = res1_pad.shape
    blocks = res1_pad.view(H // bh, bh, W // bw, bw).permute(0, 2, 1, 3).reshape(-1, bh * bw)
    
    block_norms = (blocks ** 2).sum(dim=1)
    k_keep = max(1, int(blocks.shape[0] * keep_ratio))
    _, top_indices = torch.topk(block_norms, k_keep)
    
    sparse_mask = torch.zeros(blocks.shape[0], dtype=torch.bool, device=w.device)
    sparse_mask[top_indices] = True
    
    # Quantize only top blocks
    abs_b = blocks.abs()
    delta = 0.7 * abs_b.mean(dim=1, keepdim=True)
    mask = abs_b >= delta
    t_b = torch.sign(blocks) * mask
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
    alpha = (abs_b * mask).sum(dim=1, keepdim=True) / cnt
    q_b = alpha * t_b
    q_b[~sparse_mask] = 0.0
    
    w2 = q_b.view(H // bh, W // bw, bh, bw).permute(0, 2, 1, 3).reshape(H, W)[:out_f, :in_f]
    return w1 + w2

# =========================================================================
# EXPERIMENT 1: Sparse Residual 50% (Keep 50% W2 blocks -> ~3.0 bits)
# =========================================================================
tag = "Sparse_Residual_50pct_3.0Bit"
log(f"=== Running {tag} ===")
restore_weights()
linear_steps_dict = {}
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        linear_steps_dict[name] = 1.5 # 1 step + 0.5 step
        w_q = sparse_step2_quantize(w, bh=2, bw=2, keep_ratio=0.5)
        mod.weight.copy_(w_q)
results["experiments"][tag] = eval_current_state(tag, linear_steps_dict)

# =========================================================================
# EXPERIMENT 2: Sparse Residual 30% (Keep 30% W2 blocks -> ~2.6 bits)
# =========================================================================
tag = "Sparse_Residual_30pct_2.6Bit"
log(f"=== Running {tag} ===")
restore_weights()
linear_steps_dict = {}
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        linear_steps_dict[name] = 1.3 # 1 step + 0.3 step
        w_q = sparse_step2_quantize(w, bh=2, bw=2, keep_ratio=0.3)
        mod.weight.copy_(w_q)
results["experiments"][tag] = eval_current_state(tag, linear_steps_dict)

# =========================================================================
# EXPERIMENT 3: Middle-Layer Selective 2-Step (Layers 6 to 18 use 2-Step)
# =========================================================================
tag = "Middle_Layer_Selective_3.0Bit"
log(f"=== Running {tag} ===")
restore_weights()
linear_steps_dict = {}
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        # Extract layer index
        layer_idx = int(name.split(".layers.")[1].split(".")[0])
        if 6 <= layer_idx <= 18:
            steps = 2
            w1 = block_quantize_tensor(w, bh=2, bw=2)
            res1 = w - w1
            w2 = block_quantize_tensor(res1, bh=2, bw=2)
            w_q = w1 + w2
        else:
            steps = 1
            w_q = block_quantize_tensor(w, bh=2, bw=2)
        linear_steps_dict[name] = steps
        mod.weight.copy_(w_q)
results["experiments"][tag] = eval_current_state(tag, linear_steps_dict)

restore_weights()

os.makedirs("eval/ternary_lab", exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log(f"Ultra compression experiments finished! Results saved to {RESULTS_PATH}")
