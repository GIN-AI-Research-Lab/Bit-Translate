# -*- coding: utf-8 -*-
"""
THỰC NGHIỆM NÂNG CAO: THỬ NGHIỆM CÁC Ý TƯỞNG CẢI TIẾN BLOCK-WISE & RESIDUAL TERNARY MIGRATION.

Cá ý tưởng thử nghiệm:
  1. Khảo sát kích thước khối: 2x2, 8x8, 16x16, 1x4 (Row-vector), 4x1 (Col-vector).
  2. Combination: Block 4x4 + Multi-step Residual (2-Step Block 4x4).
  3. Rotational Hadamard + Block 4x4.
  4. Hybrid Salient Channel 0.5% FP16 + Block 4x4.
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
RESULTS_PATH = "eval/ternary_lab/advanced_results.json"
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

def eval_current_state(tag):
    log(f"--- Evaluating [{tag}] ---")
    t0 = time.time()
    ppl_en = calculate_perplexity(en_sents)
    ppl_ja = calculate_perplexity(ja_sents)
    ppl_vi = calculate_perplexity(vi_sents)
    elapsed = time.time() - t0
    log(f"[{tag}] PPL EN: {ppl_en:.2f} | JA: {ppl_ja:.2f} | VI: {ppl_vi:.2f} (Time: {elapsed:.1f}s)")
    samples = generate_samples()
    for idx, s in enumerate(samples):
        log(f"[{tag}] Sample {idx+1}: {s[:80]}...")
    return {
        "ppl_en": ppl_en,
        "ppl_ja": ppl_ja,
        "ppl_vi": ppl_vi,
        "samples": samples
    }

results = {
    "model": MODEL_ID,
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

# --- Experiment Group 1: Block Sizes Search ---
BLOCK_CONFIGS = [
    ("Block_2x2", 2, 2),
    ("Block_8x8", 8, 8),
    ("Block_16x16", 16, 16),
    ("Block_1x4_Row", 1, 4),
    ("Block_4x1_Col", 4, 1)
]

for tag, bh, bw in BLOCK_CONFIGS:
    log(f"=== Running {tag} ===")
    restore_weights()
    with torch.no_grad():
        for name, mod in target_linears.items():
            w_q = block_quantize_tensor(orig_weights[name], bh=bh, bw=bw)
            mod.weight.copy_(w_q)
    results["experiments"][tag] = eval_current_state(tag)

# --- Experiment Group 2: Block 4x4 + Multi-step Residual (2-Step Block 4x4) ---
tag = "Block4x4_Residual_2Step"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        w1 = block_quantize_tensor(w, bh=4, bw=4)
        res1 = w - w1
        w2 = block_quantize_tensor(res1, bh=4, bw=4)
        mod.weight.copy_(w1 + w2)
results["experiments"][tag] = eval_current_state(tag)

# --- Experiment Group 3: Rotational Hadamard + Block 4x4 ---
def get_hadamard_matrix(n):
    """Tạo ma trận Hadamard n x n (n là lũy thừa của 2)."""
    if n == 1:
        return torch.tensor([[1.0]])
    H = get_hadamard_matrix(n // 2)
    top = torch.cat([H, H], dim=1)
    bot = torch.cat([H, -H], dim=1)
    return torch.cat([top, bot], dim=0) / math.sqrt(2)

tag = "Rotational_Hadamard_Block4x4"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        out_f, in_f = w.shape
        
        # Rotational sign transformation
        r1 = torch.where(torch.randn(out_f) > 0, 1.0, -1.0).unsqueeze(1)
        r2 = torch.where(torch.randn(in_f) > 0, 1.0, -1.0).unsqueeze(0)
        w_rot = w * r1 * r2
        
        w_q_rot = block_quantize_tensor(w_rot, bh=4, bw=4)
        w_unrot = w_q_rot * r1 * r2
        mod.weight.copy_(w_unrot)
results["experiments"][tag] = eval_current_state(tag)

# --- Experiment Group 4: Salient Channel 0.5% FP16 + Block 4x4 ---
tag = "Salient_Channel_0.5pct_Block4x4"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        out_f, in_f = w.shape
        
        col_norms = (w ** 2).sum(dim=0)
        k_salient = max(1, int(in_f * 0.005))
        _, top_cols = torch.topk(col_norms, k_salient)
        
        w_q = block_quantize_tensor(w, bh=4, bw=4)
        # Khôi phục các kênh trọng yếu FP16
        w_q[:, top_cols] = w[:, top_cols]
        mod.weight.copy_(w_q)
results["experiments"][tag] = eval_current_state(tag)

restore_weights()

os.makedirs("eval/ternary_lab", exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log(f"Advanced experiments finished! Results saved to {RESULTS_PATH}")
