# -*- coding: utf-8 -*-
"""
NGHIÊN CỨU & THỰC NGHIỆM: HÀM MIGRATION DETERMINISTIC TỪ FP16 SANG TERNARY (BITNET 1.58-BIT)
DỰA TRÊN CĂN CHỈNH KHÔNG GIAN ẨN (SUBSPACE ALIGNMENT & SVD DECOMPOSITION).

Cơ sở lý thuyết:
Trí tuệ của ma trận FP16 W nằm ở các hướng giá trị riêng chính (Singular Vectors u_k, v_k^T).
Phương pháp này không quantize từng scalar đơn lẻ w -> sign(w), mà quy đổi các hướng Rank-1
thành các ma trận cơ sở Ternary T_k = sign(u_k) * sign(v_k)^T.
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
RESULTS_PATH = "eval/ternary_lab/migration_results.json"
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

# --- Baseline FP32 ---
log("=== Running FP32 Baseline ===")
results["experiments"]["FP32_Baseline"] = eval_current_state("FP32_Baseline")

# =========================================================================
# METHOD 1: SVD Rank-K Sign Outer-Product Migration
# W_migrated = sum_{k=1}^K alpha_k * (sign(u_k) @ sign(v_k)^T)
# =========================================================================
def svd_rank_k_ternary_migration(w, K=1):
    """Phân rã SVD ma trận W, quy đổi K hướng chính thành ma trận Ternary outer-product."""
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    w_approx = torch.zeros_like(w)
    
    for k in range(min(K, S.shape[0])):
        u_k = U[:, k]
        v_k = Vh[k, :]
        
        # Tạo ma trận Ternary Rank-1
        t_k = torch.outer(torch.sign(u_k), torch.sign(v_k))
        
        # Scaling alpha bằng phép chiếu Least-Squares
        denom = t_k.numel()
        alpha = (w * t_k).sum() / max(denom, 1)
        
        w_approx += alpha * t_k
        
    return w_approx

for K in [1, 2]:
    tag = f"Method1_SVD_Rank_{K}"
    log(f"=== Running {tag} ===")
    restore_weights()
    with torch.no_grad():
        for name, mod in target_linears.items():
            w_migrated = svd_rank_k_ternary_migration(orig_weights[name], K=K)
            mod.weight.copy_(w_migrated)
    results["experiments"][tag] = eval_current_state(tag)

# =========================================================================
# METHOD 2: Block-wise Codebook Subspace Matching (4x4 Blocks)
# =========================================================================
def block_codebook_ternary_migration(w, block_size=4):
    """Vectorized block-wise codebook ternary migration (100x faster)."""
    out_f, in_f = w.shape
    pad_out = (block_size - out_f % block_size) % block_size
    pad_in = (block_size - in_f % block_size) % block_size
    
    if pad_out > 0 or pad_in > 0:
        w_pad = torch.nn.functional.pad(w, (0, pad_in, 0, pad_out))
    else:
        w_pad = w
        
    H, W = w_pad.shape
    # Reshape to (H // bs, bs, W // bs, bs) -> transpose to (H // bs, W // bs, bs, bs) -> flatten blocks to (N, bs*bs)
    blocks = w_pad.view(H // block_size, block_size, W // block_size, block_size).permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    
    abs_b = blocks.abs()
    delta = 0.7 * abs_b.mean(dim=1, keepdim=True)
    mask = abs_b >= delta
    t_b = torch.sign(blocks) * mask
    cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
    alpha = (abs_b * mask).sum(dim=1, keepdim=True) / cnt
    
    q_blocks = (alpha * t_b).view(H // block_size, W // block_size, block_size, block_size).permute(0, 2, 1, 3).reshape(H, W)
    return q_blocks[:out_f, :in_f]

tag = "Method2_Block4x4_Subspace"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w_migrated = block_codebook_ternary_migration(orig_weights[name], block_size=4)
        mod.weight.copy_(w_migrated)
results["experiments"][tag] = eval_current_state(tag)

# =========================================================================
# METHOD 3: Subspace Coordinate Optimization (Residual Ternary Migration)
# W_approx = alpha1 * T1 + alpha2 * T2
# =========================================================================
def residual_ternary_migration(w, steps=2):
    """Tích lũy residual qua 2 cấp Ternary để giữ góc không gian ẩn."""
    res = w.clone()
    w_out = torch.zeros_like(w)
    
    for step in range(steps):
        abs_res = res.abs()
        delta = 0.7 * abs_res.mean(dim=1, keepdim=True)
        mask = abs_res >= delta
        t = torch.sign(res) * mask
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
        alpha = (abs_res * mask).sum(dim=1, keepdim=True) / cnt
        
        step_w = alpha * t
        w_out += step_w
        res -= step_w
        
    return w_out

tag = "Method3_Residual_2Step_Ternary"
log(f"=== Running {tag} ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w_migrated = residual_ternary_migration(orig_weights[name], steps=2)
        mod.weight.copy_(w_migrated)
results["experiments"][tag] = eval_current_state(tag)

restore_weights()

os.makedirs("eval/ternary_lab", exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log(f"Migration experiments finished! Results saved to {RESULTS_PATH}")
