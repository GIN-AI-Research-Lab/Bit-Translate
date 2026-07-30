# -*- coding: utf-8 -*-
"""
THI NGHIEM: PTQ Convert-only (Zero-train) Qwen3-0.6B sang Ternary (BitNet 1.58-bit).
Cac dieu kien test:
  Condition A: FP32 gốc (Baseline)
  Condition B: TWN làm tròn thuần (Per-row)
  Condition C: GPTQ-ternary (Hessian-based closed-form compensation từ 128 câu calibration en+ja+vi)
  Condition D: Q4_0 -> GPTQ-ternary (Ternary từ weight đã Q4)
  Condition E: Hybrid GPTQ-ternary (Giữ 1% column FP16)
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
RESULTS_PATH = "eval/ternary_lab/qwen3_results.json"
DEV_JA_PATH = "D:/Bit-Translate-data/clean_v6/dev.ja"
DEV_VI_PATH = "D:/Bit-Translate-data/clean_v6/dev.vi"

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

CALIB_EN_SENTS = [
    "Artificial intelligence and deep learning models are advancing rapidly across many domain areas.",
    "The translation quality between Japanese and Vietnamese requires context-aware neural architectures.",
    "Post-training quantization reduces memory footprint while attempting to preserve language modeling quality.",
    "Quantization aware training allows models to adapt parameter distributions to low-bit constraints.",
    "Edge deployment on resource constrained hardware requires lightweight matrix operations and binary execution.",
    "Matrix multiplication performance is primarily bounded by memory bandwidth during autoregressive inference.",
    "Attention mechanisms allow transformers to dynamically weight relevant input tokens across sequences.",
    "Recurrent and state space models provide alternative trade-offs for continuous sequence modeling tasks."
]

TEST_PROMPTS = [
    "Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: 来週の会議は資料が間に合わないので、日程を変更したいと思います。",
    "Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: 今日は天気がいいですね。",
    "Dịch câu sau sang tiếng Việt, chỉ trả về bản dịch: 彼は毎朝コーヒーを飲みながら新聞を読みます。"
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_sentences(path, max_count=60):
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

# Load Tokenizer & Model
log(f"Loading tokenizer & model {MODEL_ID}...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.eval()

# Collect linear modules
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

# Load evaluation data
ja_sents = load_sentences(DEV_JA_PATH, 50)
vi_sents = load_sentences(DEV_VI_PATH, 50)
en_sents = CALIB_EN_SENTS * 6

log(f"Eval sentences: JA={len(ja_sents)}, VI={len(vi_sents)}, EN={len(en_sents)}")

@torch.no_grad()
def calculate_perplexity(sents):
    if not sents:
        return 0.0
    total_nll, total_tok = 0.0, 0
    for s in sents:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
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
        out = model.generate(ids, attention_mask=mask, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
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
        log(f"[{tag}] Sample {idx+1}: {s[:100]}...")
    return {
        "ppl_en": ppl_en,
        "ppl_ja": ppl_ja,
        "ppl_vi": ppl_vi,
        "samples": samples
    }

results = {
    "model": MODEL_ID,
    "num_params": sum(p.numel() for p in model.parameters()),
    "num_target_linears": len(target_linears),
    "conditions": {}
}

# --- Condition A: FP32 Baseline ---
log("=== Running Condition A: FP32 Baseline ===")
res_a = eval_current_state("Condition_A_FP32")
results["conditions"]["A_FP32"] = res_a

# --- Condition B: TWN Pure Rounding ---
log("=== Running Condition B: Pure TWN Rounding ===")
restore_weights()
with torch.no_grad():
    total_zeros, total_params = 0, 0
    for name, mod in target_linears.items():
        w = mod.weight.data
        absw = w.abs()
        delta = 0.7 * absw.mean(dim=1, keepdim=True)
        mask = absw >= delta
        t = torch.sign(w) * mask
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1)
        alpha = (absw * mask).sum(dim=1, keepdim=True) / cnt
        w_q = alpha * t
        mod.weight.copy_(w_q)
        total_zeros += (t == 0).sum().item()
        total_params += w.numel()

res_b = eval_current_state("Condition_B_TWN")
res_b["zero_pct"] = (total_zeros / total_params) * 100.0
results["conditions"]["B_TWN"] = res_b

# --- Calibration for GPTQ (Conditions C, D, E) ---
log("=== Gathering Calibration Hessians for GPTQ ===")
restore_weights()

calib_sents = ja_sents[:30] + vi_sents[:30] + CALIB_EN_SENTS * 8 # total ~128 sents
H_matrices = {name: torch.zeros((mod.in_features, mod.in_features), dtype=torch.float32) for name, mod in target_linears.items()}
counts = {name: 0 for name in target_linears}

hooks = []
def make_hook(name):
    def hook(module, inp, out):
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        # Accumulate X^T X
        H_matrices[name] += x.t().mm(x).cpu()
        counts[name] += x.shape[0]
    return hook

for name, mod in target_linears.items():
    hooks.append(mod.register_forward_hook(make_hook(name)))

log(f"Forward pass over {len(calib_sents)} calibration sentences...")
with torch.no_grad():
    for s in calib_sents:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] > 1:
            model(input_ids=ids)

for h in hooks:
    h.remove()

log("Hessian calculation complete. Normalizing H matrices...")
for name in H_matrices:
    if counts[name] > 0:
        H_matrices[name] /= counts[name]

def quantize_gptq_layer(w_orig, H, block_size=128, per_row_alpha=True, hybrid_pct=0.0):
    """GPTQ quantization for a single weight matrix into ternary values."""
    w = w_orig.clone()
    out_features, in_features = w.shape
    
    # Add damping to H
    damp = 0.01 * torch.diag(H).mean().item()
    H_damped = H + damp * torch.eye(in_features)
    
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H_damped))
    except Exception:
        H_inv = torch.linalg.pinv(H_damped)
        
    diag_H_inv = torch.diag(H_inv)
    
    # Identify hybrid top columns if requested
    unquantized_cols = set()
    if hybrid_pct > 0.0:
        k = max(1, int(in_features * hybrid_pct))
        col_norms = (w.abs() ** 2).sum(dim=0) * torch.diag(H)
        _, top_cols = torch.topk(col_norms, k)
        unquantized_cols = set(top_cols.tolist())

    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        count = i2 - i1
        
        W1 = w[:, i1:i2].clone()
        Hinv1 = H_inv[i1:i2, i1:i2]
        
        for j in range(count):
            col_idx = i1 + j
            w_col = W1[:, j]
            
            if col_idx in unquantized_cols:
                continue
                
            absw = w_col.abs()
            delta = 0.7 * absw.mean()
            mask = absw >= delta
            t = torch.sign(w_col) * mask
            cnt = mask.sum().clamp(min=1)
            alpha = (absw * mask).sum() / cnt
            q_col = alpha * t
            
            err = w_col - q_col
            W1[:, j] = q_col
            
            # Error compensation update for remaining columns in block
            if j < count - 1:
                h_inv_jj = Hinv1[j, j]
                h_inv_sub = Hinv1[j, j+1:]
                W1[:, j+1:] -= torch.outer(err, h_inv_sub / h_inv_jj)
                
        w[:, i1:i2] = W1
        
        # Global update for remaining blocks
        if i2 < in_features:
            h_inv_block = H_inv[i1:i2, i2:]
            h_inv_diag = torch.diag(Hinv1)
            w[:, i2:] -= torch.matmul((W1 - w_orig[:, i1:i2]) / h_inv_diag.unsqueeze(0), h_inv_block)
            
    return w

# --- Condition C: GPTQ-Ternary ---
log("=== Running Condition C: GPTQ-Ternary ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w_gptq = quantize_gptq_layer(orig_weights[name], H_matrices[name], block_size=128)
        mod.weight.copy_(w_gptq)

res_c = eval_current_state("Condition_C_GPTQ")
results["conditions"]["C_GPTQ_Ternary"] = res_c

# --- Condition D: Q4 -> GPTQ-Ternary ---
log("=== Running Condition D: Q4 -> GPTQ-Ternary ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w = orig_weights[name]
        # Simulate Q4 quantization (INT4 uniform per-row scale)
        max_v = w.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = max_v / 7.0
        q4_w = torch.round(w / scale).clamp(-7, 7) * scale
        
        # Apply GPTQ ternary starting from Q4 weight
        w_gptq_q4 = quantize_gptq_layer(q4_w, H_matrices[name], block_size=128)
        mod.weight.copy_(w_gptq_q4)

res_d = eval_current_state("Condition_D_Q4_GPTQ")
results["conditions"]["D_Q4_GPTQ_Ternary"] = res_d

# --- Condition E: Hybrid 1% FP16 ---
log("=== Running Condition E: Hybrid 1% FP16 GPTQ ===")
restore_weights()
with torch.no_grad():
    for name, mod in target_linears.items():
        w_hybrid = quantize_gptq_layer(orig_weights[name], H_matrices[name], block_size=128, hybrid_pct=0.01)
        mod.weight.copy_(w_hybrid)

res_e = eval_current_state("Condition_E_Hybrid1pct")
results["conditions"]["E_Hybrid_1pct"] = res_e

# Restore original weights
restore_weights()

# Save final results JSON
os.makedirs("eval/ternary_lab", exist_ok=True)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

log(f"All experiments finished! Results saved to {RESULTS_PATH}")
