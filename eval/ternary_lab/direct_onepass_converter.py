# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH CONVERT 1-LẦN TRỰC TIẾP (DIRECT 1-PASS CONVERTER)
Chuyển đổi trực tiếp từ FP16/FP32 sang các mốc bit (6.0b, 4.8b, 4.0b, 3.0b, 2.6b)
trong ĐÚNG 1 LẦN DUY NHẤT và đo chính xác thời gian convert (giây).
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
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEV_JA_PATH = "D:/Bit-Translate-data/clean_v6/dev.ja"
DEV_VI_PATH = "D:/Bit-Translate-data/clean_v6/dev.vi"

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

log(f"Đang tải mô hình {MODEL_ID}...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

target_linears = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and ".layers." in name:
        sfx = name.split(".")[-1]
        if sfx in TARGET_SUFFIXES:
            target_linears[name] = mod

orig_weights = {name: mod.weight.detach().clone() for name, mod in target_linears.items()}
ja_sents = load_sentences(DEV_JA_PATH, 40)
vi_sents = load_sentences(DEV_VI_PATH, 40)

def restore_weights():
    with torch.no_grad():
        for name, mod in target_linears.items():
            mod.weight.copy_(orig_weights[name])

def evaluate_ppl(sentences, max_len=128):
    if not sentences:
        return 0.0
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in sentences:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1] - 1
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    if total_tokens == 0:
        return 0.0
    return math.exp(total_loss / total_tokens)

# --- THUẬT TOÁN CONVERT 1-PASS TRỰC TIẾP ---
def quantize_block2x2_step(W):
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1)
    W_q = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
    return W_q, W_ternary, scales

def convert_direct_onepass(target_bitrate="4.0b"):
    """
    Chuyển đổi trực tiếp từ FP16/FP32 sang mốc bit target chỉ trong 1 lần duyệt ma trận!
    """
    start_time = time.time()
    restore_weights()
    
    with torch.no_grad():
        for name, mod in target_linears.items():
            W_orig = orig_weights[name]
            out_dim, in_dim = W_orig.shape
            
            if target_bitrate == "6.0b": # 2x2 3-Step
                W_q1, _, _ = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_q2, _, _ = quantize_block2x2_step(R1)
                R2 = R1 - W_q2
                W_q3, _, _ = quantize_block2x2_step(R2)
                W_final = W_q1 + W_q2 + W_q3
                
            elif target_bitrate == "4.8b": # Asymmetric Mixed Hybrid (Attention 3-Step, MLP 2-Step)
                W_q1, _, _ = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_q2, _, _ = quantize_block2x2_step(R1)
                if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    R2 = R1 - W_q2
                    W_q3, _, _ = quantize_block2x2_step(R2)
                    W_final = W_q1 + W_q2 + W_q3
                else:
                    W_final = W_q1 + W_q2
                    
            elif target_bitrate == "4.0b": # Block 2x2 2-Step
                W_q1, _, _ = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_q2, _, _ = quantize_block2x2_step(R1)
                W_final = W_q1 + W_q2
                
            elif target_bitrate == "3.0b": # Sparse Residual 50%
                W_q1, _, _ = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                m_scales = scales.squeeze(-1)
                thresh = torch.quantile(m_scales, 0.50)
                mask = (m_scales >= thresh).float().unsqueeze(-1)
                W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1) * mask
                W_q2 = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                W_final = W_q1 + W_q2
                
            elif target_bitrate == "2.6b": # Sparse Residual 30%
                W_q1, _, _ = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                m_scales = scales.squeeze(-1)
                thresh = torch.quantile(m_scales, 0.70)
                mask = (m_scales >= thresh).float().unsqueeze(-1)
                W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1) * mask
                W_q2 = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                W_final = W_q1 + W_q2
                
            mod.weight.copy_(W_final)
            
    elapsed = time.time() - start_time
    return elapsed

log("=== THỬ NGHIỆM TỐC ĐỘ CONVERT 1-LẦN TRỰC TIẾP ===")
configs = ["6.0b", "4.8b", "4.0b", "3.0b", "2.6b"]
results = {}

for cfg in configs:
    log(f"--- Đang Convert trực tiếp FP32 -> {cfg} ---")
    conv_time = convert_direct_onepass(cfg)
    ppl_vi = evaluate_ppl(vi_sents)
    ppl_ja = evaluate_ppl(ja_sents)
    log(f"--> Done {cfg}: Thời gian convert = {conv_time:.2f} giây | PPL VI = {ppl_vi:.2f} | PPL JA = {ppl_ja:.2f}")
    results[cfg] = {
        "convert_time_sec": conv_time,
        "ppl_vi": ppl_vi,
        "ppl_ja": ppl_ja
    }

log("=== KẾT QUẢ TỔNG HỢP CONVERT 1-PASS ===")
for cfg, res in results.items():
    print(f"Config {cfg:5s} | Convert Time: {res['convert_time_sec']:6.2f}s | PPL VI: {res['ppl_vi']:6.2f} | PPL JA: {res['ppl_ja']:6.2f}")
