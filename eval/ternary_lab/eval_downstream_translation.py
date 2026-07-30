# -*- coding: utf-8 -*-
"""
OPTION 3: ĐÁNH GIÁ CHẤT LƯỢNG DỊCH THUẬT THỰC TẾ (DOWNSTREAM EVAL)
So sánh chỉ số BLEU / ChrF và mẫu câu sinh thực tế giữa mô hình gốc FP32
và các phiên bản nén Ternary Multi-Step (4.8b, 4.0b, 3.0b, 2.6b).
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
os.environ.setdefault("OMP_NUM_THREADS", "6")

torch.set_num_threads(6)
torch.manual_seed(42)

MODEL_ID = "Qwen/Qwen3-0.6B"
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEV_JA_PATH = "D:/Bit-Translate-data/clean_v6/dev.ja"
DEV_VI_PATH = "D:/Bit-Translate-data/clean_v6/dev.vi"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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

def restore_weights():
    with torch.no_grad():
        for name, mod in target_linears.items():
            mod.weight.copy_(orig_weights[name])

def quantize_block2x2_step(W):
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1)
    W_q = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
    return W_q

def apply_quantization(target_bitrate):
    restore_weights()
    if target_bitrate == "fp32":
        return
    with torch.no_grad():
        for name, mod in target_linears.items():
            W_orig = orig_weights[name]
            out_dim, in_dim = W_orig.shape
            
            if target_bitrate == "4.8b":
                W_q1 = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_q2 = quantize_block2x2_step(R1)
                if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    R2 = R1 - W_q2
                    W_q3 = quantize_block2x2_step(R2)
                    W_final = W_q1 + W_q2 + W_q3
                else:
                    W_final = W_q1 + W_q2
            elif target_bitrate == "4.0b":
                W_q1 = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_q2 = quantize_block2x2_step(R1)
                W_final = W_q1 + W_q2
            elif target_bitrate == "3.0b":
                W_q1 = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                thresh = torch.quantile(scales.squeeze(-1), 0.50)
                mask = (scales.squeeze(-1) >= thresh).float().unsqueeze(-1)
                W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1) * mask
                W_q2 = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                W_final = W_q1 + W_q2
            elif target_bitrate == "2.6b":
                W_q1 = quantize_block2x2_step(W_orig)
                R1 = W_orig - W_q1
                W_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                thresh = torch.quantile(scales.squeeze(-1), 0.70)
                mask = (scales.squeeze(-1) >= thresh).float().unsqueeze(-1)
                W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1) * mask
                W_q2 = (W_ternary * scales).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                W_final = W_q1 + W_q2
            mod.weight.copy_(W_final)

TEST_SAMPLES = [
    "来週の会議は資料が間に合わないので、日程を変更したいと思います。",
    "今日は天気がとてもいいですね。公園に mảng xanh Đi dạo nhé.",
    "This system provides fast zero-shot model compression without training."
]

def generate_translation(prompt, max_new_tokens=64):
    formatted = f"Dịch câu sau sang tiếng Việt: {prompt}\nBản dịch:"
    inputs = tok(formatted, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    resp = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return resp

log("=== ĐÁNH GIÁ CHẤT LƯỢNG MẪU DỊCH THUẬT ===")
bitrates = ["fp32", "4.8b", "4.0b", "3.0b", "2.6b"]

for b_rate in bitrates:
    log(f"\n--- Đang đánh giá Cấu hình: {b_rate.upper()} ---")
    apply_quantization(b_rate)
    for idx, sample in enumerate(TEST_SAMPLES, 1):
        gen_text = generate_translation(sample)
        print(f"[{b_rate}] Mẫu {idx}: {gen_text}")
