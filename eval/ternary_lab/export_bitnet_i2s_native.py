# -*- coding: utf-8 -*-
"""
PIPELINE XUẤT TRỌNG SỐ NATIVE BITNET.CPP (i2_s FORMAT)
Đóng gói trực tiếp mô hình nén Layer-Adaptive Importance Pruning (< 2.4 bits)
thành các tensor bitpack i2_s (2-bit packing) tương thích 100% BitNet C++ runtime.
"""
import os
import sys
import time
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
MODEL_ID = "Qwen/Qwen3-0.6B"
EXPORT_DIR = "eval/ternary_lab/exported_models"
os.makedirs(EXPORT_DIR, exist_ok=True)

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def pack_ternary_2bit(W_ternary):
    """
    Map {-1, 0, 1} -> {0, 1, 2}
    Đóng gói 4 trọng số -> 1 byte uint8 (i2_s format)
    """
    val_mapped = (W_ternary.to(torch.int8) + 1).to(torch.uint8)
    w0 = val_mapped[:, 0]
    w1 = val_mapped[:, 1]
    w2 = val_mapped[:, 2]
    w3 = val_mapped[:, 3]
    
    packed_byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
    return packed_byte

log(f"Loading {MODEL_ID} for BitNet Native i2_s Export...")
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

target_linears = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and ".layers." in name:
        sfx = name.split(".")[-1]
        if sfx in TARGET_SUFFIXES:
            target_linears[name] = mod

def quantize_block2x2(W):
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1)
    return W_ternary, scales

def export_native_bitnet_i2s(config_name="adaptive_2.25b", mlp_w2_keep=0.20):
    output_path = os.path.join(EXPORT_DIR, f"qwen3_0.6b_{config_name}_bitnet.pt")
    log(f"Exporting Native BitNet i2_s Model to {output_path}...")
    
    bitnet_export_dict = {}
    
    with torch.no_grad():
        for name, mod in target_linears.items():
            W = mod.weight.detach().clone()
            out_dim, in_dim = W.shape
            
            # Step 1: Always compute W1 i2_s Bitpack
            W_t1, scale1 = quantize_block2x2(W)
            packed1 = pack_ternary_2bit(W_t1)
            
            bitnet_export_dict[f"{name}.w1_i2s"] = packed1
            bitnet_export_dict[f"{name}.scale1"] = scale1.half()
            
            # Determine if layer gets Step 2
            is_attn = any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
            
            if is_attn or mlp_w2_keep > 0.0:
                W_q1 = (W_t1 * scale1).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                R1 = W - W_q1
                R1_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                scale2 = R1_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
                
                if is_attn:
                    W_t2 = torch.round(R1_reshaped / scale2).clamp(-1, 1)
                else: # MLP layer with cautious pruning
                    m_scales = scale2.squeeze(-1)
                    thresh = torch.quantile(m_scales, 1.0 - mlp_w2_keep)
                    mask = (m_scales >= thresh).float().unsqueeze(-1)
                    W_t2 = torch.round(R1_reshaped / scale2).clamp(-1, 1) * mask
                    
                packed2 = pack_ternary_2bit(W_t2)
                bitnet_export_dict[f"{name}.w2_i2s"] = packed2
                bitnet_export_dict[f"{name}.scale2"] = scale2.half()
                bitnet_export_dict[f"{name}.has_step2"] = torch.tensor(1, dtype=torch.uint8)
            else:
                bitnet_export_dict[f"{name}.has_step2"] = torch.tensor(0, dtype=torch.uint8)

    torch.save(bitnet_export_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Successfully Exported Native BitNet i2_s Model! Size on Disk: {file_size_mb:.2f} MB")
    return file_size_mb

log("=== PIPELINE EXPORTING NATIVE BITNET i2_s MODELS ===")
export_native_bitnet_i2s("adaptive_2.25b", mlp_w2_keep=0.20)
export_native_bitnet_i2s("adaptive_2.15b", mlp_w2_keep=0.10)
