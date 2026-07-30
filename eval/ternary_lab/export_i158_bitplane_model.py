# -*- coding: utf-8 -*-
"""
TRÌNH XUẤT MÔ HÌNH CHUẨN NÉN MỚI: i1.58_bitplane
Đóng gói mô hình trực tiếp sang định dạng 2 Mặt phẳng Bit (nonzero_mask & sign_mask)
cho dung lượng đĩa cứng siêu nhỏ < 360 MB.
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

log(f"Loading {MODEL_ID} for i1.58_bitplane Model Export...")
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

def export_bitplane_binary_model(config_name="sparse70_1.9b", w2_prune_ratio=0.70):
    output_path = os.path.join(EXPORT_DIR, f"qwen3_0.6b_{config_name}_bitplane.pt")
    log(f"Exporting i1.58_bitplane Model to {output_path}...")
    
    export_dict = {}
    
    with torch.no_grad():
        for name, mod in target_linears.items():
            W = mod.weight.detach().clone()
            out_dim, in_dim = W.shape
            
            # Step 1 Bitplane
            W_t1, scale1 = quantize_block2x2(W)
            
            # Encode Bit-Planes uint8
            nonzero1 = (W_t1 != 0).to(torch.uint8)
            sign1 = (W_t1 == 1).to(torch.uint8)
            
            export_dict[f"{name}.w1_nonzero"] = nonzero1
            export_dict[f"{name}.w1_sign"] = sign1
            export_dict[f"{name}.scale1"] = scale1.half()
            
            # Step 2 Residual Bitplane (Sparse 70%)
            W_q1 = (W_t1 * scale1).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
            R1 = W - W_q1
            W_t2, scale2 = quantize_block2x2(R1)
            
            if w2_prune_ratio > 0.0:
                R1_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
                m_scales = scale2.squeeze(-1)
                thresh = torch.quantile(m_scales, w2_prune_ratio)
                mask = (m_scales >= thresh).float().unsqueeze(-1)
                W_t2 = W_t2 * mask
                
            nonzero2 = (W_t2 != 0).to(torch.uint8)
            sign2 = (W_t2 == 1).to(torch.uint8)
            
            export_dict[f"{name}.w2_nonzero"] = nonzero2
            export_dict[f"{name}.w2_sign"] = sign2
            export_dict[f"{name}.scale2"] = scale2.half()

    torch.save(export_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Successfully Exported i1.58_bitplane Model! Size on Disk: {file_size_mb:.2f} MB")
    return file_size_mb

log("=== PIPELINE EXPORTING i1.58_bitplane BINARY MODELS ===")
export_bitplane_binary_model("sparse70_1.9b", w2_prune_ratio=0.70)
