# -*- coding: utf-8 -*-
"""
OPTION 1: TRÌNH XUẤT TRỌNG SỐ TERNARY BITPACK REAL FORMAT (i2_s)
Đóng gói trực tiếp 4 trọng số Ternary (-1, 0, +1) vào 1 byte uint8 (2 bits/weight)
kèm theo Vector Scale khối alpha_1, alpha_2 và lưu ra file nhị phân đĩa cứng.
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

log(f"Đang tải mô hình {MODEL_ID} để đóng gói Bitpack...")
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.eval()

target_linears = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and ".layers." in name:
        sfx = name.split(".")[-1]
        if sfx in TARGET_SUFFIXES:
            target_linears[name] = mod

def pack_ternary_2bit(W_ternary):
    """
    W_ternary shape (-1, 4) chứa các giá trị {-1.0, 0.0, +1.0}
    Map: -1 -> 0 (00b), 0 -> 1 (01b), +1 -> 2 (10b)
    Đóng gói 4 trọng số -> 1 byte uint8
    """
    val_mapped = (W_ternary.to(torch.int8) + 1).to(torch.uint8) # {-1, 0, 1} -> {0, 1, 2}
    w0 = val_mapped[:, 0]
    w1 = val_mapped[:, 1]
    w2 = val_mapped[:, 2]
    w3 = val_mapped[:, 3]
    
    packed_byte = w0 | (w1 << 2) | (w2 << 4) | (w3 << 6)
    return packed_byte

def export_bitpack_model(target_bitrate="4.0b"):
    output_path = os.path.join(EXPORT_DIR, f"qwen3_0.6b_{target_bitrate}.pt")
    log(f"--> Đang xuất cấu hình {target_bitrate} sang file bitpack {output_path}...")
    
    packed_state_dict = {}
    
    with torch.no_grad():
        for name, mod in target_linears.items():
            W = mod.weight.detach().clone()
            out_dim, in_dim = W.shape
            
            # Step 1: Block 2x2 Quantize
            W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
            scale1 = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            W_t1 = torch.round(W_reshaped / scale1).clamp(-1, 1)
            packed1 = pack_ternary_2bit(W_t1)
            
            packed_state_dict[f"{name}.w1_packed"] = packed1
            packed_state_dict[f"{name}.scale1"] = scale1.half() # Store FP16 scale
            
            # Step 2: Residual Block 2x2 Quantize
            W_q1 = (W_t1 * scale1).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
            R1 = W - W_q1
            
            R1_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
            scale2 = R1_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            
            if target_bitrate in ("3.0b", "2.6b"):
                keep_pct = 0.50 if target_bitrate == "3.0b" else 0.30
                thresh = torch.quantile(scale2.squeeze(-1), 1.0 - keep_pct)
                mask = (scale2.squeeze(-1) >= thresh).float().unsqueeze(-1)
                W_t2 = torch.round(R1_reshaped / scale2).clamp(-1, 1) * mask
            else:
                W_t2 = torch.round(R1_reshaped / scale2).clamp(-1, 1)
                
            packed2 = pack_ternary_2bit(W_t2)
            packed_state_dict[f"{name}.w2_packed"] = packed2
            packed_state_dict[f"{name}.scale2"] = scale2.half()

    torch.save(packed_state_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"Xuất file THÀNH CÔNG: {output_path} | Dung lượng đĩa thật: {file_size_mb:.2f} MB")
    return file_size_mb

log("=== BẮT ĐẦU XUẤT FILE TRỌNG SỐ TERNARY BITPACK REAL FORMAT ===")
for b_rate in ["4.0b", "3.0b", "2.6b"]:
    export_bitpack_model(b_rate)
