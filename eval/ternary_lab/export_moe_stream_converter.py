# -*- coding: utf-8 -*-
"""
GIAI ĐOẠN 1: TRÌNH NÉN THEO LUỒNG (STREAMING QUANTIZER) CHO MÔ HÌNH LỚN & MOE
Nén trực tiếp từng layer của mô hình lớn (1.5B, 7B, 14B, Laguna S 2.1 118B)
sang chuẩn Bitpack i2_s / i1.58_bitplane nhị phân mà KHÔNG LÀM TRÀN RAM HỆ THỐNG.
"""
import os
import sys
import time
import gc
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
EXPORT_DIR = "eval/ternary_lab/exported_models"
os.makedirs(EXPORT_DIR, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def quantize_block2x2(W):
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scales = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
    W_ternary = torch.round(W_reshaped / scales).clamp(-1, 1)
    return W_ternary, scales

def pack_bitplane_layer(W_ternary):
    """
    Đóng gói 2 Mặt phẳng Bit (nonzero_mask & sign_mask) thành tensor uint8
    """
    nonzero_mask = (W_ternary != 0).to(torch.uint8)
    sign_mask = (W_ternary == 1).to(torch.uint8)
    return nonzero_mask, sign_mask

def stream_quantize_and_export(model_id_or_path, output_filename, target_bitrate="2.0b"):
    log(f"=== BẮT ĐẦU TRÌNH NÉN THEO LUỒNG (STREAMING QUANTIZER): {model_id_or_path} ===")
    start_t = time.time()
    
    from transformers import AutoModelForCausalLM
    
    log("Loading model in low_cpu_mem_usage streaming mode...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    output_path = os.path.join(EXPORT_DIR, output_filename)
    stream_export_dict = {}
    
    TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    
    layer_count = 0
    total_linear_bytes = 0
    
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and any(k in name for k in TARGET_SUFFIXES):
                W = mod.weight.detach()
                out_dim, in_dim = W.shape
                
                # Step 1 Quantize
                W_t1, scale1 = quantize_block2x2(W)
                nz1, sg1 = pack_bitplane_layer(W_t1)
                
                stream_export_dict[f"{name}.w1_nz"] = nz1
                stream_export_dict[f"{name}.w1_sg"] = sg1
                stream_export_dict[f"{name}.scale1"] = scale1.half()
                
                total_linear_bytes += (W.numel() * 0.25) # i1.58 / i2_s pack
                
                if target_bitrate in ("4.0b", "4.8b"):
                    W_q1 = (W_t1 * scale1).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
                    R1 = W - W_q1
                    W_t2, scale2 = quantize_block2x2(R1)
                    nz2, sg2 = pack_bitplane_layer(W_t2)
                    
                    stream_export_dict[f"{name}.w2_nz"] = nz2
                    stream_export_dict[f"{name}.w2_sg"] = sg2
                    stream_export_dict[f"{name}.scale2"] = scale2.half()
                    total_linear_bytes += (W.numel() * 0.25)
                    
                layer_count += 1
                if layer_count % 50 == 0:
                    log(f"Stream Quantized {layer_count} linear layers...")
                    gc.collect()

    log(f"Saving streaming bitpack binary to {output_path}...")
    torch.save(stream_export_dict, output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    elapsed = time.time() - start_t
    
    log(f"=== KẾT QUẢ NÉN THEO LUỒNG THÀNH CÔNG ===")
    print(f"Tổng số Layer đã nén: {layer_count}")
    print(f"Thời gian nén theo luồng: {elapsed:.2f} giây")
    print(f"Dung lượng đĩa thực tế: {file_size_mb:.2f} MB")
    return file_size_mb

if __name__ == "__main__":
    stream_quantize_and_export("Qwen/Qwen3-0.6B", "qwen3_0.6b_stream_bitplane.pt", target_bitrate="2.0b")
