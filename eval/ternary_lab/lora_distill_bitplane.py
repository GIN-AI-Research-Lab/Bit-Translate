# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH LORA DISTILLATION ĐA MIỀN TRÍ TUỆ (GENERAL REASONING & CHAT PRESERVATION)
Mục tiêu: Kéo Perplexity của bản nén 1.9b về ngang FP32 gốc trong 30 giây trên RTX 3060 Ti (Ampere BF16)
nhưng BẢO TOÀN 100% KHẢ NĂNG SUY LUẬN (THINKING), CHAT VÀ KIẾN THỨC TỔNG HỢP!
"""
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ.setdefault("HF_HOME", "D:/Bit-Translate-data/hf_cache")
torch.manual_seed(42)

MODEL_ID = "Qwen/Qwen3-0.6B"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if device.type == "cuda":
    log(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")

from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MODEL_ID)

log("Loading Teacher (BF16 Model) & Student (Quantized BF16)...")
teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)
teacher_model.eval()

student_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)

def quantize_block2x2_2step(W):
    out_dim, in_dim = W.shape
    W_reshaped = W.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scale1 = W_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-4)
    W_t1 = torch.round(W_reshaped / scale1).clamp(-1, 1)
    W_q1 = (W_t1 * scale1).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
    
    R1 = W - W_q1
    R1_reshaped = R1.view(out_dim // 2, 2, in_dim // 2, 2).permute(0, 2, 1, 3).reshape(-1, 4)
    scale2 = R1_reshaped.abs().mean(dim=1, keepdim=True).clamp(min=1e-4)
    W_t2 = torch.round(R1_reshaped / scale2).clamp(-1, 1)
    W_q2 = (W_t2 * scale2).view(out_dim // 2, in_dim // 2, 2, 2).permute(0, 2, 1, 3).reshape(out_dim, in_dim)
    
    return W_q1 + W_q2

# Quantize Student linear layers with BF16
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
lora_params = []
lora_layers = []

with torch.no_grad():
    for name, mod in student_model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name:
            sfx = name.split(".")[-1]
            if sfx in TARGET_SUFFIXES:
                W_q2step = quantize_block2x2_2step(mod.weight.float()).to(torch.bfloat16)
                mod.weight.copy_(W_q2step)
                
                if sfx in ("q_proj", "v_proj", "down_proj"):
                    in_dim = mod.in_features
                    out_dim = mod.out_features
                    r = 8
                    A = nn.Parameter(torch.zeros(r, in_dim, device=device, dtype=torch.bfloat16))
                    B = nn.Parameter(torch.zeros(out_dim, r, device=device, dtype=torch.bfloat16))
                    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    nn.init.zeros_(B)
                    
                    lora_params.extend([A, B])
                    lora_layers.append((mod, A, B))

log(f"Attached LoRA parameters to {len(lora_layers)} layers ({sum(p.numel() for p in lora_params):,} trainable params).")

# Register forward hooks for non-inplace BF16 LoRA addition
def make_lora_hook(A, B, scaling=2.0):
    def hook(module, input, output):
        x = input[0]
        lora_out = (x.to(torch.bfloat16) @ A.T) @ B.T
        return output + lora_out * scaling
    return hook

for mod, A, B in lora_layers:
    mod.register_forward_hook(make_lora_hook(A, B))

# 3. DỮ LIỆU CHƯNG CẤT ĐA MIỀN TRÍ TUỆ
GENERAL_CALIBRATION_CORPUS = [
    "Let's think step by step to analyze the logical relationship between premises and conclusions.",
    "Phân tích từng bước để giải quyết bài toán tối ưu hóa quy hoạch tuyến tính.",
    "Explain the fundamental difference between supervised and unsupervised learning algorithms.",
    "Write a Python function to implement binary search on a sorted array with edge cases handled.",
    "Giải thích nguyên lý hoạt động của kiến trúc Transformer và cơ chế Multi-Head Attention.",
    "Chào bạn! Hãy giúp tôi lập kế hoạch quản lý thời gian hiệu quả trong tuần này.",
    "Summarize the key factors driving international trade dynamics and global supply chain resilience.",
    "Dịch câu sau sang tiếng Việt chuẩn ngữ nghĩa: Next generation AI architectures focus on efficiency.",
    "来週のスケジュールを điều chỉnh sang tuần sau."
]

train_data = GENERAL_CALIBRATION_CORPUS * 15 # 135 calibration steps
log(f"Loaded {len(train_data)} General Multi-Domain Calibration Samples.")

optimizer = torch.optim.AdamW(lora_params, lr=2e-4)

log("=== BẮT ĐẦU CHẠY 1-EPOCH GENERAL KNOWLEDGE DISTILLATION (BF16 ENGINE) ===")
start_train_t = time.time()
student_model.train()
torch.cuda.empty_cache()

for step, text in enumerate(train_data):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=48).to(device)
    if inputs["input_ids"].shape[1] < 2:
        continue
        
    with torch.no_grad():
        teacher_logits = teacher_model(**inputs).logits.float()
        
    student_logits = student_model(**inputs).logits.float()
    loss = F.mse_loss(student_logits, teacher_logits)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    
    if (step + 1) % 30 == 0 or step == len(train_data) - 1:
        log(f"Step {step+1}/{len(train_data)} | Distillation Loss: {loss.item():.6f}")

train_time = time.time() - start_train_t
log(f"--> HOÀN THÀNH 1-EPOCH GENERAL DISTILLATION! Thời gian GPU RTX 3060 Ti: {train_time:.2f} giây ({train_time/60:.2f} phút)")

# Evaluation sample generation test
student_model.eval()
test_prompt = "Hãy giải thích ngắn gọn nguyên lý của thuật toán nén mô hình AI."
inputs_test = tok(f"User: {test_prompt}\nAssistant:", return_tensors="pt").to(device)
with torch.no_grad():
    out_ids = student_model.generate(**inputs_test, max_new_tokens=48, do_sample=False)
gen_resp = tok.decode(out_ids[0][inputs_test["input_ids"].shape[1]:], skip_special_tokens=True).strip()
log(f"=== KẾT QUẢ THỬ NGHIỆM CHAT SUY LUẬN SAU FINETUNE LORA ===")
print(f"Prompt: {test_prompt}")
print(f"Assistant Response: {gen_resp}")
