# -*- coding: utf-8 -*-
"""
Mổ kiến trúc CỤ THỂ 3 model: (1) v7a 152M ternary của dự án, (2) Qwen3-0.6B FP (bf16),
(3) Qwen3-0.6B Q4_K_M (gguf). In config + tensor/shape/dtype + cách trọng số được lưu.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

V7A = r"D:\Bit-Translate-data\ckpt_v7a_ng\last.pt"
QWEN_ST = glob.glob(r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors")[0]
QWEN_Q4 = r"D:\Bit-Translate-data\ref_models\Qwen3-0.6B-Q4_K_M.gguf"


def line(name, shape, dtype, nparams, extra=""):
    return f"  {name:44s} {str(tuple(shape)):16s} {str(dtype):12s} {nparams:>11,d}  {extra}"


def wstat(t):
    """thống kê trọng số: %zero, số mức giá trị duy nhất (mẫu), absmean — phát hiện ternary."""
    a = t.detach().float().numpy().ravel()
    samp = a[: min(len(a), 20000)]
    uniq = len(np.unique(np.round(samp, 6)))
    return f"zero={100*(samp==0).mean():4.1f}% uniq~{uniq:<5d} absmean={np.abs(a).mean():.4f}"


# ========================= (1) v7a 152M =========================
print("=" * 92)
print("(1) MODEL v7a 152M — BitNet b1.58 ternary (checkpoint train, master weights)")
print("=" * 92)
obj = torch.load(V7A, map_location="cpu", weights_only=False)
cfg = obj.get("cfg")
if cfg is not None and not isinstance(cfg, dict):
    cfg = getattr(cfg, "__dict__", vars(cfg) if hasattr(cfg, "__dict__") else str(cfg))
print(f"step: {obj.get('step')}")
print(f"CONFIG: {json.dumps(cfg, ensure_ascii=False, default=str, indent=2) if isinstance(cfg, dict) else cfg}\n")

sd = obj["model"]
print(f"Tổng số tensor: {len(sd)}   |   Tổng params: {sum(v.numel() for v in sd.values()):,}\n")

# nhóm theo tiền tố (bỏ số lớp) để thấy cấu trúc lặp
groups = {}
for k, v in sd.items():
    key = ".".join(["<N>" if part.isdigit() else part for part in k.split(".")])
    groups.setdefault(key, []).append((k, v))

print("--- CẤU TRÚC (gộp các lớp lặp; [×n] = số lớp) ---")
for gk, items in groups.items():
    k0, v0 = items[0]
    n = len(items)
    tag = f"[×{n}]" if n > 1 else ""
    print(line(gk + " " + tag, v0.shape, v0.dtype, v0.numel(), wstat(v0)))

print("\n--- MẪU CHI TIẾT: toàn bộ tensor của LỚP 0 (các lớp khác giống hệt về shape) ---")
for k, v in sd.items():
    if ".0." in k or "layers.0" in k or "blocks.0" in k or "0." in k.split(".")[1:2]:
        if any(f".{d}." in k for d in range(1, 30)):
            continue
        print(line(k, v.shape, v.dtype, v.numel(), wstat(v)))


# ========================= (2) Qwen3-0.6B FP =========================
print("\n" + "=" * 92)
print("(2) Qwen3-0.6B — FP gốc (safetensors, bf16)")
print("=" * 92)
from safetensors import safe_open

with safe_open(QWEN_ST, framework="pt") as f:
    keys = list(f.keys())
    groups2 = {}
    total = 0
    for k in keys:
        sl = f.get_slice(k)
        shape = sl.get_shape()
        dt = sl.get_dtype()
        n = int(np.prod(shape)) if shape else 0
        total += n
        gk = ".".join(["<N>" if p.isdigit() else p for p in k.split(".")])
        groups2.setdefault(gk, []).append((k, shape, dt, n))
    print(f"Tổng tensor: {len(keys)}   |   Tổng params: {total:,}\n")
    print("--- CẤU TRÚC (gộp lớp lặp) ---")
    for gk, items in groups2.items():
        k0, shape, dt, n = items[0]
        cnt = len(items)
        tag = f"[×{cnt}]" if cnt > 1 else ""
        print(line(gk + " " + tag, shape, dt, n))


# ========================= (3) Qwen3-0.6B Q4_K_M =========================
print("\n" + "=" * 92)
print("(3) Qwen3-0.6B Q4_K_M — bản lượng tử gguf (mỗi tensor một kiểu ggml)")
print("=" * 92)
try:
    import gguf
    reader = gguf.GGUFReader(QWEN_Q4)
    print(f"Tổng tensor: {len(reader.tensors)}\n")
    print("--- CẤU TRÚC + KIỂU LƯỢNG TỬ mỗi loại tensor (lớp 0 + toàn cục) ---")
    seen = {}
    for t in reader.tensors:
        gk = ".".join(["<N>" if p.isdigit() else p for p in t.name.split(".")])
        qtype = t.tensor_type.name
        # đếm phân bố kiểu quant theo nhóm
        seen.setdefault(gk, {"shape": tuple(reversed(t.shape)), "types": {}})
        seen[gk]["types"][qtype] = seen[gk]["types"].get(qtype, 0) + 1
    for gk, info in seen.items():
        types = ", ".join(f"{q}×{c}" for q, c in info["types"].items())
        print(f"  {gk:44s} {str(info['shape']):16s} {types}")
    # bảng tổng kiểu quant
    print("\n--- TỔNG: mỗi KIỂU lượng tử dùng cho bao nhiêu tensor ---")
    allt = {}
    for t in reader.tensors:
        allt[t.tensor_type.name] = allt.get(t.tensor_type.name, 0) + 1
    for q, c in sorted(allt.items(), key=lambda x: -x[1]):
        print(f"  {q:12s}: {c} tensor")
except Exception as e:
    print(f"Lỗi đọc gguf: {e}")

print("\nXONG.")
