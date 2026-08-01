# -*- coding: utf-8 -*-
"""
Exp B — Bảng sai số PTQ trên TRỌNG SỐ THẬT Qwen3-0.6B (không mô phỏng, không hệ số bịa).

Mục đích: lấp dải 1.58–3 bpw còn thiếu trong bảng đã kiểm chứng ở native_engine/KETQUA_ENGINE.md §5
(ternary-bitplane 44-47%, int3-g32 23%, int4-g32 10%, int8 <1%) bằng cùng phương pháp:
sai số tương đối ||W_q - W|| / ||W|| trên các ma trận linear thật, group-32 theo cột.

Scheme đo:
  - int4-g32 (anchor, kỳ vọng ~10%)
  - int3-g32 (anchor, kỳ vọng ~23%)
  - int2-g32 RTN (absmax/1.5, clip [-2,1])
  - int2-g32 Lloyd (codebook 4 mức tối ưu Gaussian ±0.4528σ, ±1.5104σ)
  - ternary-g32 Lloyd (3 mức: ngưỡng 0.6120σ, giá trị = mean |w| phần đuôi — tối ưu cho Gaussian)
  - ternary 2-plane residual (T1 + T2 trên phần dư, mỗi plane scale riêng)
  - int2-g32 Lloyd + 1% outlier giữ f16 (kiểu BiLLM-lite)

Mỗi scheme in kèm chi phí lưu trữ thật (bits/weight gồm scale f16/G32 + outlier nếu có).
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

import numpy as np
import torch
from safetensors import safe_open

sys.stdout.reconfigure(encoding="utf-8")

SNAP_GLOB = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*\model.safetensors"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_b_results.json")

# Lấy đại diện các loại ma trận ở đầu/giữa/cuối mạng
LAYERS = [0, 6, 13, 20, 27]
MAT_KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def rel_err(w, wq):
    return float(np.linalg.norm(wq - w) / (np.linalg.norm(w) + 1e-12))


def group_view(w):
    """[R,C] -> [R, G, 32]; yêu cầu C chia hết 32 (Qwen3-0.6B thỏa)."""
    R, C = w.shape
    assert C % 32 == 0, f"C={C} không chia hết 32"
    return w.reshape(R, C // 32, 32)


def dequant_int_g32(w, qmax_pos, qmax_neg, denom):
    """RTN đối xứng absmax: s = absmax/denom, q = round(w/s).clip."""
    wg = group_view(w)
    s = np.abs(wg).max(axis=2, keepdims=True) / denom
    s = np.maximum(s, 1e-8).astype(np.float16).astype(np.float32)
    q = np.rint(wg / s).clip(qmax_neg, qmax_pos)
    return (q * s).reshape(w.shape)


def dequant_codebook_g32(w, code_pos):
    """Codebook đối xứng theo σ của group: mức = ±code_pos σ. Gán mức gần nhất."""
    wg = group_view(w)
    sigma = wg.std(axis=2, keepdims=True)
    sigma = np.maximum(sigma, 1e-8)
    levels = np.array(sorted([-c for c in code_pos] + list(code_pos)), dtype=np.float32)
    z = wg / sigma  # chuẩn hóa
    idx = np.abs(z[..., None] - levels[None, None, None, :]).argmin(axis=-1)
    zq = levels[idx]
    return (zq * sigma).reshape(w.shape)


def dequant_ternary_lloyd_g32(w):
    """3 mức tối ưu cho Gaussian: ngưỡng t=0.6120σ; giá trị ± = mean|w| của phần |w|>t (per group)."""
    wg = group_view(w)
    sigma = np.maximum(wg.std(axis=2, keepdims=True), 1e-8)
    t = 0.6120 * sigma
    mask = np.abs(wg) > t
    # giá trị mức: trung bình |w| trên phần vượt ngưỡng của từng group
    absw = np.abs(wg) * mask
    cnt = np.maximum(mask.sum(axis=2, keepdims=True), 1)
    v = absw.sum(axis=2, keepdims=True) / cnt
    wq = np.sign(wg) * v * mask
    return wq.reshape(w.shape)


def dequant_ternary_2plane(w):
    """Plane 1 ternary Lloyd, plane 2 ternary Lloyd trên residual."""
    w1 = dequant_ternary_lloyd_g32(w)
    r = w - w1
    w2 = dequant_ternary_lloyd_g32(r)
    return w1 + w2


def apply_outliers(w, wq, frac=0.01):
    """Giữ lại frac trọng số |w| lớn nhất ở f16 (theo tensor)."""
    k = max(1, int(w.size * frac))
    thr = np.partition(np.abs(w).ravel(), w.size - k)[w.size - k]
    mask = np.abs(w) >= thr
    out = wq.copy()
    out[mask] = w[mask].astype(np.float16).astype(np.float32)
    return out, float(mask.mean())


SCHEMES = {
    # tên: (hàm, bits/weight lưu trữ thực tế: payload + scale f16/32w [+ outlier])
    "int4-g32 (anchor)":       (lambda w: dequant_int_g32(w, 7, -7, 7.0),        4 + 0.5),
    "int3-g32 (anchor)":       (lambda w: dequant_int_g32(w, 3, -3, 3.0),        3 + 0.5),
    "int2-g32 RTN":            (lambda w: dequant_int_g32(w, 1, -2, 1.5),        2 + 0.5),
    "int2-g32 Lloyd":          (lambda w: dequant_codebook_g32(w, [0.4528, 1.5104]), 2 + 0.5),
    "ternary-g32 Lloyd":       (dequant_ternary_lloyd_g32,                       2 + 0.5),   # 2 bit pack + scale
    "ternary 2-plane":         (dequant_ternary_2plane,                          4 + 1.0),   # 2 plane + 2 scale
}


def main():
    paths = glob.glob(SNAP_GLOB)
    if not paths:
        print(f"KHÔNG tìm thấy safetensors: {SNAP_GLOB}")
        sys.exit(1)
    st_path = paths[0]
    print(f"Trọng số thật: {st_path}")

    results = {name: {"sse": 0.0, "ref": 0.0} for name in SCHEMES}
    results["int2-g32 Lloyd + 1% outlier f16"] = {"sse": 0.0, "ref": 0.0}
    n_mats = 0

    with safe_open(st_path, framework="pt") as f:
        keys = list(f.keys())
        for layer in LAYERS:
            for kind in MAT_KINDS:
                cands = [k for k in keys if f"layers.{layer}." in k and kind in k and k.endswith(".weight")]
                if not cands:
                    continue
                key = cands[0]
                w = f.get_tensor(key).float().numpy()
                if w.ndim != 2 or w.shape[1] % 32 != 0:
                    continue
                n_mats += 1
                ref = float(np.linalg.norm(w) ** 2)
                for name, (fn, _bpw) in SCHEMES.items():
                    wq = fn(w)
                    results[name]["sse"] += float(np.linalg.norm(wq - w) ** 2)
                    results[name]["ref"] += ref
                # outlier variant dựa trên int2 Lloyd
                wq2 = dequant_codebook_g32(w, [0.4528, 1.5104])
                wq2o, _ = apply_outliers(w, wq2, 0.01)
                results["int2-g32 Lloyd + 1% outlier f16"]["sse"] += float(np.linalg.norm(wq2o - w) ** 2)
                results["int2-g32 Lloyd + 1% outlier f16"]["ref"] += ref
                print(f"  [{n_mats:3d}] layer {layer:2d} {kind:10s} {tuple(w.shape)}", flush=True)

    bpw_map = {name: bpw for name, (_fn, bpw) in SCHEMES.items()}
    # outlier: int2 payload 2 + scale 0.5 + 1% * (16 bit giá trị + ~20 bit vị trí) ≈ 2.86
    bpw_map["int2-g32 Lloyd + 1% outlier f16"] = 2 + 0.5 + 0.01 * (16 + 20)

    print("\n" + "=" * 78)
    print(f"BẢNG SAI SỐ PTQ TRÊN {n_mats} MA TRẬN THẬT Qwen3-0.6B (layers {LAYERS})")
    print("=" * 78)
    print(f"{'Scheme':38s} {'bpw lưu trữ':>12s} {'sai số tương đối':>18s}")
    print("-" * 78)
    out = {}
    for name in results:
        r = results[name]
        e = float(np.sqrt(r["sse"] / (r["ref"] + 1e-12)))
        out[name] = {"bpw": bpw_map[name], "rel_err": e}
        print(f"{name:38s} {bpw_map[name]:12.2f} {e * 100:17.1f}%")
    print("=" * 78)
    print("Chuẩn đối chiếu KETQUA_ENGINE.md §5: int4-g32 ~10%, int3-g32 ~23%, ternary cũ 44-47%")

    with open(OUT_JSON, "w", encoding="utf-8") as fo:
        json.dump({"snapshot": st_path, "n_mats": n_mats, "layers": LAYERS, "results": out}, fo,
                  ensure_ascii=False, indent=2)
    print(f"Đã lưu: {OUT_JSON}")


if __name__ == "__main__":
    main()
