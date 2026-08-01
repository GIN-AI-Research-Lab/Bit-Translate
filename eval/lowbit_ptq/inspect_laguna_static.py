# -*- coding: utf-8 -*-
"""
(a) Phân tích TĨNH streaming trên Laguna S2.1 Q4 89GB (MoE 118B) — không load cả model.
Đọc vài tensor THẬT (expert MoE + attention + embed), dequant Q4_K -> FP, đo:
  - sai số ternary / binary / int4-g32 (rate-distortion tầng trọng số)
  - hình học: stable rank, r90%, concentration (ternary đánh trúng hướng chính?)
So với Qwen3-0.6B (số từ các exp trước) để xem model 118B có khác ở tầng trọng số không.
"""
import sys
import numpy as np
import gguf

sys.stdout.reconfigure(encoding="utf-8")

PATH = r"D:\Bit-Translate-data\models\Laguna-S-2.1-Q4_K_M.gguf"
GROUP = 32


def deq(t):
    """dequant 1 tensor gguf -> np.float32 theo shape thật."""
    arr = gguf.quants.dequantize(t.data, t.tensor_type)
    shape = tuple(int(x) for x in reversed(t.shape))  # gguf shape đảo
    return arr.reshape(shape).astype(np.float32)


def grp(W):
    R, C = W.shape
    pad = (GROUP - C % GROUP) % GROUP
    if pad:
        W = np.pad(W, ((0, 0), (0, pad)))
    return W.reshape(R, -1, GROUP), C


def err_ternary(W):
    Wg, C = grp(W)
    s = np.maximum(np.abs(Wg).mean(2, keepdims=True), 1e-8)
    T = (np.round(Wg / s).clip(-1, 1) * s).reshape(W.shape[0], -1)[:, :C]
    return float(np.linalg.norm(T - W) / (np.linalg.norm(W) + 1e-9))


def err_binary(W):
    Wg, C = grp(W)
    s = np.maximum(np.abs(Wg).mean(2, keepdims=True), 1e-8)
    B = (np.sign(Wg) * s).reshape(W.shape[0], -1)[:, :C]
    return float(np.linalg.norm(B - W) / (np.linalg.norm(W) + 1e-9))


def err_int4(W):
    Wg, C = grp(W)
    s = np.maximum(np.abs(Wg).max(2, keepdims=True) / 7.0, 1e-8)
    Q = (np.round(Wg / s).clip(-7, 7) * s).reshape(W.shape[0], -1)[:, :C]
    return float(np.linalg.norm(Q - W) / (np.linalg.norm(W) + 1e-9))


def geom(W):
    m, n = W.shape
    sv = np.linalg.svd(W, compute_uv=False)
    s2 = sv ** 2
    stable = float(s2.sum() / s2[0])
    cum = np.cumsum(s2) / s2.sum()
    r90 = int((cum < 0.90).sum()) + 1
    # concentration: sai số ternary rơi vào top-32 hướng phải
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    Wg, C = grp(W)
    s = np.maximum(np.abs(Wg).mean(2, keepdims=True), 1e-8)
    T = (np.round(Wg / s).clip(-1, 1) * s).reshape(W.shape[0], -1)[:, :C]
    R = W - T
    Vk = Vh[:32].T
    etop = float((R @ Vk).__pow__(2).sum() / ((R ** 2).sum() + 1e-9))
    conc = etop / (32 / min(m, n))
    return stable, r90, min(m, n), conc


def main():
    print(f"Đọc metadata Laguna Q4 (mmap, không load data)...")
    r = gguf.GGUFReader(PATH)
    by_name = {t.name: t for t in r.tensors}

    # chọn tensor 2D đại diện: 1 attn, 1 expert (lấy slice nếu 3D), embed
    cands = []
    for key in ["blk.1.attn_q.weight", "blk.1.attn_output.weight", "blk.1.ffn_down_exps.weight",
                "blk.1.ffn_gate_exps.weight", "blk.10.ffn_down_exps.weight", "token_embd.weight"]:
        if key in by_name:
            cands.append(key)
    if not cands:
        # fallback: liệt kê vài tên có thật
        print("Không thấy tên mặc định. Vài tensor có trong file:")
        for t in r.tensors[:30]:
            print(f"  {t.name}  {tuple(reversed(t.shape))}  {t.tensor_type.name}")
        return

    print(f"\n{'tensor':30s}{'shape':>16s}{'qtype':>7s}{'ternErr':>9s}{'binErr':>9s}"
          f"{'int4Err':>9s}{'stblRk':>8s}{'r90':>6s}{'concen':>8s}")
    print("-" * 108)
    for key in cands:
        t = by_name[key]
        W = deq(t)
        # nếu 3D (expert gộp) -> lấy expert 0
        note = ""
        if W.ndim == 3:
            W = W[0]
            note = " (expert#0)"
        if W.ndim != 2:
            continue
        # embed quá lớn -> lấy 4096 hàng đầu cho SVD nhanh
        Wg = W if W.shape[0] <= 4096 else W[:4096]
        et, eb, ei = err_ternary(W), err_binary(W), err_int4(W)
        st, r90, dim, cc = geom(Wg)
        print(f"{key+note:30s}{str(W.shape):>16s}{t.tensor_type.name:>7s}"
              f"{et*100:8.1f}%{eb*100:8.1f}%{ei*100:8.1f}%{st:8.1f}{r90:6d}{cc:8.2f}")

    print("\n--- ĐỐI CHIẾU Qwen3-0.6B (đo trước) ---")
    print("  ternary ~43% | binary ~52% | int4 ~10% | stableRank 17-117 | concen 3-9.5")
    print("Nếu Laguna cùng dải => tầng TRỌNG SỐ giống nhau (rate-distortion), lợi thế model lớn")
    print("chỉ ở tầng PPL/output (không đo được ở đây).")


if __name__ == "__main__":
    main()
