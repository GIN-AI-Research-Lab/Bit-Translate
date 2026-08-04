# -*- coding: utf-8 -*-
r"""TQ33 Runner — Giai đoạn 2, bước 4: so khớp qwen3_runner_tq33.exe (TQ33 + int8-activation)
với oracle PyTorch, per-layer, để thấy nhiễu lượng tử hoá tích luỹ qua 28 layer thế nào
(kỳ vọng rel err > Phase 1 nhưng vẫn nhỏ, top-1 token vẫn hợp lý)."""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ORACLE_DIR = r"D:\Bit-Translate-data\tq33_runner\oracle"
RUNNER_OUT = r"D:\Bit-Translate-data\tq33_runner\runner_output_tq33.bin"
F32_RUNNER_OUT = r"D:\Bit-Translate-data\tq33_runner\runner_output_f32.bin"


def rel_err(a, b):
    num = np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    den = np.sqrt(np.sum(b.astype(np.float64) ** 2)) + 1e-30
    return float(num / den)


def load_runner_bin(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        seq_len, hidden, vocab, n_layer = [int(x) for x in hdr]
        layer_outs = []
        for l in range(n_layer):
            arr = np.fromfile(f, dtype=np.float32, count=seq_len * hidden).reshape(seq_len, hidden)
            layer_outs.append(arr)
        final_h = np.fromfile(f, dtype=np.float32, count=seq_len * hidden).reshape(seq_len, hidden)
        logits = np.fromfile(f, dtype=np.float32, count=seq_len * vocab).reshape(seq_len, vocab)
    return seq_len, hidden, vocab, n_layer, layer_outs, final_h, logits


def main():
    seq_len, hidden, vocab, n_layer, layer_outs, final_h, logits = load_runner_bin(RUNNER_OUT)
    print(f"header: seq_len={seq_len} hidden={hidden} vocab={vocab} n_layer={n_layer}")

    print(f"\n=== TQ33 (int8-activation) vs oracle PyTorch (F32 chinh xac) ===")
    print(f"{'layer':>6} {'rel_err (L2)':>14} {'max_abs_diff':>14}")
    for l in range(n_layer):
        oracle = np.load(f"{ORACLE_DIR}\\hidden_l{l}.npy")
        re = rel_err(layer_outs[l], oracle)
        md = float(np.max(np.abs(layer_outs[l].astype(np.float64) - oracle.astype(np.float64))))
        print(f"{l:>6} {re:>14.3e} {md:>14.3e}")

    oracle_final = np.load(f"{ORACLE_DIR}\\hidden_final.npy")
    re_final = rel_err(final_h, oracle_final)
    print(f"\nfinal (post-norm)  rel_err = {re_final:.3e}")

    oracle_logits = np.load(f"{ORACLE_DIR}\\logits.npy")
    re_logits = rel_err(logits, oracle_logits)
    md_logits = float(np.max(np.abs(logits.astype(np.float64) - oracle_logits.astype(np.float64))))
    print(f"logits             rel_err = {re_logits:.3e}   max_abs_diff = {md_logits:.3e}")

    top1_tq = int(np.argmax(logits[-1]))
    top1_py = int(np.argmax(oracle_logits[-1]))
    print(f"\ntop-1 token (TQ33 runner) @ vi tri cuoi: {top1_tq}")
    print(f"top-1 token (PyTorch)     @ vi tri cuoi: {top1_py}")
    print(f"top-1 KHOP: {top1_tq == top1_py}")

    top5_tq = set(np.argsort(-logits[-1])[:5].tolist())
    top5_py = set(np.argsort(-oracle_logits[-1])[:5].tolist())
    print(f"top-5 overlap: {len(top5_tq & top5_py)}/5   (TQ33={sorted(top5_tq)}  oracle={sorted(top5_py)})")

    # so voi Phase 1 (F32 runner) neu co, de thay CHINH XAC bao nhieu nhiễu la do TQ33
    try:
        _, _, _, _, layer_outs_f32, final_h_f32, logits_f32 = load_runner_bin(F32_RUNNER_OUT)
        re_logits_vs_f32runner = rel_err(logits, logits_f32)
        print(f"\n(doi chieu) TQ33 runner vs F32 runner (Phase 1, cung may, khac kernel) "
              f"logits rel_err = {re_logits_vs_f32runner:.3e}")
        print(f"{'layer':>6} {'TQ33 vs F32-runner rel_err':>28}")
        worst = (0, -1.0)
        for l in range(n_layer):
            re = rel_err(layer_outs[l], layer_outs_f32[l])
            if re > worst[1]:
                worst = (l, re)
            print(f"{l:>6} {re:>28.3e}")
        print(f"\nworst layer (nhieu tich luy TQ33 lon nhat): layer {worst[0]}  rel_err={worst[1]:.3e}")
    except FileNotFoundError:
        print("\n(khong tim thay runner_output_f32.bin cua Phase 1 de doi chieu truc tiep)")

    print(f"\n{'='*70}")
    ok = top1_tq == top1_py
    print(f"PHAN QUYET GIAI DOAN 2 (dung/sai): top-1 token {'KHOP' if ok else 'LECH'} "
          f"— rel_err logits so oracle = {re_logits:.3e}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
