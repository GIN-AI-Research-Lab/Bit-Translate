# -*- coding: utf-8 -*-
r"""TQ33 Runner — Giai đoạn 1, bước 3: so khớp qwen3_runner.exe (C, float32 thuần) với
oracle PyTorch (ref_forward_pytorch.py), per-layer + logits cuối, in rel err từng layer
để nếu lệch thì biết CHÍNH XÁC layer nào sai (không chỉ biết "logits cuối sai")."""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ORACLE_DIR = r"D:\Bit-Translate-data\tq33_runner\oracle"
RUNNER_OUT = r"D:\Bit-Translate-data\tq33_runner\runner_output_f32.bin"


def rel_err(a, b):
    num = np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    den = np.sqrt(np.sum(b.astype(np.float64) ** 2)) + 1e-30
    return float(num / den)


def main():
    with open(RUNNER_OUT, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        seq_len, hidden, vocab, n_layer = [int(x) for x in hdr]
        print(f"header: seq_len={seq_len} hidden={hidden} vocab={vocab} n_layer={n_layer}")

        layer_outs = []
        for l in range(n_layer):
            arr = np.fromfile(f, dtype=np.float32, count=seq_len * hidden).reshape(seq_len, hidden)
            layer_outs.append(arr)
        final_h = np.fromfile(f, dtype=np.float32, count=seq_len * hidden).reshape(seq_len, hidden)
        logits = np.fromfile(f, dtype=np.float32, count=seq_len * vocab).reshape(seq_len, vocab)

    print(f"\n{'layer':>6} {'rel_err (L2)':>14} {'max_abs_diff':>14}")
    worst = (0, -1.0)
    for l in range(n_layer):
        oracle = np.load(f"{ORACLE_DIR}\\hidden_l{l}.npy")
        assert oracle.shape == layer_outs[l].shape, (oracle.shape, layer_outs[l].shape)
        re = rel_err(layer_outs[l], oracle)
        md = float(np.max(np.abs(layer_outs[l].astype(np.float64) - oracle.astype(np.float64))))
        print(f"{l:>6} {re:>14.3e} {md:>14.3e}")
        if re > worst[1]:
            worst = (l, re)

    oracle_final = np.load(f"{ORACLE_DIR}\\hidden_final.npy")
    re_final = rel_err(final_h, oracle_final)
    print(f"\nfinal (post-norm)  rel_err = {re_final:.3e}")

    oracle_logits = np.load(f"{ORACLE_DIR}\\logits.npy")
    re_logits = rel_err(logits, oracle_logits)
    md_logits = float(np.max(np.abs(logits.astype(np.float64) - oracle_logits.astype(np.float64))))
    print(f"logits             rel_err = {re_logits:.3e}   max_abs_diff = {md_logits:.3e}")

    top1_c = int(np.argmax(logits[-1]))
    top1_py = int(np.argmax(oracle_logits[-1]))
    print(f"\ntop-1 token (C runner)  @ vi tri cuoi: {top1_c}")
    print(f"top-1 token (PyTorch)   @ vi tri cuoi: {top1_py}")
    print(f"top-1 KHOP: {top1_c == top1_py}")

    print(f"\nworst layer: {worst[0]}  rel_err={worst[1]:.3e}")

    ok = re_logits < 1e-3 and top1_c == top1_py
    print(f"\n{'='*60}")
    print(f"PHAN QUYET GIAI DOAN 1: {'DAT — rel_err<1e-3 va top-1 khop' if ok else 'KHONG DAT'}")
    print(f"{'='*60}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
