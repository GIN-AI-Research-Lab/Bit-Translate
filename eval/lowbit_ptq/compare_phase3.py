# -*- coding: utf-8 -*-
r"""TQ33 SPEED-100 — so khớp runner FAST (TQ33 + int8 embed/head) với oracle PyTorch.
Giống compare_phase2.py nhưng nhận đường dẫn qua argv để chạy được trên nhiều prompt.

Dùng: python compare_phase3.py <oracle_dir> <runner_out_bin>
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def rel_err(a, b):
    num = np.sqrt(np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    den = np.sqrt(np.sum(b.astype(np.float64) ** 2)) + 1e-30
    return float(num / den)


def load_runner_bin(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=4)
        seq_len, hidden, vocab, n_layer = [int(x) for x in hdr]
        layer_outs = []
        for _ in range(n_layer):
            layer_outs.append(np.fromfile(f, dtype=np.float32,
                                          count=seq_len * hidden).reshape(seq_len, hidden))
        final_h = np.fromfile(f, dtype=np.float32, count=seq_len * hidden).reshape(seq_len, hidden)
        logits = np.fromfile(f, dtype=np.float32, count=seq_len * vocab).reshape(seq_len, vocab)
    return seq_len, hidden, vocab, n_layer, layer_outs, final_h, logits


def main():
    oracle_dir = sys.argv[1]
    runner_bin = sys.argv[2]
    seq_len, hidden, vocab, n_layer, layer_outs, final_h, logits = load_runner_bin(runner_bin)
    print(f"[{os.path.basename(oracle_dir)}] seq_len={seq_len}")

    print(f"{'layer':>6} {'rel_err (L2)':>14} {'max_abs_diff':>14}")
    show = {0, 1, 13, 26, 27}
    for l in range(n_layer):
        p = os.path.join(oracle_dir, f"hidden_l{l}.npy")
        if not os.path.exists(p):
            continue
        oracle = np.load(p)
        re = rel_err(layer_outs[l], oracle)
        md = float(np.max(np.abs(layer_outs[l].astype(np.float64) - oracle.astype(np.float64))))
        if l in show:
            print(f"{l:>6} {re:>14.3e} {md:>14.3e}")

    oracle_final = np.load(os.path.join(oracle_dir, "hidden_final.npy"))
    print(f"final (post-norm)  rel_err = {rel_err(final_h, oracle_final):.3e}")

    oracle_logits = np.load(os.path.join(oracle_dir, "logits.npy"))
    re_logits = rel_err(logits, oracle_logits)
    print(f"logits             rel_err = {re_logits:.3e}")

    n_top1 = 0
    n_top5 = 0
    for t in range(seq_len):
        t1r, t1 = int(np.argmax(oracle_logits[t])), int(np.argmax(logits[t]))
        s5r = set(np.argsort(-oracle_logits[t])[:5].tolist())
        s5 = set(np.argsort(-logits[t])[:5].tolist())
        n_top1 += (t1 == t1r)
        n_top5 += len(s5 & s5r)
    top1_last_r = int(np.argmax(oracle_logits[-1]))
    top1_last = int(np.argmax(logits[-1]))
    s5r = set(np.argsort(-oracle_logits[-1])[:5].tolist())
    s5 = set(np.argsort(-logits[-1])[:5].tolist())
    print(f"top-1 vi tri CUOI: runner={top1_last} oracle={top1_last_r} "
          f"{'KHOP' if top1_last == top1_last_r else 'LECH'}; top-5 overlap={len(s5 & s5r)}/5")
    print(f"top-1 khop {n_top1}/{seq_len} vi tri; top-5 overlap trung binh {n_top5/seq_len:.2f}/5")


if __name__ == "__main__":
    main()
