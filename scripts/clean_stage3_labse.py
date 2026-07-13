#!/usr/bin/env python3
"""Bước 2, stage 3 — LaBSE semantic-alignment scoring on GPU.

For every surviving pair, embed the ja and vi sentences with LaBSE (built for
bitext mining) and score cosine similarity. Low similarity => the two sides are
not translations (mining noise). We only WRITE scores here; the threshold is
chosen in stage 4 via an agent audit, not hard-coded blind.

Design for a ~2h GPU run on an 8GB card with a flaky host disk:
  - stream scores to disk batch-by-batch (never hold 768-d vectors; only the
    scalar score per line survives in memory),
  - RESUMABLE: on restart, skip the pairs already scored (line count of the
    existing stage3.scores), so a crash/OOM/disk-full loses at most one batch,
  - cap max_seq_length (sentences are short after stage-1 length filter).

Input:  data/interim/stage2.{ja,vi,src}
Output: data/interim/stage3.scores  (one float per line, aligned to input)
        data/interim/stage3.hist    (histogram + per-source stats, written at end)
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
INTERIM = ROOT / "data" / "interim"
SCORES = INTERIM / "stage3.scores"
BATCH = 256   # keep peak VRAM well under the 8GB card (shared with desktop)
MAX_SEQ = 128


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def main():
    ja = read_lines(INTERIM / "stage2.ja")
    vi = read_lines(INTERIM / "stage2.vi")
    assert len(ja) == len(vi), "input files misaligned"
    n = len(ja)

    done = 0
    if SCORES.exists():
        with open(SCORES) as f:
            done = sum(1 for _ in f)
        if done > n:
            sys.exit(f"stage3.scores has {done} lines > {n} inputs; delete it to restart")
        print(f"resuming: {done:,}/{n:,} already scored", flush=True)

    print(f"scoring {n - done:,} remaining pairs with LaBSE (batch={BATCH})...", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
    model.max_seq_length = MAX_SEQ
    # FP16: ~4x faster on this Ampere card (Tensor Cores), and more work-per-watt
    # under the host's SW power cap. Scores differ from fp32 by <0.0011 — far
    # below any threshold we'd pick, so mixing fp16/fp32 across a resume is fine.
    model.half()

    fout = open(SCORES, "a", encoding="utf-8", buffering=1 << 20)
    try:
        for i in range(done, n, BATCH):
            j = min(i + BATCH, n)
            e_ja = model.encode(ja[i:j], batch_size=BATCH, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
            e_vi = model.encode(vi[i:j], batch_size=BATCH, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
            batch_scores = np.sum(e_ja * e_vi, axis=1)
            fout.write("\n".join(f"{s:.4f}" for s in batch_scores) + "\n")
            if (i // BATCH) % 200 == 0:
                pct = 100 * j / n
                print(f"  {j:,}/{n:,} ({pct:.1f}%)", flush=True)
    finally:
        fout.close()

    # summary pass over the full score file
    scores = np.loadtxt(SCORES, dtype=np.float32)
    src = read_lines(INTERIM / "stage2.src")
    edges = np.arange(0.0, 1.01, 0.05)
    hist, _ = np.histogram(scores, bins=edges)
    by_src = defaultdict(list)
    for s, sc in zip(src, scores):
        by_src[s].append(sc)

    lines = ["=== score histogram (all pairs) ==="]
    for k in range(len(hist)):
        lines.append(f"  [{edges[k]:.2f},{edges[k+1]:.2f})  {hist[k]:>9,}")
    lines.append("=== per-source mean / median / frac>=0.7 ===")
    for s in sorted(by_src):
        arr = np.array(by_src[s])
        lines.append(f"  {s:<22} mean={arr.mean():.3f} med={np.median(arr):.3f} "
                     f">=0.7: {100*np.mean(arr >= 0.7):.0f}%  (n={len(arr):,})")
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8):
        lines.append(f"  threshold {thr:.2f} -> keep {int(np.sum(scores >= thr)):,}")
    report = "\n".join(lines)
    (INTERIM / "stage3.hist").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwrote {len(scores):,} scores -> {SCORES}", flush=True)


if __name__ == "__main__":
    main()
