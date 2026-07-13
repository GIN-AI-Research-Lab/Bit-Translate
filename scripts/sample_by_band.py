#!/usr/bin/env python3
"""Sample pairs by LaBSE score band (optionally filtered to one source) so a
human/agent can judge where "valid translation" turns into "misaligned" — used
to choose the stage-5 threshold instead of hard-coding it blind.
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
INTERIM = ROOT / "data" / "interim"
BANDS = [(0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.70),
         (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]


def read(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=6)
    ap.add_argument("--source", default=None, help="filter to one source tag")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ja = read(INTERIM / "stage2.ja")
    vi = read(INTERIM / "stage2.vi")
    src = read(INTERIM / "stage2.src")
    sc = [float(x) for x in read(INTERIM / "stage3.scores")]
    rng = random.Random(args.seed)

    for lo, hi in BANDS:
        idxs = [i for i in range(len(sc))
                if lo <= sc[i] < hi and (args.source is None or src[i] == args.source)]
        print(f"\n===== band [{lo:.2f},{hi:.2f})  (n={len(idxs):,}) =====")
        if not idxs:
            continue
        for i in rng.sample(idxs, min(args.per_band, len(idxs))):
            print(f"[{sc[i]:.3f}] {src[i]}")
            print(f"  JA: {ja[i]}")
            print(f"  VI: {vi[i]}")


if __name__ == "__main__":
    main()
