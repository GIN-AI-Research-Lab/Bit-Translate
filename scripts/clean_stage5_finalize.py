#!/usr/bin/env python3
"""Bước 2, stage 5 — apply the chosen LaBSE threshold and split train/dev.

Reads the stage-2 survivors + their stage-3 LaBSE scores, keeps pairs with
score >= threshold, and writes the final clean corpus with a held-out dev set.
FLORES-200 (data/clean/flores) stays the neutral TEST set and is never mixed in.

Usage: clean_stage5_finalize.py --threshold 0.70 [--dev-size 5000] [--seed 42]

Output (parallel line-aligned files):
  data/clean/train.ja / train.vi / train.src
  data/clean/dev.ja   / dev.vi   / dev.src
The dev set is a random stratified-by-source holdout so both splits match the
corpus mix; global dedup already happened in stage 1, so splits are disjoint.
"""
import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

# Content junk that LaBSE lets through (semantically it still "matches"):
_ARTIFACT = re.compile(r"X[I]?NUMX|\bNUMX\b")          # bad text-extraction placeholders
_ADULT = re.compile(r"porn|xvideos|voyeur|sex tape|\bjav\b|escort", re.I)
_URL = re.compile(r"https?://|www\.|@[a-z0-9.]+\.(?:com|net|org)", re.I)


def is_junk(ja, vi):
    for rx in (_ARTIFACT, _ADULT, _URL):
        if rx.search(ja) or rx.search(vi):
            return True
    return False


def latin_frac(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 0x300) / len(letters)

ROOT = Path(__file__).parent.parent
INTERIM = ROOT / "data" / "interim"
CLEAN = ROOT / "data" / "clean"

# Per-source LaBSE thresholds, chosen from the stage-3 band audit:
#   - OpenSubtitles: low (0.55) — subtitle translations are loose/condensed and
#     score low even when correct; this is our main conversational data.
#   - CCMatrix / MultiCCAligned: high (0.70) — web-mined, hallucinate plausible
#     wrong sentences up through ~0.65; junk number-pairs also sit below 0.70.
#   - TED2020: 0.62 (semi-formal talks, moderately loose).
#   - WikiMatrix: 0.68 (encyclopedic, mostly clean but mined).
#   - Tatoeba / synth_jlpt: 0.60 (already high-quality; keep almost all good).
#   - pivot: 0.72 (weak source, only ~9k pairs — cut aggressively).
SOURCE_THRESHOLDS = {
    "opus_OpenSubtitles": 0.55,
    "opus_CCMatrix": 0.70,
    "opus_MultiCCAligned": 0.70,
    "opus_TED2020": 0.62,
    "opus_WikiMatrix": 0.68,
    "opus_Tatoeba": 0.60,
    "synth_jlpt": 0.60,
    "pivot": 0.72,
}
DEFAULT_THRESHOLD = 0.70


def read_lines(p):
    with open(p, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-latin-ja", type=float, default=1.0,
                    help="drop pairs whose JA side exceeds this Latin-letter fraction "
                         "(1.0 = off; 0.6 targets product-spam/code-switch)")
    args = ap.parse_args()

    ja = read_lines(INTERIM / "stage2.ja")
    vi = read_lines(INTERIM / "stage2.vi")
    src = read_lines(INTERIM / "stage2.src")
    scores = [float(x) for x in read_lines(INTERIM / "stage3.scores")]
    assert len(ja) == len(vi) == len(src) == len(scores), "misaligned inputs"

    thr = {s: SOURCE_THRESHOLDS.get(s, DEFAULT_THRESHOLD) for s in set(src)}
    dropped = Counter()
    kept = []
    for i, s in enumerate(scores):
        if s < thr[src[i]]:
            dropped["below_threshold:" + src[i]] += 1
            continue
        if is_junk(ja[i], vi[i]):
            dropped["content_junk"] += 1
            continue
        if latin_frac(ja[i]) > args.max_latin_ja:
            dropped["latin_ja"] += 1
            continue
        kept.append(i)

    print("per-source thresholds:", json.dumps(thr, ensure_ascii=False))
    print(f"max-latin-ja: {args.max_latin_ja}")
    print(f"kept {len(kept):,} / {len(ja):,}")
    print("=== dropped ===")
    for k in sorted(dropped):
        print(f"  {k:<28} {dropped[k]:>9,}")

    # stratified dev holdout: sample dev proportionally per source
    by_src = defaultdict(list)
    for i in kept:
        by_src[src[i]].append(i)
    rng = random.Random(args.seed)
    dev_idx = set()
    total = len(kept)
    for s, idxs in by_src.items():
        share = max(1, round(args.dev_size * len(idxs) / total))
        share = min(share, len(idxs))
        dev_idx.update(rng.sample(idxs, share))
    # trim/pad to requested size deterministically
    dev_list = sorted(dev_idx)
    train_list = [i for i in kept if i not in dev_idx]

    CLEAN.mkdir(parents=True, exist_ok=True)

    def dump(split, idxs):
        with (CLEAN / f"{split}.ja").open("w", encoding="utf-8") as fj, \
             (CLEAN / f"{split}.vi").open("w", encoding="utf-8") as fv, \
             (CLEAN / f"{split}.src").open("w", encoding="utf-8") as fs:
            for i in idxs:
                fj.write(ja[i] + "\n")
                fv.write(vi[i] + "\n")
                fs.write(src[i] + "\n")

    dump("train", train_list)
    dump("dev", dev_list)

    print(f"train: {len(train_list):,}  dev: {len(dev_list):,}")
    print("=== train per-source ===")
    c = Counter(src[i] for i in train_list)
    for s, n in c.most_common():
        print(f"  {s:<22} {n:>9,}")
    print("=> data/clean/{train,dev}.{ja,vi,src}; test = data/clean/flores (FLORES-200)")


if __name__ == "__main__":
    main()
