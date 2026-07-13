#!/usr/bin/env python3
"""Bước 2, stage 2 — language-ID filter with fastText lid.176.

Stage 1's script check confirms the ja side contains Japanese and the vi side
does not — but it can't tell Vietnamese from other Latin-script languages.
Crawled corpora (CCMatrix/MultiCCAligned) leak English (or other) text into
the vi field; those pass stage 1 but are useless as VI-JA pairs. fastText
distinguishes vi from en/fr/id/... reliably.

Keep a pair only if:
  - vi side top language is 'vi' with prob >= VI_MIN, and
  - ja side top language is 'ja' with prob >= JA_MIN.

Reads data/interim/stage1.{ja,vi,src}; writes data/interim/stage2.{ja,vi,src}.
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as _np

# fasttext-wheel 0.9.2 calls np.array(probs, copy=False), which raises under
# numpy>=2 when a copy is unavoidable. Map copy=False -> copy=None (numpy 2's
# "copy only if needed") so predict() works without downgrading numpy.
_orig_array = _np.array
def _patched_array(obj, *args, **kwargs):
    if kwargs.get("copy", True) is False:
        kwargs["copy"] = None
    return _orig_array(obj, *args, **kwargs)
_np.array = _patched_array

import fasttext

ROOT = Path(__file__).parent.parent
INTERIM = ROOT / "data" / "interim"
MODEL = ROOT / "models" / "lid.176.bin"

# Soft rule: reject a side only when fastText is CONFIDENT it is some other
# language (top-1 != target with prob >= REJECT_CONF). Short/all-caps
# Vietnamese subtitle lines often score vi at only 0.3-0.5 or get a
# low-confidence wrong guess; dropping those would lose good data, and LaBSE
# (stage 3) still catches true cross-language mismatches by meaning.
REJECT_CONF = 0.50


def main():
    fasttext.FastText.eprint = lambda *a, **k: None  # silence warning
    model = fasttext.load_model(str(MODEL))

    ja_in = (INTERIM / "stage1.ja").open(encoding="utf-8")
    vi_in = (INTERIM / "stage1.vi").open(encoding="utf-8")
    src_in = (INTERIM / "stage1.src").open(encoding="utf-8")
    ja_out = (INTERIM / "stage2.ja").open("w", encoding="utf-8")
    vi_out = (INTERIM / "stage2.vi").open("w", encoding="utf-8")
    src_out = (INTERIM / "stage2.src").open("w", encoding="utf-8")

    stats = Counter()
    per_src_out = Counter()
    vi_wrong = Counter()  # what the vi field actually was, when rejected

    for ja, vi, src in zip(ja_in, vi_in, src_in):
        ja, vi, src = ja.rstrip("\n"), vi.rstrip("\n"), src.rstrip("\n")
        stats["read"] += 1

        # fastText wants no newlines; predict top-1
        vi_lab, vi_prob = model.predict(vi.replace("\n", " "))
        ja_lab, ja_prob = model.predict(ja.replace("\n", " "))
        vi_lang = vi_lab[0].replace("__label__", "")
        ja_lang = ja_lab[0].replace("__label__", "")

        if vi_lang != "vi" and vi_prob[0] >= REJECT_CONF:
            stats["vi_not_vi"] += 1
            vi_wrong[vi_lang] += 1
            continue
        if ja_lang != "ja" and ja_prob[0] >= REJECT_CONF:
            stats["ja_not_ja"] += 1
            continue

        ja_out.write(ja + "\n")
        vi_out.write(vi + "\n")
        src_out.write(src + "\n")
        per_src_out[src] += 1
        stats["kept"] += 1

    for f in (ja_in, vi_in, src_in, ja_out, vi_out, src_out):
        f.close()

    print("=== stage2 langid ===")
    for k in ["read", "vi_not_vi", "ja_not_ja", "kept"]:
        print(f"  {k:<12} {stats[k]:>10,}")
    print("=== top languages found in rejected vi field ===")
    for lang, n in vi_wrong.most_common(10):
        print(f"  {lang:<6} {n:>9,}")
    print("=== per-source kept ===")
    for src, n in sorted(per_src_out.items()):
        print(f"  {src:<22} {n:>9,}")
    print(f"\nstage2 kept {stats['kept']:,} -> data/interim/stage2.{{ja,vi,src}}")


if __name__ == "__main__":
    main()
