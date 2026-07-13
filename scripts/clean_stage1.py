#!/usr/bin/env python3
"""Bước 2, stage 1 — cheap deterministic cleaning of the merged VI-JA corpus.

Runs BEFORE the expensive LaBSE semantic step so LaBSE only scores survivors.
Reads every line-aligned source, tags provenance, normalizes, and applies
order-sensitive filters, counting drops at each gate. Writes parallel output:
  data/interim/stage1.ja / stage1.vi / stage1.src

Normalization:
  - Japanese: Unicode NFKC (folds half/full-width kana + width variants).
  - Vietnamese: Unicode NFC (canonical composed diacritics).
  - strip control chars, collapse whitespace.

Filters (in order): empty, ja==vi copy-through, char-length bounds,
length-ratio band, script check (ja must contain Japanese; vi must not),
exact dedup on (ja,vi), near-dup on a normalized key.

Dedup uses 8-byte blake2b hashes kept in RAM sets to bound memory on ~8M rows.
"""
import sys
import unicodedata
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "interim"

# (name, ja_path, vi_path)
SOURCES = []
_opus = ROOT / "data" / "raw" / "opus"
for corpus in ["OpenSubtitles", "CCMatrix", "MultiCCAligned", "TED2020", "WikiMatrix", "Tatoeba"]:
    # NB: build extensions by string, not Path.with_suffix — the ".ja-vi.ja"
    # double extension would make with_suffix(".ja") collapse to ".ja".
    d = _opus / corpus
    SOURCES.append((f"opus_{corpus}", d / f"{corpus}.ja-vi.ja", d / f"{corpus}.ja-vi.vi"))
SOURCES.append(("pivot", ROOT / "data/interim/pivot/pivot.ja", ROOT / "data/interim/pivot/pivot.vi"))
SOURCES.append(("synth_jlpt", ROOT / "data/synthetic/jlpt.ja", ROOT / "data/synthetic/jlpt.vi"))

# hiragana/katakana, katakana phonetic ext, CJK (+ext A), compat ideographs,
# half-width katakana — the scripts that mark text as Japanese.
_JP = re.compile(
    "[぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)
CONTROL = dict.fromkeys(range(0, 32))  # drop control chars incl. \n; \t handled first
_ws = re.compile(r"\s+")
_nonword = re.compile(r"[^\w぀-ヿ一-鿿]+", re.UNICODE)

MIN_JA, MAX_JA = 2, 250
MIN_VI, MAX_VI = 2, 500
RATIO_LO, RATIO_HI = 0.4, 9.0


def has_japanese(s):
    return _JP.search(s) is not None


def clean_text(s, lang):
    s = s.replace("\t", " ")
    s = s.translate(CONTROL)
    if lang == "ja":
        s = unicodedata.normalize("NFKC", s)
    else:
        s = unicodedata.normalize("NFC", s)
    return _ws.sub(" ", s).strip()


def h8(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def nearkey(ja, vi):
    k = _nonword.sub("", ja.lower()) + "\x00" + _nonword.sub("", vi.lower())
    return h8(k)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    stats = Counter()
    per_src_in = Counter()
    per_src_out = Counter()
    seen_exact = set()
    seen_near = set()

    f_ja = (OUT / "stage1.ja").open("w", encoding="utf-8")
    f_vi = (OUT / "stage1.vi").open("w", encoding="utf-8")
    f_src = (OUT / "stage1.src").open("w", encoding="utf-8")

    for name, ja_path, vi_path in SOURCES:
        if not ja_path.exists() or not vi_path.exists():
            print(f"WARN missing source {name}", file=sys.stderr)
            continue
        with open(ja_path, encoding="utf-8", errors="replace") as fj, \
             open(vi_path, encoding="utf-8", errors="replace") as fv:
            for ja_raw, vi_raw in zip(fj, fv):
                stats["read"] += 1
                per_src_in[name] += 1
                ja = clean_text(ja_raw, "ja")
                vi = clean_text(vi_raw, "vi")

                if not ja or not vi:
                    stats["empty"] += 1
                    continue
                if ja == vi:
                    stats["identical"] += 1
                    continue
                if not (MIN_JA <= len(ja) <= MAX_JA):
                    stats["ja_len"] += 1
                    continue
                if not (MIN_VI <= len(vi) <= MAX_VI):
                    stats["vi_len"] += 1
                    continue
                ratio = len(vi) / len(ja)
                if not (RATIO_LO <= ratio <= RATIO_HI):
                    stats["ratio"] += 1
                    continue
                if not has_japanese(ja):
                    stats["ja_no_jp"] += 1
                    continue
                if has_japanese(vi):
                    stats["vi_has_jp"] += 1
                    continue

                he = h8(ja + "\x00" + vi)
                if he in seen_exact:
                    stats["dup_exact"] += 1
                    continue
                hn = nearkey(ja, vi)
                if hn in seen_near:
                    stats["dup_near"] += 1
                    continue
                seen_exact.add(he)
                seen_near.add(hn)

                f_ja.write(ja + "\n")
                f_vi.write(vi + "\n")
                f_src.write(name + "\n")
                stats["kept"] += 1
                per_src_out[name] += 1

    f_ja.close(); f_vi.close(); f_src.close()

    print("=== drop counts ===")
    for k in ["read", "empty", "identical", "ja_len", "vi_len", "ratio",
              "ja_no_jp", "vi_has_jp", "dup_exact", "dup_near", "kept"]:
        print(f"  {k:<12} {stats[k]:>10,}")
    print("=== per-source (in -> kept) ===")
    for name, _, _ in SOURCES:
        print(f"  {name:<22} {per_src_in[name]:>9,} -> {per_src_out[name]:>9,}")
    print(f"\nstage1 kept {stats['kept']:,} pairs -> data/interim/stage1.{{ja,vi,src}}")


if __name__ == "__main__":
    main()
