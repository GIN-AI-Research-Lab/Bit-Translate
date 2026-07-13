#!/usr/bin/env python3
"""Bridge VI-JA pairs via shared English sentences (Bước 1, item 2):
  EN-VI (PhoMT, MTet) x EN-JA (JParaCrawl) joined on normalized English text.

Streaming implementation (no DuckDB, no temp spill). The WSL root fs lives on
an ext4.vhdx backed by the chronically-full Windows C: drive, so a DuckDB hash
join that spills tens of GB to temp fills C: and gets killed (see project
memory "wsl-disk-reclaim"). Instead:

  1. Load the small EN-VI side (~7M rows) into an in-RAM dict:
     normalized_english -> vietnamese. First value per key wins.
  2. Stream the large EN-JA side (JParaCrawl, ~25.7M rows) line by line,
     normalize its English, look it up, and write matches straight to the
     output files. One EN key emits at most one pivot pair.

Peak memory is bounded by the EN-VI dict (~2-4 GB); disk use is only the
output. Exact-match pivot after light normalization (whitespace + casefold);
real recall is only known once it runs. Quality filtering (LaBSE etc.) is a
later Bước 2 step — this only performs the join.

Output: data/interim/pivot/pivot.vi / pivot.ja (+ pivot.en for traceability).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "interim" / "pivot"


def normalize(text):
    # collapse all whitespace + casefold; used only as the join key.
    return " ".join(text.strip().lower().split())


def find_parallel_files(base, en_ext, other_ext):
    """Find all (en_file, other_file) pairs sharing a basename under base."""
    pairs = []
    if not base.exists():
        return pairs
    for en_file in sorted(base.rglob(f"*.{en_ext}")):
        other_file = en_file.with_suffix(f".{other_ext}")
        if other_file.exists():
            pairs.append((en_file, other_file))
    return pairs


def iter_parallel(en_file, other_file):
    """Yield (en_line, other_line) from two line-aligned bitext files."""
    with open(en_file, encoding="utf-8", errors="replace") as ef, \
         open(other_file, encoding="utf-8", errors="replace") as of:
        for en_line, other_line in zip(ef, of):
            yield en_line.rstrip("\n"), other_line.rstrip("\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # PhoMT ships both "detokenization" (natural text) and "tokenization"
    # (space-segmented) copies. Only natural text can share a join key with
    # JParaCrawl's raw web text.
    en_vi_pairs = (
        find_parallel_files(RAW / "phomt" / "PhoMT" / "detokenization", "en", "vi")
        + find_parallel_files(RAW / "mtet", "en", "vi")
    )
    en_ja_pairs = find_parallel_files(RAW / "jparacrawl", "en", "ja")

    if not en_vi_pairs:
        sys.exit("no EN-VI files found under data/raw/phomt or data/raw/mtet")
    if not en_ja_pairs:
        sys.exit("no EN-JA files found under data/raw/jparacrawl")

    print("EN-VI sources:", [str(p[0]) for p in en_vi_pairs], flush=True)
    print("EN-JA sources:", [str(p[0]) for p in en_ja_pairs], flush=True)

    # 1. Build normalized_en -> vi dict from the EN-VI side.
    en_to_vi = {}
    n_read = 0
    for en_file, vi_file in en_vi_pairs:
        for en, vi in iter_parallel(en_file, vi_file):
            n_read += 1
            if not en.strip() or not vi.strip():
                continue
            key = normalize(en)
            if key and key not in en_to_vi:
                en_to_vi[key] = vi
    print(f"EN-VI: read {n_read} rows -> {len(en_to_vi)} unique EN keys", flush=True)

    # 2. Stream the EN-JA side, matching against the dict.
    emitted = set()
    n_ja = 0
    n_match = 0
    with open(OUT_DIR / "pivot.en", "w", encoding="utf-8") as f_en, \
         open(OUT_DIR / "pivot.vi", "w", encoding="utf-8") as f_vi, \
         open(OUT_DIR / "pivot.ja", "w", encoding="utf-8") as f_ja:
        for en_file, ja_file in en_ja_pairs:
            for en, ja in iter_parallel(en_file, ja_file):
                n_ja += 1
                if n_ja % 5_000_000 == 0:
                    print(f"  scanned {n_ja} EN-JA rows, {n_match} matches so far", flush=True)
                if not en.strip() or not ja.strip():
                    continue
                key = normalize(en)
                vi = en_to_vi.get(key)
                if vi is None or key in emitted:
                    continue
                emitted.add(key)
                n_match += 1
                f_en.write(en + "\n")
                f_vi.write(vi + "\n")
                f_ja.write(ja + "\n")

    print(f"scanned {n_ja} EN-JA rows total", flush=True)
    print(f"matched {n_match} pivot VI-JA pairs -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
