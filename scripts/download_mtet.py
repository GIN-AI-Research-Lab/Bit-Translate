#!/usr/bin/env python3
"""Download MTet (EN-VI) from the phongmt184172/mtet HF mirror and emit
parallel .en/.vi files. The official albertvillanova/mtet repo only ships a
(now-unsupported) loading script pointing at the original GCS bucket, so we
use this mirror's parquet files instead.

The mirror stores each pair twice (once per translation direction, as
prompt/response instruction data) via a `translation` dict with source/target
keys whose language flips depending on direction. We dedupe on the unordered
pair and split into en/vi by script heuristic (Vietnamese diacritics).

License: no official commercial-use grant found for MTet; treat as
non-commercial/research-only (see project memory "data-license-scope").
"""
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

DEST_DIR = Path(__file__).parent.parent / "data" / "raw" / "mtet"
REPO = "phongmt184172/mtet"

# Vietnamese-specific letters that never occur in English text.
VI_CHARS = re.compile(
    "[à-ãè-êìíò-õùúăđ"
    "ĩũơưẠ-ỹ]",
    re.IGNORECASE,
)


def is_vietnamese(text):
    return bool(VI_CHARS.search(text))


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    done_marker = DEST_DIR / ".done"
    if done_marker.exists():
        print("already processed, skipping")
        return

    api = HfApi()
    files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet")]
    print(f"downloading {len(files)} parquet shards...")

    seen = set()
    en_out, vi_out = [], []
    ambiguous = 0

    for i, f in enumerate(files):
        path = hf_hub_download(REPO, f, repo_type="dataset")
        df = pd.read_parquet(path, columns=["translation"])
        for row in df["translation"]:
            a, b = row["source"], row["target"]
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)

            a_vi, b_vi = is_vietnamese(a), is_vietnamese(b)
            if a_vi and not b_vi:
                vi, en = a, b
            elif b_vi and not a_vi:
                vi, en = b, a
            else:
                ambiguous += 1
                continue
            en_out.append(en.strip())
            vi_out.append(vi.strip())
        print(f"  shard {i+1}/{len(files)} done, {len(en_out)} pairs so far", file=sys.stderr)

    (DEST_DIR / "mtet.en").write_text("\n".join(en_out) + "\n", encoding="utf-8")
    (DEST_DIR / "mtet.vi").write_text("\n".join(vi_out) + "\n", encoding="utf-8")
    done_marker.touch()

    print(f"done: {len(en_out)} unique pairs, {ambiguous} ambiguous/skipped")


if __name__ == "__main__":
    main()
