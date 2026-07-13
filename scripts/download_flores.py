#!/usr/bin/env python3
"""Download FLORES-200/FLORES+ vie_Latn and jpn_Jpan splits (dev + devtest)
as the neutral eval benchmark for Bước 2/5. Gated dataset — requires an HF
account that has clicked "Agree" on the dataset page, and an HF token
available via `huggingface-cli login` or the HF_TOKEN env var.

One-time manual steps (cannot be done by Claude Code — needs a browser):
  1. Create/log into a Hugging Face account: https://huggingface.co/join
  2. Visit https://huggingface.co/datasets/openlanguagedata/flores_plus
     and click "Agree and access repository" (auto-approved).
  3. Create a read token: https://huggingface.co/settings/tokens
  4. Run `huggingface-cli login` (paste the token) or `export HF_TOKEN=...`
"""
import json
from pathlib import Path

from datasets import load_dataset

DEST_DIR = Path(__file__).parent.parent / "data" / "clean" / "flores"


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    vie = load_dataset("openlanguagedata/flores_plus", "vie_Latn")
    jpn = load_dataset("openlanguagedata/flores_plus", "jpn_Jpan")

    for split in ("dev", "devtest"):
        vie_by_id = {row["id"]: row["text"] for row in vie[split]}
        jpn_by_id = {row["id"]: row["text"] for row in jpn[split]}
        shared_ids = sorted(set(vie_by_id) & set(jpn_by_id))

        vi_path = DEST_DIR / f"flores.{split}.vi"
        ja_path = DEST_DIR / f"flores.{split}.ja"
        vi_path.write_text("\n".join(vie_by_id[i] for i in shared_ids) + "\n", encoding="utf-8")
        ja_path.write_text("\n".join(jpn_by_id[i] for i in shared_ids) + "\n", encoding="utf-8")
        print(f"{split}: {len(shared_ids)} aligned sentence pairs -> {vi_path.name}/{ja_path.name}")


if __name__ == "__main__":
    main()
