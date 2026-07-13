#!/usr/bin/env python3
"""Download PhoMT (EN-VI, VinAI) from the gated vinai/PhoMT HF dataset.

One-time manual steps (cannot be done by Claude Code — needs a browser):
  1. Create/log into a Hugging Face account: https://huggingface.co/join
  2. Visit https://huggingface.co/datasets/vinai/PhoMT and click through the
     "Agree and access repository" gate (asks to share contact info).
  3. Create a read token: https://huggingface.co/settings/tokens
  4. Run `huggingface-cli login` (paste the token) or `export HF_TOKEN=...`

License: research/educational use only, no redistribution of the dataset in
original or modified form. Must cite the EMNLP 2021 paper. Only include in
training if project stays personal/research scope (see project memory
"data-license-scope").
"""
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DEST_DIR = Path(__file__).parent.parent / "data" / "raw" / "phomt"


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if (DEST_DIR / ".done").exists():
        print("already downloaded+extracted, skipping")
        return

    zip_path = hf_hub_download("vinai/PhoMT", "PhoMT.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DEST_DIR)

    (DEST_DIR / ".done").touch()
    print("extracted to", DEST_DIR)
    for p in sorted(DEST_DIR.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(DEST_DIR))


if __name__ == "__main__":
    main()
