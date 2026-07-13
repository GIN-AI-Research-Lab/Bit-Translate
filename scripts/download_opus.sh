#!/usr/bin/env bash
# Download OPUS vi-ja parallel corpora.
# Verified working URLs as of 2026-07 (opus.nlpl.eu object storage).
# CCAligned has no direct vi-ja pair on OPUS; MultiCCAligned is used instead.
#
# License note: TED2020 is CC BY-NC-ND (non-commercial, no-derivatives) — only
# include it if the project is scoped to personal/research use (see project
# memory "data-license-scope"). All others carry no NC/ND restriction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/../data/raw/opus"
mkdir -p "$RAW_DIR"

# name -> "url license"
CORPORA_OpenSubtitles="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/ja-vi.txt.zip"
CORPORA_CCMatrix="https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/ja-vi.txt.zip"
CORPORA_WikiMatrix="https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/ja-vi.txt.zip"
CORPORA_Tatoeba="https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/ja-vi.txt.zip"
CORPORA_MultiCCAligned="https://object.pouta.csc.fi/OPUS-MultiCCAligned/v1.1/moses/ja-vi.txt.zip"
CORPORA_TED2020="https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-vi.txt.zip"
CORPORA_MultiHPLT="https://object.pouta.csc.fi/OPUS-MultiHPLT/v2/moses/ja-vi.txt.zip"
CORPORA_KDE4="https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/ja-vi.txt.zip"
CORPORA_ALT="https://object.pouta.csc.fi/OPUS-ALT/v20191206/moses/ja-vi.txt.zip"

NAMES="OpenSubtitles CCMatrix WikiMatrix Tatoeba MultiCCAligned TED2020 MultiHPLT KDE4 ALT"

for name in $NAMES; do
  var="CORPORA_${name}"
  url="${!var}"
  dest_dir="$RAW_DIR/$name"
  mkdir -p "$dest_dir"
  zip_path="$dest_dir/ja-vi.txt.zip"

  echo "== $name =="
  if [ -f "$dest_dir/.done" ]; then
    echo "already downloaded+extracted, skipping"
    continue
  fi

  if [ ! -f "$zip_path" ]; then
    wget -c -q --show-progress -O "$zip_path" "$url"
  fi
  python3 -m zipfile -e "$zip_path" "$dest_dir/"
  rm -f "$zip_path"
  touch "$dest_dir/.done"
done

echo
echo "=== Pair counts ==="
for name in $NAMES; do
  dest_dir="$RAW_DIR/$name"
  ja_file=$(find "$dest_dir" -maxdepth 1 -name "*.ja" | head -1)
  if [ -n "$ja_file" ]; then
    count=$(wc -l < "$ja_file")
    echo "$name: $count pairs"
  else
    echo "$name: NO .ja FILE FOUND (check archive structure)"
    find "$dest_dir" -maxdepth 1 -type f
  fi
done
