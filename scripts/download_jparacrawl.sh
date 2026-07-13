#!/usr/bin/env bash
# Download JParaCrawl v3.0 (EN-JA, ~22M pairs, NTT).
# License: research-only, non-commercial (including any model trained on
# derived data). Only include in training if project stays personal/research
# scope — see project memory "data-license-scope".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/../data/raw/jparacrawl"
mkdir -p "$DEST_DIR"

URL="https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/en/3.0/bitext/en-ja.tar.gz"
TAR_PATH="$DEST_DIR/en-ja.tar.gz"

if [ -f "$DEST_DIR/.done" ]; then
  echo "already downloaded+extracted, skipping"
else
  wget -c -q --show-progress -O "$TAR_PATH" "$URL"
  tar -xzf "$TAR_PATH" -C "$DEST_DIR"
  rm -f "$TAR_PATH"

  # The archive ships one combined file: url<TAB>url<TAB>bicleaner_score<TAB>en<TAB>ja.
  # Split into plain .en/.ja (+ .scores sidecar) to match the other corpora's
  # layout. Use `cut`, not awk — this corpus has some invalid UTF-8 byte
  # sequences (common in crawled web text) that made gawk's multibyte-aware
  # field splitting merge multiple records into one giant corrupted line.
  # `cut` treats the tab as a plain byte delimiter, so it isn't affected.
  bitext="$DEST_DIR/en-ja/en-ja.bicleaner05.txt"
  cut -f4 "$bitext" > "$DEST_DIR/en-ja/en-ja.en"
  cut -f5 "$bitext" > "$DEST_DIR/en-ja/en-ja.ja"
  cut -f3 "$bitext" > "$DEST_DIR/en-ja/en-ja.scores"

  touch "$DEST_DIR/.done"
fi

echo "=== Pair count ==="
en_file=$(find "$DEST_DIR" -name "*.en" | head -1)
if [ -n "$en_file" ]; then
  wc -l < "$en_file"
else
  find "$DEST_DIR" -maxdepth 2 -type f
fi
