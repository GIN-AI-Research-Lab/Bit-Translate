#!/usr/bin/env python3
"""Bước 3 — train a shared SentencePiece tokenizer on the project's VI+JA data.

One shared 32k vocab over both languages (the model is bidirectional in one
net). Trained ONLY on our cleaned train side (not dev/test) so eval stays
honest. Direction tags >>vie<< / >>jpn<< are user-defined symbols so each is a
single atomic token; at training time (Bước 4) the target-language tag is
prepended to the source.

Design choices:
  - model_type=unigram: standard for multilingual NMT (mBART/NLLB-style).
  - character_coverage=0.9995: high, to cover the long tail of kanji.
  - byte_fallback=True: rare/unseen chars fall back to bytes -> no <unk> holes.
  - explicit pad/unk/bos/eos ids 0..3 (NMT convention).
  - input_sentence_size subsamples for tractable training on ~10.8M lines.
"""
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "tokenizer"
OUT.mkdir(parents=True, exist_ok=True)

spm.SentencePieceTrainer.train(
    input=[str(CLEAN / "train.ja"), str(CLEAN / "train.vi")],
    model_prefix=str(OUT / "spm_vija_32k"),
    vocab_size=32000,
    model_type="unigram",
    character_coverage=0.9995,
    input_sentence_size=8_000_000,
    shuffle_input_sentence=True,
    user_defined_symbols=[">>vie<<", ">>jpn<<"],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    byte_fallback=True,
    normalization_rule_name="nmt_nfkc",
    num_threads=12,
    train_extremely_large_corpus=True,
)
print("done ->", OUT / "spm_vija_32k.model")
