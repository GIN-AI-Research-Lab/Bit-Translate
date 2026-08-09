#!/usr/bin/env python3
"""Dich bang GGUF i2s qua llama-server + TOKEN IDS (ISSUES #3/#4 — cam de
llama.cpp tu tokenize). Chay trong WSL (vija-venv).

  python gguf_client.py <src.jsonl|sanity> <out.jsonl>
src.jsonl: moi dong {"id","src",...}. sanity: bo 7 cau kiem chung.
"""
import json, sys, time, urllib.request
import unicodedata, re
import sentencepiece as spm

SP = spm.SentencePieceProcessor(model_file="/mnt/e/Bit-Translate/tokenizer/spm_vija_32k.model")
BOS, EOS = SP.bos_id(), SP.eos_id()
VIE = SP.piece_to_id(">>vie<<")
URL = "http://127.0.0.1:8791/completion"


def normalize_for_model(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t)).strip()


def tr(src):
    ids = [BOS, VIE] + SP.encode(normalize_for_model(src)) + [EOS]
    body = json.dumps({"prompt": ids, "n_predict": 200, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out["content"].strip()


SANITY = ["猫が好きです。", "おはようございます。", "これはテストです。",
          "会議の資料を準備してください。", "つい子どもに当たってしまう。",
          "情報が漏れかけた。", "心ばかりの品ですがお納めください。"]

src_arg, out_path = sys.argv[1], sys.argv[2]
if src_arg == "sanity":
    rows = [{"id": str(i + 1), "src": s} for i, s in enumerate(SANITY)]
else:
    rows = [json.loads(l) for l in open(src_arg, encoding="utf-8") if l.strip()]

t0 = time.time()
outs = []
for i, r in enumerate(rows):
    hyp = tr(r["src"])
    outs.append({**r, "hyp": hyp, "sys": "200b_v8avgi2s"})
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
with open(out_path, "w", encoding="utf-8") as f:
    for o in outs:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"XONG {len(outs)} cau, {time.time()-t0:.0f}s -> {out_path}")
if src_arg == "sanity":
    for o in outs:
        print(f"  {o['src']}\n    -> {o['hyp']}")
