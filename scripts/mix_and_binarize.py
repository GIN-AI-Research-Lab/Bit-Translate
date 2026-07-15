#!/usr/bin/env python3
"""Vòng 1 — lọc + trộn + binarize data mới vào data/bin để train tiếp (PLAN_BUOC5 §2.5).

Làm 4 việc, gói trong 1 lệnh:
  1) GOM data synthetic mới (glossary_sents, glossary_direct, it_docs, phase2,
     copythrough, codeswitch, bt) — mỗi nguồn có chiều + trọng số riêng.
  2) LỌC: loại cặp trùng test-set thuật ngữ (chống rò rỉ eval) + LaBSE >= ngưỡng
     cho data free-text do LLM sinh (glossary_sents/it_docs/phase2). Data sinh
     bằng code (copythrough/codeswitch), cặp term, và BT (target là JA thật) =>
     KHÔNG lọc LaBSE.
  3) OVERSAMPLE: glossary ×glossary-mult; nhân toàn bộ data mới để đạt ~new-frac
     tỉ trọng so với base replay (tự tính hệ số, chặn trần --max-rep).
  4) TRỘN Ở MỨC BINARY: base (data/bin/train.*) là bản đã tokenize sẵn 5.4M cặp —
     KHÔNG tokenize lại; chỉ nối token stream + index của data mới vào cuối.
     train.py xáo trộn toàn cục mỗi epoch => tỉ trọng new xuất hiện đều mọi batch.

Base gốc được sao lưu 1 lần sang data/bin/base.train.* nên chạy lại LẶP được
(luôn dựng train.* = base.train.* + new).

Chiều theo binarize.py: VI->JA thẻ >>jpn<< (src=vi,tgt=ja); JA->VI thẻ >>vie<<.
BT chỉ sinh chiều vi->ja (target = câu Nhật thật).

Usage:
  python scripts/mix_and_binarize.py            # LaBSE bật, new~30%, glossary×3
  python scripts/mix_and_binarize.py --no-labse # bỏ LaBSE (nhanh, kém sạch hơn)
"""
import argparse
import array
import json
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent.parent
SYN = ROOT / "data" / "synthetic"
GLO = ROOT / "data" / "glossary"
BIN = ROOT / "data" / "bin"
MAX_SEQ = 256

sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = sp.bos_id(), sp.eos_id()
VIE, JPN = sp.piece_to_id(">>vie<<"), sp.piece_to_id(">>jpn<<")

# name, format(pair=<name>.ja/.vi | jsonl=file), direction, class, LaBSE?
SOURCES = [
    ("glossary_sents", "jsonl:glossary_sents.jsonl", "both", "glossary", True),
    ("glossary_direct", "pair:glossary_direct",      "both", "glossary", False),
    ("it_docs",         "pair:it_docs",              "both", "other",    True),
    ("phase2",          "jsonl:phase2_pairs.jsonl",  "both", "other",    True),
    ("copythrough",     "pair:copythrough",          "both", "other",    False),
    ("codeswitch",      "pair:codeswitch",           "both", "other",    False),
    ("bt",              "pair:bt",                   "vi2ja", "other",   False),
]


def nfc(s):
    return unicodedata.normalize("NFC", s.strip())


def load_source(spec):
    """Trả list record dict {ja, vi, term}. Bỏ dòng thiếu/rỗng."""
    kind, name = spec.split(":", 1)
    recs = []
    if kind == "jsonl":
        p = SYN / name
        if not p.exists():
            return recs
        for line in p.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            ja, vi = nfc(o.get("ja") or ""), nfc(o.get("vi") or "")
            if ja and vi:
                recs.append({"ja": ja, "vi": vi, "term": (o.get("term") or "").strip()})
    else:  # pair
        pja, pvi = SYN / f"{name}.ja", SYN / f"{name}.vi"
        if not (pja.exists() and pvi.exists()):
            return recs
        ja_l = pja.read_text(encoding="utf-8").splitlines()
        vi_l = pvi.read_text(encoding="utf-8").splitlines()
        for ja, vi in zip(ja_l, vi_l):
            ja, vi = nfc(ja), nfc(vi)
            if ja and vi:                       # dòng bt bị loại (vi rỗng) tự rớt
                recs.append({"ja": ja, "vi": vi, "term": ""})
    return recs


def labse_keep(pairs, thr, device):
    """Trả mask giữ (cosine LaBSE >= thr). pairs: list (ja, vi)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("[mix] CẢNH BÁO: không import được sentence_transformers -> BỎ LaBSE.", flush=True)
        return np.ones(len(pairs), dtype=bool)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[mix] LaBSE score {len(pairs):,} cặp (device={device}, thr={thr})...", flush=True)
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    model.max_seq_length = 128
    if device == "cuda":
        model.half()
    B = 256
    keep = np.ones(len(pairs), dtype=bool)
    ja = [p[0] for p in pairs]
    vi = [p[1] for p in pairs]
    for i in range(0, len(pairs), B):
        j = min(i + B, len(pairs))
        eja = model.encode(ja[i:j], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
        evi = model.encode(vi[i:j], batch_size=B, convert_to_numpy=True, normalize_embeddings=True)
        sc = np.sum(eja * evi, axis=1)
        keep[i:j] = sc >= thr
        if (i // B) % 100 == 0:
            print(f"  {j:,}/{len(pairs):,} | giữ {int(keep[:j].sum()):,}", flush=True)
    return keep


def seqs_for(ja_ids, vi_ids, direction):
    """Sinh (seq, tgt_start) theo chiều — khớp hệt binarize.py."""
    out = []
    dirs = ((JPN, vi_ids, ja_ids),) if direction == "vi2ja" else \
           ((JPN, vi_ids, ja_ids), (VIE, ja_ids, vi_ids))   # VI->JA, JA->VI
    for tag, src, tgt in dirs:
        seq = [BOS, tag] + src + [EOS] + tgt + [EOS]
        if len(seq) <= MAX_SEQ:
            out.append((seq, len(src) + 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-frac", type=float, default=0.30, help="tỉ trọng data mới mong muốn")
    ap.add_argument("--glossary-mult", type=int, default=3, help="oversample riêng cho glossary")
    ap.add_argument("--max-rep", type=int, default=6, help="trần hệ số nhân data mới (chống overfit)")
    ap.add_argument("--labse", dest="labse", action="store_true", default=True)
    ap.add_argument("--no-labse", dest="labse", action="store_false")
    ap.add_argument("--labse-thr", type=float, default=0.80)
    ap.add_argument("--labse-device", default="auto")
    args = ap.parse_args()

    # base bin phải có sẵn (giải nén từ release train-assets-step14000)
    base_idx_p = BIN / "base.train.index.npy"
    base_tok_p = BIN / "base.train.tokens.u16"
    if not base_idx_p.exists():
        cur_idx, cur_tok = BIN / "train.index.npy", BIN / "train.tokens.u16"
        if not cur_idx.exists():
            sys.exit("[mix] THIẾU data/bin/train.* — giải nén vija_data.tar.zst từ release trước.")
        print("[mix] sao lưu base gốc -> base.train.*", flush=True)
        shutil.copyfile(cur_idx, base_idx_p)
        shutil.copyfile(cur_tok, base_tok_p)
    base_idx = np.load(base_idx_p)
    base_seq = base_idx.shape[0]

    # 1) gom
    recs = []
    per_src_raw = {}
    for name, spec, direction, cls, use_labse in SOURCES:
        r = load_source(spec)
        per_src_raw[name] = len(r)
        for x in r:
            x.update(name=name, dir=direction, cls=cls, labse=use_labse)
        recs.extend(r)
    print("[mix] đọc thô:", per_src_raw, flush=True)

    # 2a) dedup toàn cục theo (ja,vi)
    seen = set()
    ded = []
    for x in recs:
        k = (x["ja"], x["vi"])
        if k in seen:
            continue
        seen.add(k)
        ded.append(x)
    print(f"[mix] sau dedup: {len(ded):,} (bỏ {len(recs)-len(ded):,})", flush=True)

    # 2b) loại cặp trùng test-set thuật ngữ
    test_terms = set()
    ttp = GLO / "glossary_test_terms.txt"
    if ttp.exists():
        test_terms = {t.strip() for t in ttp.read_text(encoding="utf-8").splitlines() if t.strip()}
    n_leak = 0
    kept = []
    for x in ded:
        tok = x["term"] if x["name"] == "glossary_sents" else (x["ja"] if x["name"] == "glossary_direct" else None)
        if tok and tok in test_terms:
            n_leak += 1
            continue
        kept.append(x)
    print(f"[mix] loại trùng test-term: {n_leak:,} | còn {len(kept):,}", flush=True)

    # 2c) LaBSE cho các nguồn free-text (giữ nguyên phần không cần lọc)
    if args.labse:
        need = [i for i, x in enumerate(kept) if x["labse"]]
        if need:
            mask = labse_keep([(kept[i]["ja"], kept[i]["vi"]) for i in need],
                              args.labse_thr, args.labse_device)
            drop = {need[i] for i in range(len(need)) if not mask[i]}
            kept = [x for i, x in enumerate(kept) if i not in drop]
            print(f"[mix] LaBSE loại {len(drop):,} | còn {len(kept):,}", flush=True)
    else:
        print("[mix] --no-labse: bỏ qua lọc LaBSE.", flush=True)

    # 3) tính hệ số oversample tự động
    def rep_of(x):
        return args.glossary_mult if x["cls"] == "glossary" else 1

    unit_seq = sum((1 if x["dir"] == "vi2ja" else 2) * rep_of(x) for x in kept)
    target_new = base_seq * args.new_frac / max(1e-9, 1 - args.new_frac)
    new_rep = int(round(target_new / max(unit_seq, 1)))
    new_rep = max(1, min(args.max_rep, new_rep))
    print(f"[mix] base_seq={base_seq:,} | unit_seq(new,×1)={unit_seq:,} | "
          f"new_rep={new_rep} (glossary hiệu dụng ×{new_rep*args.glossary_mult})", flush=True)

    # 4) tokenize data mới (cache encode cho nhanh) + dựng sequence
    uniq = sorted({x["ja"] for x in kept} | {x["vi"] for x in kept})
    enc = {}
    for s in range(0, len(uniq), 20000):
        chunk = uniq[s:s + 20000]
        for t, ids in zip(chunk, sp.encode(chunk)):
            enc[t] = ids
    print(f"[mix] tokenize {len(uniq):,} câu duy nhất xong", flush=True)

    buf = array.array("H")
    idx = array.array("i")
    n_new = 0
    n_skip = 0
    for x in kept:
        pairs = seqs_for(enc[x["ja"]], enc[x["vi"]], x["dir"])
        if not pairs:
            n_skip += 1
            continue
        r = new_rep * rep_of(x)
        for seq, tstart in pairs:
            for _ in range(r):
                buf.extend(seq)
                idx.append(len(seq))
                idx.append(tstart)
                n_new += 1

    # 5) trộn binary: train.* = base.train.* + new
    BIN.mkdir(parents=True, exist_ok=True)
    out_tok = BIN / "train.tokens.u16"
    shutil.copyfile(base_tok_p, out_tok)               # bắt đầu từ base
    with open(out_tok, "ab") as f:
        buf.tofile(f)
    new_idx = np.frombuffer(idx, dtype=np.int32).reshape(-1, 2)
    all_idx = np.vstack([base_idx, new_idx])
    np.save(BIN / "train.index.npy", all_idx)

    total = base_seq + n_new
    print(f"[mix] XONG: base {base_seq:,} + new {n_new:,} = {total:,} seq "
          f"(new {100*n_new/total:.1f}%) | bỏ quá dài {n_skip:,}", flush=True)
    print("[mix] -> data/bin/train.{tokens.u16,index.npy} (dev giữ nguyên base)", flush=True)


if __name__ == "__main__":
    main()
