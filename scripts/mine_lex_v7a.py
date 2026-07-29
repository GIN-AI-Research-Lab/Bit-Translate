#!/usr/bin/env python3
"""V7A — mine câu chứa từ ghép văn hoá/quán ngữ (danh mục lex_terms.txt) từ CC-100.

Mẫu theo scripts/mine_skills2.py (Aho-Corasick Rust, multiprocessing).
Neo đuôi biến cách cho quán ngữ/ことわざ: cắt đuôi động từ/tính từ thành STEM
(目が肥える -> 目が肥え, 心を鬼にする -> 心を鬼に) — stem phải >= 4 ký tự,
không bao giờ dùng mẫu < 3 ký tự (bẫy それな).

Dedup: blake2b(digest_size=8) trên câu strip so với TOÀN BỘ clean_v6/train.ja.

Chạy (Git Bash):
  wsl -e bash -lc "xz -dc -T0 /mnt/d/Bit-Translate-data/raw/ja.txt.xz" | \
    PYTHONUTF8=1 python scripts/mine_lex_v7a.py --workers 12
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")
GODAN = set("るうくぐすつぬぶむい")


def stems(term, cls):
    """Trả các mẫu quét cho một mục. Quán ngữ: neo đuôi biến cách."""
    if cls in ("kanyouku", "kotowaza", "yoji"):
        if term.endswith("する") and len(term) - 2 >= 4:
            return [term[:-2]]
        if term[-1] in GODAN and len(term) - 1 >= 4:
            return [term[:-1]]
    return [term]


def _init(pack, cap_w):
    global _A, _WL, _P2T, _CAPW, _GOT
    import ahocorasick_rs as ar
    _WL = pack["pats"]
    _P2T = pack["p2t"]
    _A = ar.AhoCorasick(_WL, implementation=ar.Implementation.DFA)
    _CAPW = cap_w
    _GOT = Counter()          # cap cục bộ mỗi worker, bền qua các block


def _clean(s):
    if not (20 <= len(s) <= 180):
        return False
    if "http" in s or "www." in s or "@" in s:
        return False
    asc = sum(1 for c in s if c.isascii() and c.isalnum())
    if asc / len(s) > 0.3:
        return False
    return True


def _work(blob):
    occ = Counter()
    cand = []
    n = 0
    for raw in blob.split(b"\n"):
        if len(raw) < 30:
            continue
        try:
            s = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        n += 1
        if not _clean(s):
            continue
        hits = {_P2T[_WL[i]] for i, _, _ in _A.find_matches_as_indexes(s, overlapping=True)}
        if not hits:
            continue
        occ.update(hits)
        keep = [t for t in hits if _GOT[t] < _CAPW]
        if not keep:
            continue
        for t in keep:
            _GOT[t] += 1
        cand.append((s, keep))
    return occ, cand, n


def blocks(fh, size):
    tail = b""
    while True:
        chunk = fh.read(size)
        if not chunk:
            if tail:
                yield tail
            return
        chunk = tail + chunk
        cut = chunk.rfind(b"\n")
        if cut < 0:
            tail = chunk
            continue
        yield chunk[:cut]
        tail = chunk[cut + 1:]


def dig(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=str(D / "v7a" / "lex_terms.txt"))
    ap.add_argument("--train", default=str(D / "clean_v6" / "train.ja"))
    ap.add_argument("--out", default=str(D / "v7a" / "lex_todo.txt"))
    ap.add_argument("--stats", default=str(D / "v7a" / "lex_stats.json"))
    ap.add_argument("--cap", type=int, default=60)       # trần mỗi mục (toàn cục)
    ap.add_argument("--cap-w", type=int, default=12)     # trần mỗi mục MỖI worker
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    ap.add_argument("--max-min", type=float, default=38.0)
    a = ap.parse_args()

    terms = {}          # term -> class
    p2t = {}            # pattern -> term
    for line in open(a.terms, encoding="utf-8"):
        t, c = line.rstrip("\n").split("\t")
        terms[t] = c
        for p in stems(t, c):
            p2t.setdefault(p, t)
    pats = list(p2t)
    pack = {"pats": pats, "p2t": p2t}
    print(f"danh mục {len(terms):,} mục -> {len(pats):,} mẫu quét", file=sys.stderr, flush=True)

    print("dựng tập dedup từ train.ja (15,3M dòng)...", file=sys.stderr, flush=True)
    t0 = time.time()
    used = set()
    with open(a.train, encoding="utf-8", errors="ignore") as f:
        for line in f:
            used.add(dig(line.strip()))
    print(f"  {len(used):,} hash ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    got = Counter()      # đã ghi mỗi mục
    seen_all = Counter() # xuất hiện (trước cap/dedup)
    seen = set()
    n = kept = marks = 0
    t0 = time.time()
    partial = False

    fo = open(a.out, "w", encoding="utf-8")
    with mp.Pool(a.workers, initializer=_init, initargs=(pack, a.cap_w)) as P:
        it = P.imap(_work, blocks(sys.stdin.buffer, a.block_mb << 20), chunksize=1)
        for occ, cand, cnt in it:
            n += cnt
            seen_all.update(occ)
            for s, ts in cand:
                need = [t for t in ts if got[t] < a.cap]
                if not need:
                    continue
                h = dig(s)
                if h in used or h in seen:
                    continue
                seen.add(h)
                for t in need:
                    got[t] += 1
                fo.write(s + "\n")
                kept += 1
            if n // a.chunk > marks:
                marks = n // a.chunk
                el = time.time() - t0
                full = sum(1 for t in terms if got[t] >= a.cap)
                print(f"  {n:,} | chọn {kept:,} | {n/el/1e6:.2f}M dòng/s | "
                      f"mục đầy {full:,}/{len(terms):,}", file=sys.stderr, flush=True)
            if time.time() - t0 > a.max_min * 60:
                partial = True
                print("!! chạm trần thời gian, dừng sớm", file=sys.stderr, flush=True)
                P.terminate()
                break
    fo.close()

    el = time.time() - t0
    ge10 = sum(1 for t in terms if got[t] >= 10)
    zero = sum(1 for t in terms if seen_all[t] == 0)
    print(f"\n=== XONG ({el:.0f}s, partial={partial}) ===", file=sys.stderr)
    print(f"quét {n:,} dòng | giữ {kept:,} câu", file=sys.stderr)
    print(f"mục >=10 câu: {ge10:,} | mục 0 lần xuất hiện: {zero:,}", file=sys.stderr)

    Path(a.stats).write_text(json.dumps({
        "scanned": n, "kept": kept, "elapsed_s": round(el), "partial": partial,
        "cap": a.cap, "n_terms": len(terms),
        "per_term": {t: got[t] for t in terms},
        "per_term_seen": {t: seen_all[t] for t in terms},
        "per_class_kept": dict(sum((Counter({c: got[t]}) for t, c in terms.items()),
                                   Counter())),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.stats}", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()
