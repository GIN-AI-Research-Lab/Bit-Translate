#!/usr/bin/env python3
"""Bản NHIỀU TIẾN TRÌNH của mine_rare_terms.py — cùng kết quả, nhanh ~5x.

Vì sao bản một luồng chậm (đã đo): quét Aho-Corasick bằng backend Rust đạt 170 MB/s
khi chuỗi đã nằm sẵn trong RAM, nhưng đọc từ ống `xz -dc` chỉ đạt ~26 MB/s — phần
chênh là GIẢI MÃ UTF-8 từng dòng trong Python, không phải việc khớp chuỗi.
⇒ Đẩy cả giải mã LẪN quét sang tiến trình con; tiến trình chính chỉ đọc BYTE thô,
cắt theo dòng và phát đi.

Chỉ ~1% số dòng có chứa thuật ngữ đang thiếu (đo được), nên tiến trình chính chỉ phải
xử lý 1% dữ liệu — logic HẠN MỨC (mỗi thuật ngữ lấy tối đa N câu) vẫn nằm tập trung ở
đó, không phân tán ra tiến trình con. Nhờ vậy kết quả giống hệt bản một luồng.

Trần tốc độ là `xz` (~133 MB/s) ⇒ 8 tiến trình con là đủ bão hoà; thêm nữa vô ích.

  xz -dc D:/Bit-Translate-data/raw/ja.txt.xz | python scripts/mine_rare_terms_mp.py
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
_A = None
_WL = None
_LO = 20
_HI = 160


def _init(words, lo, hi):
    global _A, _WL, _LO, _HI
    import ahocorasick_rs as ar
    _WL = words
    _A = ar.AhoCorasick(words, implementation=ar.Implementation.DFA)
    _LO, _HI = lo, hi


def _work(blob):
    """Nhận một khối byte thô -> (đếm thuật ngữ, danh sách câu ứng viên, số dòng)."""
    occ = Counter()
    cand = []
    n = 0
    for raw in blob.split(b"\n"):
        if len(raw) < 10:
            continue
        try:
            s = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        n += 1
        hits = _A.find_matches_as_indexes(s, overlapping=True)
        if not hits:
            continue
        ts = {_WL[i] for i, _, _ in hits}
        for t in ts:
            occ[t] += 1
        if _LO <= len(s) <= _HI:
            cand.append((s, ts))
    return occ, cand, n


def blocks(fh, size):
    """Cắt stdin thành khối ~size byte, luôn kết thúc ở ranh giới dòng."""
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
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=str(D / "domain_terms.json"))
    ap.add_argument("--cov", default="eval/term_coverage.json")
    ap.add_argument("--out", default=str(D / "raw" / "mined_rare.txt"))
    ap.add_argument("--stats", default="eval/mine_stats.json")
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--margin", type=float, default=1.3)
    ap.add_argument("--cap-per-term", type=int, default=150)
    ap.add_argument("--max-out", type=int, default=700000)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=40_000_000)
    a = ap.parse_args()

    cov = json.loads(Path(a.cov).read_text(encoding="utf-8"))
    cur, pool = cov["counts_cur"], cov["counts_pool"]
    doms = json.loads(Path(a.terms).read_text(encoding="utf-8"))

    want = {}
    for t in cur:
        have = cur[t] + pool[t]
        if have < a.min_occ:
            want[t] = min(a.cap_per_term, int((a.min_occ - have) * a.margin))
    wl = list(want)
    print(f"nhắm {len(want):,} thuật ngữ | hạn mức {sum(want.values()):,} câu | "
          f"{a.workers} tiến trình", file=sys.stderr, flush=True)

    print("dựng tập câu đã dùng...", file=sys.stderr, flush=True)
    used = set()
    with open(D / "kd_v5_merged.jsonl", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                used.add(dig(json.loads(line)["ja"]))
            except Exception:
                pass
    print(f"  {len(used):,}", file=sys.stderr, flush=True)

    got = Counter()
    total_occ = Counter()
    chunk_occ, cur_chunk = [], 0
    seen_out = set()
    n = kept = 0
    t0 = time.time()

    fo = open(a.out, "w", encoding="utf-8")
    with mp.Pool(a.workers, initializer=_init,
                 initargs=(wl, a.min_len, a.max_len)) as pool_:
        it = pool_.imap(_work, blocks(sys.stdin.buffer, a.block_mb << 20), chunksize=1)
        for occ, cand, cnt in it:
            n += cnt
            cur_chunk += sum(occ.values())
            total_occ.update(occ)
            if kept < a.max_out:
                for s, ts in cand:
                    need = [t for t in ts if got[t] < want[t]]
                    if not need:
                        continue
                    h = dig(s)
                    if h in used or h in seen_out:
                        continue
                    seen_out.add(h)
                    for t in need:
                        got[t] += 1
                    fo.write(s + "\n")
                    kept += 1
                    if kept >= a.max_out:
                        break
            if n // a.chunk > len(chunk_occ):
                chunk_occ.append(cur_chunk)
                cur_chunk = 0
                el = time.time() - t0
                print(f"  {n:,} câu | chọn {kept:,} | {n/el/1e6:.1f}M dòng/s",
                      file=sys.stderr, flush=True)
    fo.close()
    if cur_chunk:
        chunk_occ.append(cur_chunk)

    filled = sum(1 for t in want if cur[t] + pool[t] + total_occ[t] >= a.min_occ)
    zero = sum(1 for t in want if total_occ[t] == 0)
    print(f"\n=== ĐÀO XONG ({time.time()-t0:.0f}s) ===", file=sys.stderr)
    print(f"quét      : {n:,} câu", file=sys.stderr)
    print(f"chọn      : {kept:,} câu -> {a.out}", file=sys.stderr)
    print(f"lấp được  : {filled:,}/{len(want):,} từ ({100*filled/len(want):.0f}%)", file=sys.stderr)
    print(f"vẫn 0 lần : {zero:,} từ -> bắt buộc SINH", file=sys.stderr)
    if len(chunk_occ) > 2:
        m = sum(chunk_occ) / len(chunk_occ)
        sd = (sum((x - m) ** 2 for x in chunk_occ) / len(chunk_occ)) ** 0.5
        verdict = "ĐỀU, ngoại suy từ mẫu đầu file tin được" if sd / m < 0.25 \
            else "LỆCH, mẫu đầu file KHÔNG đại diện"
        print(f"đồng đều  : {len(chunk_occ)} lô, lệch chuẩn/TB = {sd/m:.1%} ({verdict})",
              file=sys.stderr)

    per = {}
    for name, items in doms.items():
        ts = [o["t"] for o in items]
        before = sum(1 for t in ts if cur[t] + pool[t] >= a.min_occ)
        after = sum(1 for t in ts if cur[t] + pool[t] + total_occ[t] >= a.min_occ)
        per[name] = {"before": round(100 * before / len(ts)),
                     "after": round(100 * after / len(ts)),
                     "still": sum(1 for t in ts if cur[t] + pool[t] + total_occ[t] < a.min_occ)}
    Path(a.stats).write_text(json.dumps(
        {"scanned": n, "kept": kept, "chunk_occ": chunk_occ, "per_domain": per,
         "counts_archive_full": {t: total_occ[t] for t in want}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'miền':<26}{'+kho tải':>10}{'+ĐÀO HẾT':>10}{'còn thiếu':>11}", file=sys.stderr)
    for k, v in sorted(per.items(), key=lambda x: x[1]["after"]):
        print(f"{k:<26}{v['before']:>9}%{v['after']:>9}%{v['still']:>11}", file=sys.stderr)
    print(f"\n-> {a.stats}", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()
