#!/usr/bin/env python3
"""Đào theo MẶT TRẬN LỖI — bản 2, nhanh gấp ~8 lần bản 1 và thêm mặt trận từ đa nghĩa.

VÌ SAO BẢN 1 CHẬM (0,1M dòng/s so với 0,9M/s của bộ đào thuật ngữ): hai regex có
lookahead `(?=.*A)(?=.*B)` phải quét lại toàn chuỗi hai lần cho MỌI dòng, kể cả 99%
dòng không liên quan. Sửa: mọi mẫu là CHUỖI CỐ ĐỊNH thì dồn hết vào MỘT máy
Aho-Corasick (một lượt quét, backend Rust); chỉ giữ regex cho mẫu cần vị trí.

MẶT TRẬN MỚI — từ đa nghĩa (22% lỗi chính, bản 1 BỎ SÓT):
  Lỗi đo được không phải thuật ngữ chuyên ngành mà là từ THƯỜNG bị chọn sai nghĩa:
  微妙 (tinh tế / hơi kém), 結構 (khá / thôi khỏi), 検討します (sẽ xem xét / từ chối khéo).
  Cần nhiều câu chứa CÙNG một từ ở NGỮ CẢNH KHÁC NHAU -> hạn mức mỗi từ cao hơn
  thuật ngữ (200 câu/từ thay vì ~60), vì phải phủ nhiều nghĩa chứ không chỉ nhớ một.

  xz -dc D:/Bit-Translate-data/raw/ja.txt.xz | python scripts/mine_skills2.py --workers 13
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")


def expand(spec):
    """'ヤバ[いくす]' -> ['ヤバい','ヤバく','ヤバす']; chuỗi thường giữ nguyên."""
    m = re.match(r"^(.*?)\[([^\]]+)\]$", spec)
    return [m.group(1) + c for c in m.group(2)] if m else [spec]


def lits(*specs):
    out = []
    for s in specs:
        out += expand(s)
    return out


# --- các mặt trận dùng CHUỖI CỐ ĐỊNH (chạy trong Aho-Corasick) ---
SLANG = lits("ヤバ[いくすぎか]", "やば[いくすぎ]", "マジ[でかっよ]", "ウケ[るた]", "エモい",
             "ガチ[でやん]", "それな。", "それな！", "草。", "草ｗ", "草w", "草生え",
             "うざ[いくっ]", "ウザ[いくっ]", "キモ[いくっ]", "きもい", "ワンチャン",
             "推し[がはをに]", "推し活", "バズ[っるり]", "エグ[いく]", "だる[いくっ]",
             "めんど[いくさ]", "むずい", "やらかし", "ドン引き", "あざ[すっ]", "パない",
             "ノリ[がでよ]", "イケてる", "ダサ[いくっ]", "ぼっち", "リア充", "尊[すい]",
             "わろた", "とりま。", "とりま、", "ぴえん", "しか勝たん", "チャラ[いく]", "しんどい")

PHU_DINH = lits("ないわけ", "なくはない", "ざるを得な", "ないこともない", "ずして", "かねない",
                "にほかならな", "とは限らな", "わけではな", "どころか", "ばかりか", "ずには",
                "なくもない", "ないではいられ", "あるまじき", "でなくてなん", "ないとも限ら")

# mệnh đề lồng: phải có CẢ dấu mệnh đề (A) LẪN dấu văn nói (B)
MD_A = lits("という", "ような", "ための", "ということ", "とされ", "と思われ", "といった",
            "における", "に関する", "とのこと")
MD_B = lits("んです", "んだけど", "なんですけど", "んですよ", "んですね", "でしょうか",
            "じゃないですか", "かなと", "かなって", "と思うんです", "ですけども", "ですよね")

# ẩn chủ ngữ: cần vị trí -> vẫn dùng regex, nhưng CHỈ chạy khi dòng có mồi dưới đây
ACN_TRIG = lits("私は", "私が", "僕は", "僕が", "彼は", "彼が", "彼女は", "彼女が",
                "さんは", "さんが", "氏は", "氏が", "大臣は", "議員は", "社長は", "先生は")
ACN_RE = re.compile(r"^[^。]{0,30}(私|僕|彼|彼女|[一-龥]{2,4}(さん|氏|大臣|議員|社長|先生))"
                    r"[はが][^。]{25,}(ます|ました|ている|ていた|です)[。、]")

QUOTA = {"slang": 450_000, "phu_dinh": 350_000, "menh_de_noi": 400_000,
         "an_chu_ngu": 350_000}


def _init(pack, lo, hi):
    global _A, _WL, _IDX, _LO, _HI
    import ahocorasick_rs as ar
    _WL = pack["words"]
    _IDX = pack["idx"]          # từ -> nhãn ('S','P','A','B','T:<term>','G' mồi ACN)
    _A = ar.AhoCorasick(_WL, implementation=ar.Implementation.DFA)
    _LO, _HI = lo, hi


def _work(blob):
    occ = Counter()
    cand = []
    n = 0
    for raw in blob.split(b"\n"):
        if len(raw) < 40:
            continue
        try:
            s = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        n += 1
        L = len(s)
        if not (_LO <= L <= _HI):
            continue
        hits = {_WL[i] for i, _, _ in _A.find_matches_as_indexes(s, overlapping=True)}
        if not hits:
            continue
        tags = set()
        terms = set()
        md_a = md_b = False
        acn_trig = False
        for w in hits:
            lab = _IDX[w]
            if lab == "S":
                tags.add("slang")
            elif lab == "P":
                tags.add("phu_dinh")
            elif lab == "A":
                md_a = True
            elif lab == "B":
                md_b = True
            elif lab == "G":
                acn_trig = True
            else:                      # thuật ngữ / katakana / đa nghĩa
                terms.add(w)
                occ[w] += 1
        if md_a and md_b:
            tags.add("menh_de_noi")
        if acn_trig and ACN_RE.match(s):
            tags.add("an_chu_ngu")
        if tags or terms:
            cand.append((s, tags, terms))
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
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", default="eval/term_coverage.json")
    ap.add_argument("--mine-stats", default="eval/mine_stats.json")
    ap.add_argument("--kata", default=str(D / "katakana_gap.json"))
    ap.add_argument("--poly", default=str(D / "polysemy.json"))
    ap.add_argument("--kata-top", type=int, default=9000)
    ap.add_argument("--poly-cap", type=int, default=200)
    ap.add_argument("--out", default=str(D / "raw" / "mined_skills.txt"))
    ap.add_argument("--stats", default="eval/mine_skills_stats.json")
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--cap-per-term", type=int, default=150)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=170)
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    a = ap.parse_args()

    want = {}
    cov = json.loads(Path(a.cov).read_text(encoding="utf-8"))
    cur, pool = cov["counts_cur"], cov["counts_pool"]
    prev = json.loads(Path(a.mine_stats).read_text(encoding="utf-8"))["counts_archive_full"]
    for t in cur:
        have = cur[t] + pool[t] + prev.get(t, 0)
        if have < a.min_occ:
            want[t] = min(a.cap_per_term, int((a.min_occ - have) * 1.3))
    n_dom = len(want)

    kg = Path(a.kata)
    if kg.exists():
        g = json.loads(kg.read_text(encoding="utf-8"))["gap"]
        for w, d in sorted(g.items(), key=lambda x: -x[1]["ref"])[:a.kata_top]:
            want[w] = min(a.cap_per_term, int((a.min_occ - d["cur"]) * 1.3))
    n_kata = len(want) - n_dom

    pp = Path(a.poly)
    n_poly = 0
    if pp.exists():
        for w in json.loads(pp.read_text(encoding="utf-8")):
            want[w] = a.poly_cap          # cao hơn: phải phủ NHIỀU NGHĨA
            n_poly += 1

    idx = {}
    for w in SLANG:
        idx[w] = "S"
    for w in PHU_DINH:
        idx.setdefault(w, "P")
    for w in MD_A:
        idx.setdefault(w, "A")
    for w in MD_B:
        idx.setdefault(w, "B")
    for w in ACN_TRIG:
        idx.setdefault(w, "G")
    for w in want:
        idx[w] = "T"                       # thuật ngữ thắng nếu trùng
    words = list(idx)
    pack = {"words": words, "idx": idx}

    print(f"thuật ngữ miền {n_dom:,} | katakana {n_kata:,} | ĐA NGHĨA {n_poly:,} "
          f"(x{a.poly_cap} câu/từ)", file=sys.stderr)
    print(f"mặt trận: " + " ".join(f"{k}={v:,}" for k, v in QUOTA.items()), file=sys.stderr)
    print(f"tổng mẫu trong máy Aho-Corasick: {len(words):,}", file=sys.stderr, flush=True)

    print("dựng tập câu đã dùng...", file=sys.stderr, flush=True)
    used = set()
    with open(D / "kd_v5_merged.jsonl", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                used.add(dig(json.loads(line)["ja"]))
            except Exception:
                pass
    mr = D / "raw" / "mined_rare.txt"
    if mr.exists():
        for line in mr.open(encoding="utf-8", errors="ignore"):
            used.add(dig(line.strip()))
    print(f"  {len(used):,}", file=sys.stderr, flush=True)

    got_f, got_t, total_occ = Counter(), Counter(), Counter()
    seen = set()
    n = kept = 0
    marks = 0
    t0 = time.time()

    fo = open(a.out, "w", encoding="utf-8")
    with mp.Pool(a.workers, initializer=_init, initargs=(pack, a.min_len, a.max_len)) as P:
        for occ, cand, cnt in P.imap(_work, blocks(sys.stdin.buffer, a.block_mb << 20),
                                     chunksize=1):
            n += cnt
            total_occ.update(occ)
            for s, tags, terms in cand:
                nf = [t for t in tags if got_f[t] < QUOTA[t]]
                nt = [t for t in terms if got_t[t] < want.get(t, 0)]
                if not nf and not nt:
                    continue
                h = dig(s)
                if h in used or h in seen:
                    continue
                seen.add(h)
                for t in nf:
                    got_f[t] += 1
                for t in nt:
                    got_t[t] += 1
                fo.write(s + "\n")
                kept += 1
            if n // a.chunk > marks:
                marks = n // a.chunk
                el = time.time() - t0
                print(f"  {n:,} | chọn {kept:,} | {n/el/1e6:.2f}M/s | "
                      + " ".join(f"{k}:{got_f[k]:,}" for k in QUOTA),
                      file=sys.stderr, flush=True)
    fo.close()

    print(f"\n=== ĐÀO XONG ({time.time()-t0:.0f}s) ===", file=sys.stderr)
    print(f"quét {n:,} | chọn {kept:,} -> {a.out}\n", file=sys.stderr)
    for k, q in QUOTA.items():
        print(f"  {k:<14}{got_f[k]:>9,} / {q:,}  ({100*got_f[k]/q:.0f}%)", file=sys.stderr)
    poly = list(json.loads(Path(a.poly).read_text(encoding="utf-8"))) if pp.exists() else []
    if poly:
        full = sum(1 for w in poly if got_t[w] >= a.poly_cap)
        print(f"  {'đa nghĩa':<14}{full:>9,} / {len(poly):,} từ đạt đủ {a.poly_cap} câu",
              file=sys.stderr)
    filled = sum(1 for t in want if cur.get(t, 0) + pool.get(t, 0) + prev.get(t, 0)
                 + total_occ[t] >= a.min_occ)
    print(f"  thuật ngữ+katakana lấp: {filled:,}/{len(want):,}", file=sys.stderr)

    Path(a.stats).write_text(json.dumps(
        {"scanned": n, "kept": kept, "per_front": dict(got_f),
         "per_term": {t: got_t[t] for t in want}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"-> {a.stats}", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()
