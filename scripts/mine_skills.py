#!/usr/bin/env python3
"""Đào kho gốc theo 6 MẶT TRẬN LỖI (không phải theo chủ đề) + phủ nốt thuật ngữ miền.

Vì sao không đào theo chủ đề nữa: phân loại 59 câu v5 hỏng (mà Google dịch được) cho
thấy thuật ngữ chuyên ngành chỉ chiếm 3% lỗi chính. Nguyên nhân thật: cấu trúc câu 37%,
nghĩa từ thông thường 22%, ẩn chủ ngữ 10%.

VÀ mật độ data KHÔNG tự động thành chất lượng — đo được:
    keigo    4,00% corpus -> 80% = NGANG Google (kỹ năng đã học xong)
    katakana 23,2% corpus -> 75% vs 95% (thừa data mà vẫn hỏng)
    slang    0,63% corpus -> 45% vs 90% (đói data thật)
⇒ chia hai loại can thiệp:
  A. ĐÓI DATA  (slang, phủ định) -> bơm lên ~4% như keigo là đủ.
  B. ĐỦ DATA MÀ VẪN HỎNG (katakana, ẩn chủ ngữ) -> bơm thêm cùng loại là VÔ ÍCH.
     Phải chọn câu có TÍNH CHẤT ĐẶC BIỆT: katakana thì lấy câu có từ mượn hiếm;
     ẩn chủ ngữ thì lấy câu mà chủ ngữ PHỤC HỒI ĐƯỢC trong chính câu đó (có tiền tố
     chỉ người ở đầu) — để model học cách truy chủ ngữ, thay vì mặc định "tôi".
  C. MỆNH ĐỀ LỒNG: model ĐÃ chứng minh xử lý được câu Quốc hội 159 ký tự (95% held-out),
     mà câu Quốc hội cực nhiều mệnh đề lồng ⇒ không phải thiếu dung lượng, mà là
     THIẾU VĂN NÓI nhiều mệnh đề. Nên lọc câu vừa nhiều mệnh đề VỪA mang dấu văn nói.

  xz -dc D:/Bit-Translate-data/raw/ja.txt.xz | python scripts/mine_skills.py --workers 10
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

# tên -> (regex, số lần khớp tối thiểu, hạn mức câu, mô tả)
#
# ⚠️ BẢN ĐẦU BẮT BỪA NẶNG — đã đo và sửa, giữ chú thích để không lặp lại:
#   それな -> khớp「それなり」(21% số khớp!) · きも -> khớp「読み書きも」「生きもの」(21%)
#   マジ   -> khớp「マジック」「マジシャン」(8%) · 沼 -> khớp「沼津」「泥沼」(5%)
#   おつ   -> khớp「おつまみ」「おつかい」(3%) · まい[。、] -> khớp「〜てしまい、」(28%!)
# Tổng cộng ~65% câu "slang" và ~28% câu "phủ định" là RÁC.
# Bài học: mẫu tiếng Nhật ngắn (2 ký tự) gần như LUÔN là chuỗi con của từ khác —
# phải neo bằng đuôi biến cách hoặc dấu câu, không được để trần.
FRONTS = {
    "slang": (
        r"ヤバ[いくすぎか]|やば[いくすぎ]|マジ[でかっ!?？よ]|ウケ[るたw]|エモい|"
        r"ガチ[でやん]|それな[。！\s]|草[。ｗw\s]|草生え|うざ[いくっ]|ウザ[いくっ]|"
        r"キモ[いくっ]|きもい|ワンチャン|推し[がはをに活変]|バズ[っるり]|エグ[いく]|"
        r"だる[いくっ]|めんど[いくさ]|むずい|やらかし|ドン引き|あざ[すっ]|パない|"
        r"ノリ[がでよ]|イケてる|ダサ[いくっ]|ぼっち|リア充|尊[すい]|わろた|"
        r"とりま[。、\s]|ぴえん|しか勝たん|チャラ[いく]|しんどい", 1, 450_000),
    "phu_dinh_kho": (
        r"ないわけ|なくはない|ざるを得な|ないこともない|ずして|かねない|"
        r"にほかならな|とは限らな|わけではな|どころか|ばかりか|ずには|なくもない|"
        r"ないではいられ|あるまじき|でなくてなん|ないとも限ら|ざるべから", 1, 350_000),
    # văn NÓI nhiều mệnh đề (không lấy văn hành chính — corpus đã thừa)
    "menh_de_noi": (
        r"(?=.*(という|ような|ための|ということ|とされ|と思われ|といった))"
        r"(?=.*(んです|んだけど|なんですけど|んですよ|んですね|でしょうか|"
        r"じゃないですか|かなと|かなって|と思うんです|ですけども))", 1, 400_000),
    # ẩn chủ ngữ PHỤC HỒI ĐƯỢC: có người/chức danh ở đầu, rồi vế sau lược chủ ngữ
    "an_chu_ngu_truy": (
        r"^[^。]{0,30}(私|僕|彼|彼女|あなた|[一-龥]{2,4}(さん|氏|大臣|議員|社長|先生))"
        r"[はがも][^。]{25,}(ます|ました|ている|ていた|です)[。、]", 1, 350_000),
    # KHÔNG còn mặt trận katakana bằng regex: `[ァ-ヶ]{7,}` chỉ bắt katakana DÀI,
    # mà dài ≠ hiếm (nó lấy về ホルモンバランス, ファイルダウンロード — corpus đã thừa).
    # Katakana chuyển sang vá THEO TỪ như thuật ngữ: xem --kata (build_katakana_gap.py).
}


def _init(fronts, term_words, lo, hi):
    global _F, _A, _WL, _LO, _HI
    import ahocorasick_rs as ar
    _F = [(k, re.compile(v[0]), v[1]) for k, v in fronts.items()]
    _WL = term_words
    _A = ar.AhoCorasick(term_words, implementation=ar.Implementation.DFA) if term_words else None
    _LO, _HI = lo, hi


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
        L = len(s)
        if not (_LO <= L <= _HI):
            continue
        tags = []
        for name, pat, need in _F:
            if need == 1:
                if pat.search(s):
                    tags.append(name)
            elif len(pat.findall(s)) >= need:
                tags.append(name)
        terms = set()
        if _A is not None:
            terms = {_WL[i] for i, _, _ in _A.find_matches_as_indexes(s, overlapping=True)}
            for t in terms:
                occ[t] += 1
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
    ap.add_argument("--terms", default=str(D / "domain_terms.json"))
    ap.add_argument("--cov", default="eval/term_coverage.json")
    ap.add_argument("--mine-stats", default="eval/mine_stats.json")
    ap.add_argument("--out", default=str(D / "raw" / "mined_skills.txt"))
    ap.add_argument("--stats", default="eval/mine_skills_stats.json")
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--cap-per-term", type=int, default=150)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--max-len", type=int, default=170)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--kata", default=str(D / "katakana_gap.json"))
    ap.add_argument("--kata-top", type=int, default=9000)
    ap.add_argument("--chunk", type=int, default=50_000_000)
    a = ap.parse_args()

    # (1) thuật ngữ miền CÒN THIẾU sau lượt đào trước
    cov = json.loads(Path(a.cov).read_text(encoding="utf-8"))
    cur, pool = cov["counts_cur"], cov["counts_pool"]
    prev = json.loads(Path(a.mine_stats).read_text(encoding="utf-8"))["counts_archive_full"]
    want_term = {}
    for t in cur:
        have = cur[t] + pool[t] + prev.get(t, 0)
        if have < a.min_occ:
            want_term[t] = min(a.cap_per_term, int((a.min_occ - have) * 1.3))
    n_dom = len(want_term)

    # (2) TỪ MƯỢN KATAKANA thiếu — vá theo TỪ, không theo regex "katakana dài".
    # Lấy top theo tần suất NGOÀI ĐỜI: từ hay gặp thật thì đáng học trước.
    kg = Path(a.kata)
    if kg.exists():
        g = json.loads(kg.read_text(encoding="utf-8"))["gap"]
        top = sorted(g.items(), key=lambda x: -x[1]["ref"])[:a.kata_top]
        for w, d in top:
            want_term[w] = min(a.cap_per_term, int((a.min_occ - d["cur"]) * 1.3))
    tw = list(want_term)
    print(f"thuật ngữ miền thiếu: {n_dom:,} | từ mượn katakana thiếu: "
          f"{len(tw)-n_dom:,}", file=sys.stderr)

    quota = {k: v[2] for k, v in FRONTS.items()}
    print(f"6 mặt trận | hạn mức {sum(quota.values()):,} câu", file=sys.stderr)
    for k, v in FRONTS.items():
        print(f"   {k:20} {v[2]:>9,}", file=sys.stderr)


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

    got_f = Counter()
    got_t = Counter()
    total_occ = Counter()
    seen = set()
    n = kept = 0
    per_front = Counter()
    t0 = time.time()

    fo = open(a.out, "w", encoding="utf-8")
    with mp.Pool(a.workers, initializer=_init,
                 initargs=(FRONTS, tw, a.min_len, a.max_len)) as P:
        for occ, cand, cnt in P.imap(_work, blocks(sys.stdin.buffer, a.block_mb << 20),
                                     chunksize=1):
            n += cnt
            total_occ.update(occ)
            for s, tags, terms in cand:
                need_f = [t for t in tags if got_f[t] < quota[t]]
                need_t = [t for t in terms if got_t[t] < want_term.get(t, 0)]
                if not need_f and not need_t:
                    continue
                h = dig(s)
                if h in used or h in seen:
                    continue
                seen.add(h)
                for t in need_f:
                    got_f[t] += 1
                    per_front[t] += 1
                for t in need_t:
                    got_t[t] += 1
                fo.write(s + "\n")
                kept += 1
            if n // a.chunk > len(getattr(main, "_m", [])):
                main._m = getattr(main, "_m", []) + [n]
                el = time.time() - t0
                print(f"  {n:,} | chọn {kept:,} | {n/el/1e6:.1f}M/s | "
                      + " ".join(f"{k}:{got_f[k]:,}" for k in quota),
                      file=sys.stderr, flush=True)
    fo.close()

    print(f"\n=== ĐÀO XONG ({time.time()-t0:.0f}s) ===", file=sys.stderr)
    print(f"quét {n:,} câu | chọn {kept:,} -> {a.out}\n", file=sys.stderr)
    print(f"{'mặt trận':<22}{'lấy được':>12}{'hạn mức':>10}{'  đạt':>7}", file=sys.stderr)
    for k in quota:
        print(f"{k:<22}{got_f[k]:>12,}{quota[k]:>10,}{100*got_f[k]/quota[k]:>6.0f}%",
              file=sys.stderr)
    filled = sum(1 for t in want_term if cur[t] + pool[t] + prev.get(t, 0) + total_occ[t] >= a.min_occ)
    print(f"\nthuật ngữ miền lấp thêm: {filled:,}/{len(want_term):,}", file=sys.stderr)

    Path(a.stats).write_text(json.dumps(
        {"scanned": n, "kept": kept, "per_front": dict(got_f),
         "term_extra": {t: total_occ[t] for t in tw}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"-> {a.stats}", file=sys.stderr)


if __name__ == "__main__":
    mp.freeze_support()
    main()
