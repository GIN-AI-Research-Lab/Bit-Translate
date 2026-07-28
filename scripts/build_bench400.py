#!/usr/bin/env python3
"""Dựng bộ đánh giá MỚI: câu NGẮN + câu DÀI, câu THẬT chưa model nào train.

Vì sao cần bộ mới: `hardbench200.jsonl` chỉ có 100 câu ja2vi, đã dùng qua 3 vòng
đánh giá nên có nguy cơ bị "tối ưu theo bài thi". Bộ này lấy câu THẬT từ nguồn
CHƯA vào corpus train, phân tầng theo đặc điểm ngôn ngữ để phủ đủ loại khó.

Nguồn (đều loại trùng với 11,88M câu corpus cũ):
  - NGẮN : OPUS OpenSubtitles ja  -> hội thoại, slang, khẩu ngữ
  - DÀI  : OPUS TED2020 ja        -> bài nói, nhiều mệnh đề, đa chủ đề
  (dev holdout của bin_v3 dùng bổ sung nếu TED không đủ)

Không có bản dịch tham chiếu (ref) — judge mù chấm trực tiếp từ câu nguồn, nên
không cần ref, và cũng tránh được thiên vị theo một bản dịch cụ thể.

  python scripts/build_bench400.py --short 100 --long 100
"""
import argparse
import gzip
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = Path("D:/Bit-Translate-data/raw")
OUT = ROOT / "eval" / "bench_new.jsonl"

JA = re.compile(r"[぀-ヿ一-鿿]")
JUNK = re.compile(r"^[\s♪♬~〜ー\-–—…・.,!?！？。、0-9０-９:：]*$")
LATIN_NAME = re.compile(r"^[A-Z][A-Z\s]{2,}[:：]")

# Phân tầng theo ĐẶC ĐIỂM NGÔN NGỮ (không phải chủ đề) — đây mới là thứ quyết định
# độ khó dịch, và khớp với các domain đang theo dõi trong hardbench.
FEATS = [
    ("keigo",   re.compile(r"いただ|くださ|ございま|でござ|いらっしゃ|おっしゃ|なさ|伺|拝見|申し上げ|存じ|承知|恐れ入|頂戴")),
    ("slang",   re.compile(r"めっちゃ|やばい|ヤバ|マジ|まじ|ウケる|それな|ワンチャン|ガチ|しんどい|うざ|きも|だる|っしょ|じゃん")),
    ("idiom",   re.compile(r"を切る|が高い|を打つ|に乗る|が利く|を割って|も借りたい|が上がらない|に落ちない|を濁す|気が|腹を|手を|目を|顔を|骨を|水を")),
    ("negation", re.compile(r"わけでは|ないとは|なくはない|とは限らな|ざるを得な|かねない|ないわけには|どころか|ではない")),
    ("katakana", re.compile(r"[ァ-ヶー]{4,}")),
    ("number",  re.compile(r"[0-9０-９]{2,}|[一二三四五六七八九十百千万億]{2,}[人円年月日個回%％]")),
    ("zeropron", re.compile(r"^(?!.*(私|僕|俺|あなた|君)は).*(てくれ|てあげ|てもら|ておい|てしま)")),
    ("question", re.compile(r"[?？]|ですか|ますか|だろうか")),
    ("clause",  re.compile(r"けれど|ものの|にもかかわらず|ながらも|とはいえ|ため、|ので、|から、")),
]


def clean(s):
    s = unicodedata.normalize("NFKC", s).strip()
    return re.sub(r"\s+", " ", s)


def ok(s, lo, hi):
    n = len(s)
    if not (lo <= n <= hi):
        return False
    if JUNK.match(s) or LATIN_NAME.match(s):
        return False
    # LỜI BÀI HÁT trong phụ đề (đánh dấu ♪) — không đại diện cho câu cần dịch,
    # và thường là thơ/ẩn dụ khiến điểm judge nhiễu. Loại hẳn.
    if "♪" in s or "♬" in s:
        return False
    if s.count("…") > 2 or s.count("―") > 1:
        return False
    return len(JA.findall(s)) / n >= 0.35


def feats_of(s):
    return [name for name, rx in FEATS if rx.search(s)]


def pick(pool, k, rng):
    """Chọn k câu phủ đều đặc điểm: ưu tiên câu mang đặc điểm hiếm trước."""
    by = {name: [] for name, _ in FEATS}
    plain = []
    for s in pool:
        f = feats_of(s)
        if f:
            by[min(f, key=lambda x: len(by[x]))].append(s)
        else:
            plain.append(s)
    out, per = [], max(1, k // (len(FEATS) + 1))
    for name, _ in FEATS:
        rng.shuffle(by[name])
        out += [(s, name) for s in by[name][:per]]
    rng.shuffle(plain)
    out += [(s, "plain") for s in plain[:max(0, k - len(out))]]
    rng.shuffle(out)
    return out[:k]


def load(path, lo, hi, old, limit=400000):
    op = gzip.open if str(path).endswith(".gz") else open
    seen, out = set(), []
    with op(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            s = clean(line)
            if not ok(s, lo, hi) or s in old or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=int, default=100)
    ap.add_argument("--long", type=int, default=100)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(20260726)

    print("nạp corpus cũ để loại trùng...", flush=True)
    old = set()
    with open(ROOT / "data" / "full_11.88m_ja_clean.txt", encoding="utf-8") as f:
        for line in f:
            old.add(line.strip())
    # loại luôn 934k câu OPUS mới (sắp vào corpus vòng 4)
    p = RAW / "os_ja_new.txt"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                old.add(line.strip())
    print(f"  {len(old):,} câu đã dùng/sẽ dùng -> loại khỏi test", flush=True)

    # Câu NGẮN lấy từ CẢ HAI nguồn: OpenSubtitles (hội thoại, khẩu ngữ) và TED2020
    # (câu nói chuẩn hơn). Chỉ dùng OpenSubtitles thì sau khi loại 934k câu sắp
    # train + lời bài hát, ứng viên còn quá ít để phân tầng cho tử tế.
    print("chọn câu NGẮN từ OpenSubtitles + TED2020...", flush=True)
    sh_pool = load(RAW / "os_ja.txt.gz", 15, 35, old)
    n_os = len(sh_pool)
    sh_pool += load(RAW / "ted_ja.txt.gz", 15, 35, old)
    print(f"  ứng viên: {len(sh_pool):,} (OpenSubtitles {n_os:,} + TED {len(sh_pool)-n_os:,})",
          flush=True)
    print("chọn câu DÀI từ TED2020...", flush=True)
    lo_pool = load(RAW / "ted_ja.txt.gz", 60, 150, old)
    print(f"  ứng viên: {len(lo_pool):,}", flush=True)

    rows = []
    for s, f in pick(sh_pool, a.short, rng):
        rows.append({"len": "short", "feat": f, "src": s})
    for s, f in pick(lo_pool, a.long, rng):
        rows.append({"len": "long", "feat": f, "src": s})
    for i, r in enumerate(rows, 1):
        r["id"] = i
        r["dir"] = "ja2vi"

    Path(a.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from collections import Counter
    print(f"\n=== BỘ TEST MỚI: {len(rows)} câu -> {a.out} ===")
    print("ngắn/dài:", dict(Counter(r["len"] for r in rows)))
    print("đặc điểm:", dict(Counter(r["feat"] for r in rows)))


if __name__ == "__main__":
    main()
