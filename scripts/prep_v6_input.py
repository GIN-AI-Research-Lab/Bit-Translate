#!/usr/bin/env python3
"""Gộp hai đợt đào thành một file đầu vào cho KD, khử trùng lặp + lọc rác nhẹ.

Đo được trên mẫu 300 câu (Gemini chấm): 74,3% tốt / 22,9% tạm / 2,9% RÁC.
2,9% của 2,25M = ~65k câu spam SEO / văn máy sinh lảm nhảm. Lọc bằng LUẬT (rẻ),
không gọi LLM cho 2,25M câu.

Dấu hiệu rác đã quan sát trong mẫu:
  - nhồi từ khoá sản phẩm/website, câu ghép rời rạc không mạch lạc
  - lặp cùng một cụm nhiều lần trong một câu
  - quá nhiều chữ số / ký hiệu so với chữ
  - quảng cáo tuyển dụng - hẹn hò - vay tiền (miền spam nặng nhất của CC-100)

  python scripts/prep_v6_input.py
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

D = Path("D:/Bit-Translate-data")

SPAM = re.compile(
    r"出会い系|出合いけい|セフレ|ナンパ|パパ活|援交|風俗嬢|キャバ|ホスト|"
    r"消費者金融|即日融資|闇金|借金地獄|債務整理|過払い|"
    r"クリック|今すぐ登録|無料登録|お申し込みはこちら|詳しくはこちら|"
    r"稼げる|副業で稼|月収\d|日給\d|高収入|"
    r"激安|最安値|通販サイト|口コミランキング|人気ランキング\d")
# lặp cụm >=4 ký tự ba lần trở lên trong một câu -> nhồi từ khoá
REP = re.compile(r"(.{4,12})\1{2,}")
JA = re.compile(r"[ぁ-んァ-ヶ一-龥]")
NONJA = re.compile(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ/／\-–—_=+*#|｜<>\[\]()（）]")


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def bad(s):
    n = len(s)
    if SPAM.search(s):
        return "spam"
    if REP.search(s):
        return "lap"
    ja = len(JA.findall(s))
    if ja < n * 0.45:
        return "it_chu_nhat"
    if len(NONJA.findall(s)) > n * 0.30:
        return "nhieu_so_ky_hieu"
    if s.count("、") > 12 or s.count("・") > 6:
        return "liet_ke_rac"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+",
                    default=[str(D / "raw" / "mined_rare.txt"),
                             str(D / "raw" / "mined_skills.txt")])
    ap.add_argument("--out", default=str(D / "raw" / "v6_todo.txt"))
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=180)
    a = ap.parse_args()

    # loại câu đã có trong corpus v5 (an toàn kép — bộ đào đã loại rồi)
    used = set()
    with open(D / "kd_v5_merged.jsonl", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                used.add(dig(json.loads(line)["ja"]))
            except Exception:
                pass
    print(f"corpus v5: {len(used):,} câu")

    seen = set()
    drop = Counter()
    kept = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for src in a.src:
            p = Path(src)
            if not p.exists():
                print(f"  BỎ QUA {p.name} (không có)")
                continue
            n = k = 0
            for line in p.open(encoding="utf-8", errors="ignore"):
                s = line.strip()
                if not s:
                    continue
                n += 1
                if not (a.min_len <= len(s) <= a.max_len):
                    drop["do_dai"] += 1
                    continue
                h = dig(s)
                if h in used or h in seen:
                    drop["trung"] += 1
                    continue
                r = bad(s)
                if r:
                    drop[r] += 1
                    continue
                seen.add(h)
                fo.write(s + "\n")
                k += 1
                kept += 1
            print(f"  {p.name:22} {n:>10,} -> giữ {k:,}")

    tot = kept + sum(drop.values())
    print(f"\nloại {sum(drop.values()):,}/{tot:,} ({100*sum(drop.values())/tot:.1f}%):")
    for r, v in drop.most_common():
        print(f"   {r:<18}{v:>10,}")
    print(f"\n=> {kept:,} câu -> {a.out}")


if __name__ == "__main__":
    main()
