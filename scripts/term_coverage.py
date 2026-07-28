#!/usr/bin/env python3
"""Đo ĐỘ PHỦ THUẬT NGỮ của từng miền — thay cho cách đếm câu bằng regex (đã chứng minh
không đủ tin, xem DOMAIN_MAP.md §1b).

Đo cái gì:
  1. Mỗi thuật ngữ xuất hiện bao nhiêu lần trong corpus v5 (phía tiếng Nhật)?
  2. Bao nhiêu % thuật ngữ của miền đạt ngưỡng >=60 lần (ngưỡng "model học được",
     đo ở vòng 4: 78 lần thì được vá, <=5 lần thì bịa)?
  3. Nếu đổ thêm TOÀN BỘ kho chưa dùng trên ổ D vào thì độ phủ lên bao nhiêu?
     -> tách bạch "miền thiếu data" và "miền thiếu NGUỒN". Đây là con số quyết định:
        miền nào kho lấp được thì chỉ cần lọc + KD dịch; miền nào kho KHÔNG lấp được
        thì mới phải đi thu thập nguồn mới.

Chia 3 tầng (cơ bản / trung cấp / chuyên sâu) để thấy ĐỘ SÂU: phủ 100% từ cơ bản mà
0% từ chuyên sâu nghĩa là model trôi câu phổ thông nhưng vỡ khi gặp văn bản thật.

Khớp chuỗi chính xác bằng Aho-Corasick -> không có khái niệm precision/recall của bộ
phân loại, không thiên lệch theo người viết bộ lọc.

  python scripts/term_coverage.py
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick

D = Path("D:/Bit-Translate-data")
MIN_OCC = 60


def build(terms):
    A = ahocorasick.Automaton()
    for t in terms:
        A.add_word(t, t)
    A.make_automaton()
    return A


def dig(s):
    # lưu hash 8 byte thay vì chuỗi: 13,1M câu -> ~800MB thay vì ~5GB
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def scan_jsonl(path, A, cnt, used, field="ja"):
    """Một lượt duy nhất: vừa đếm thuật ngữ, vừa dựng tập câu đã dùng."""
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                s = json.loads(line)[field]
            except Exception:
                continue
            n += 1
            used.add(dig(s))
            for _, t in A.iter(s):
                cnt[t] += 1
    return n


def scan_txt(path, A, cnt, skip):
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or dig(s) in skip:
                continue
            n += 1
            for _, t in A.iter(s):
                cnt[t] += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--terms", default=str(D / "domain_terms.json"))
    p.add_argument("--out", default="eval/term_coverage.json")
    p.add_argument("--min-occ", type=int, default=MIN_OCC)
    a = p.parse_args()

    doms = json.loads(Path(a.terms).read_text(encoding="utf-8"))
    all_terms = sorted({o["t"] for v in doms.values() for o in v})
    print(f"{len(doms)} miền | {len(all_terms):,} thuật ngữ\n")
    A = build(all_terms)

    # --- 1. corpus v5 đang train (một lượt: đếm + dựng tập đã dùng) ---
    print("quét corpus v5 (13,1M cặp)...", flush=True)
    cur = Counter()
    used = set()
    n1 = scan_jsonl(D / "kd_v5_merged.jsonl", A, cur, used)
    for pth in ["raw/v5_planC.txt", "raw/v5_select.txt"]:
        fp = D / pth
        if fp.exists():
            for line in fp.open(encoding="utf-8", errors="ignore"):
                used.add(dig(line.strip()))
    print(f"  {n1:,} câu | {len(used):,} câu đã dùng\n", flush=True)

    # --- 2. kho CHƯA DÙNG: loại đúng những câu đã nằm trong corpus ---
    print("quét kho CHƯA DÙNG...", flush=True)
    pool = Counter()
    for pth in ["raw/kokkai_ja.txt", "raw/cc100_pick.txt", "raw/cc100_colloq.txt"]:
        fp = D / pth
        if not fp.exists():
            continue
        n = scan_txt(fp, A, pool, skip=used)
        print(f"  {fp.name:22} {n:,} câu chưa dùng", flush=True)

    # --- 3. tổng hợp theo miền x tầng ---
    res = {}
    for name, items in doms.items():
        by_tier = defaultdict(list)
        for o in items:
            by_tier[o["tier"]].append(o["t"])
        d = {"n_terms": len(items), "tiers": {}}
        for tier in (1, 2, 3):
            ts = by_tier.get(tier, [])
            if not ts:
                continue
            now = sum(1 for t in ts if cur[t] >= a.min_occ)
            after = sum(1 for t in ts if cur[t] + pool[t] >= a.min_occ)
            d["tiers"][tier] = {
                "n": len(ts),
                "now_pct": round(100 * now / len(ts), 1),
                "after_pct": round(100 * after / len(ts), 1),
                "median_now": sorted(cur[t] for t in ts)[len(ts) // 2],
                "median_pool": sorted(pool[t] for t in ts)[len(ts) // 2],
                "still_missing": sorted(
                    (t for t in ts if cur[t] + pool[t] < a.min_occ),
                    key=lambda t: cur[t] + pool[t])[:12],
            }
        ts = [o["t"] for o in items]
        d["now_pct"] = round(100 * sum(1 for t in ts if cur[t] >= a.min_occ) / len(ts), 1)
        d["after_pct"] = round(100 * sum(1 for t in ts if cur[t] + pool[t] >= a.min_occ) / len(ts), 1)
        d["n_still_missing"] = sum(1 for t in ts if cur[t] + pool[t] < a.min_occ)
        res[name] = d

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"min_occ": a.min_occ, "domains": res,
         "counts_cur": {t: cur[t] for t in all_terms},
         "counts_pool": {t: pool[t] for t in all_terms}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'miền':<26}{'giờ':>7}{'+kho':>7}{'  cơ bản':>10}{'trung':>7}{'sâu':>6}  {'còn thiếu':>9}")
    print("-" * 78)
    for name, d in sorted(res.items(), key=lambda x: x[1]["after_pct"]):
        t = d["tiers"]
        f = lambda k, kk: f"{t[k][kk]:.0f}" if k in t else "-"
        print(f"{name:<26}{d['now_pct']:>6.0f}%{d['after_pct']:>6.0f}%"
              f"{f(1,'after_pct'):>9}%{f(2,'after_pct'):>6}%{f(3,'after_pct'):>5}%"
              f"  {d['n_still_missing']:>9}")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
