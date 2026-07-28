#!/usr/bin/env python3
"""54 miền đã đủ chưa? — KẾT HỢP hai cách kiểm, không tin cách nào một mình.

  Cách A (LLM): hỏi Gemini 3 lần độc lập "54 miền này thiếu gì" -> ra danh sách ỨNG VIÊN.
                Điểm yếu: LLM có thể bịa miền nghe hợp lý nhưng thực tế hiếm gặp.
  Cách B (đếm): đếm thuật ngữ thật trong text thật. Điểm yếu: chỉ đo được cái mình
                đã nghĩ ra để đếm — không tự phát hiện miền chưa từng nghĩ tới.

  KẾT HỢP: A sinh ứng viên -> B phán quyết bằng số. Mỗi ứng viên rơi vào 1 trong 3:
    (1) THIẾU THẬT   — từ phổ biến trong text thật NHƯNG hiếm trong corpus
    (2) ĐÃ CÓ RỒI    — từ đã đủ dày trong corpus (54 miền cũ đã phủ gián tiếp)
    (3) KHÔNG ĐÁNG   — từ hiếm cả trong text thật => LLM bịa ra miền không tồn tại

Mốc so sánh: 54 miền hiện có, độ phủ TB 40,8%. Ứng viên phủ THẤP HƠN rõ mốc này
mà vẫn phổ biến ngoài đời thì mới đáng thêm vào taxonomy.

  python scripts/verify_candidate_domains.py
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import ahocorasick_rs as ar

D = Path("D:/Bit-Translate-data")


def scan(path, words, A, field=None, limit=None):
    cnt = Counter()
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if field:
                try:
                    s = json.loads(line)[field]
                except Exception:
                    continue
            else:
                s = line.strip()
            if not s:
                continue
            n += 1
            for i, _, _ in A.find_matches_as_indexes(s, overlapping=True):
                cnt[words[i]] += 1
            if limit and n >= limit:
                break
    return cnt, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=str(D / "candidate_terms.json"))
    ap.add_argument("--old", default=str(D / "domain_terms.json"))
    ap.add_argument("--min-occ", type=int, default=60)
    ap.add_argument("--ref-lines", type=int, default=5_000_000)
    ap.add_argument("--out", default="eval/candidate_verdict.json")
    a = ap.parse_args()

    cand = json.loads(Path(a.cand).read_text(encoding="utf-8"))
    old = json.loads(Path(a.old).read_text(encoding="utf-8"))

    words = sorted({o["t"] for v in cand.values() for o in v})
    A = ar.AhoCorasick(words, implementation=ar.Implementation.DFA)
    print(f"{len(cand)} miền ứng viên | {len(words):,} thuật ngữ mới\n", flush=True)

    print("quét corpus v5 (13,1M câu)...", flush=True)
    cur, n_cur = scan(D / "kd_v5_merged.jsonl", words, A, field="ja")

    print(f"quét text thật đối chứng ({a.ref_lines:,} dòng cc100)...", flush=True)
    ref, n_ref = scan(D / "raw" / "cc100_pick.txt", words, A, limit=a.ref_lines)

    # mốc: 54 miền cũ phủ TB bao nhiêu (lấy từ term_coverage đã đo)
    cov = json.loads(Path("eval/term_coverage.json").read_text(encoding="utf-8"))
    oldc = cov["counts_cur"]
    old_pct = 100 * sum(1 for t in oldc if oldc[t] >= a.min_occ) / len(oldc)
    # mật độ tham chiếu của 54 miền cũ trong text thật -> để so "phổ biến hay không"
    ow = sorted(oldc)
    OA = ar.AhoCorasick(ow, implementation=ar.Implementation.DFA)
    print(f"quét đối chứng cho 54 miền cũ...", flush=True)
    oref, _ = scan(D / "raw" / "cc100_pick.txt", ow, OA, limit=a.ref_lines)
    old_ref_med = sorted(oref[t] for t in ow)[len(ow) // 2]

    print(f"\nMỐC 54 miền cũ: phủ {old_pct:.1f}% | trung vị tần suất ngoài đời "
          f"{old_ref_med} lần/{n_ref:,} dòng\n")

    print(f"{'miền ứng viên':<24}{'phủ':>6}{'  trung vị':>11}{'  ngoài đời':>12}  phán quyết")
    print("-" * 82)
    res = {}
    for name, items in sorted(cand.items()):
        ts = [o["t"] for o in items]
        pct = 100 * sum(1 for t in ts if cur[t] >= a.min_occ) / len(ts)
        med_cur = sorted(cur[t] for t in ts)[len(ts) // 2]
        med_ref = sorted(ref[t] for t in ts)[len(ts) // 2]
        # phán quyết
        if med_ref < old_ref_med / 4:
            v = "(3) KHÔNG ĐÁNG - hiếm cả ngoài đời"
        elif pct >= old_pct:
            v = "(2) ĐÃ CÓ RỒI - 54 miền phủ gián tiếp"
        else:
            v = f"(1) THIẾU THẬT - kém mốc {old_pct - pct:.0f} điểm"
        res[name] = {"pct": round(pct, 1), "med_cur": med_cur, "med_ref": med_ref,
                     "verdict": v, "n": len(ts)}
        print(f"{name:<24}{pct:>5.0f}%{med_cur:>11,}{med_ref:>12,}  {v}")

    print(f"\n--- ví dụ từ THIẾU nhất mỗi miền (0 lần trong corpus) ---")
    for name, items in sorted(cand.items()):
        z = [o["t"] for o in items if cur[o["t"]] == 0][:10]
        if z:
            print(f"  {name:<24} {' '.join(z)}")

    Path(a.out).write_text(json.dumps(
        {"old_baseline_pct": round(old_pct, 1), "old_ref_median": old_ref_med,
         "n_ref_lines": n_ref, "candidates": res,
         "counts_cur": {t: cur[t] for t in words},
         "counts_ref": {t: ref[t] for t in words}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
