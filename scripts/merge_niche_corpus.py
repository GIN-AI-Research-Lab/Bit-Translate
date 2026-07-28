#!/usr/bin/env python3
"""Gộp data NICHE (idiom/keigo) vào corpus KD 10,5M + OVERSAMPLE phần niche.

Vì sao oversample: niche chỉ ~110k cặp = 1% corpus 10,5M — quá loãng để model
học được. Nhân ×N cho nó có trọng số thật trong mix (HANDOFF_KD100M.md §8).
Nhân bản là cố ý, KHÔNG dedup phần niche sau khi nhân.

Lọc lại toàn bộ niche qua QC của gen_niche_kd (bắt cả rác "ja lẫn tiếng Việt"
mà lần sinh trước chưa chặn) + dedup theo câu ja so với chính corpus gốc.

  python scripts/merge_niche_corpus.py --oversample 4
  # -> data/synthetic/kd_clean/kd_filtered_niche.jsonl

Sau đó chạy lại:
  python scripts/prep_clean_split.py data/synthetic/kd_clean/kd_filtered_niche.jsonl
  python scripts/binarize_ja2vi.py
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gen_niche_kd import qc  # noqa: E402
from filter_kd_full import norm, check  # noqa: E402

BASE = ROOT / "data" / "synthetic" / "kd_clean" / "kd_filtered.jsonl"
NICHE = ROOT / "data" / "synthetic" / "gen_niche"
# Ổ E đầy (40GB, 100%) -> output ~2,5GB phải ghi sang ổ D.
OUT = Path("D:/Bit-Translate-data/kd_filtered_niche.jsonl")


NL = chr(10)


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


# Slang có 38,5% cặp mang emoji ở CẢ ja và vi — cân bằng nên model chỉ học COPY
# emoji theo input (vô hại). Nhưng 0,1% có emoji CHỈ ở vi: đó là dạy model tự bịa
# emoji vào bản dịch → loại.
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️⬀-⯿]")


def emoji_lech(ja, vi):
    return bool(EMOJI.search(vi)) and not bool(EMOJI.search(ja))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oversample", type=int, default=4,
                   help="số lần nhân bản mỗi cặp niche trong mix (mặc định 4). "
                        "VÒNG 4 dùng 2: tỷ lệ 10,42% của vòng 3 đã đẩy model lệch "
                        "phong cách mà KHÔNG cải thiện trên bench câu thật.")
    p.add_argument("--os-map", default="",
                   help="nhân bản RIÊNG theo mode, vd 'termgap=4,sciterm=3'. "
                        "Lý do tách: termgap nhắm thuật ngữ xuất hiện <=5 lần trong "
                        "10,5M câu — ×2 chỉ cho ~18 lần gặp, quá ít để nhớ ánh xạ. "
                        "Các mode phong cách (idiom/keigo/slang) thì ngược lại: "
                        "nhiều quá làm lệch giọng văn (bài học vòng 3).")
    p.add_argument("--extra-real", default="",
                   help="thêm nguồn data THẬT (jsonl có ja/vi), phân tách bằng dấu phẩy. "
                        "vd D:/Bit-Translate-data/raw/kd_os_new.jsonl (934k câu OPUS đã KD)")
    p.add_argument("--base", default=str(BASE))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--modes",
                   default="idiom,keigo,slang,negation,katakana,long,conv,zeropron,"
                           "sciterm,termgap")
    p.add_argument("--niche-dirs", default=f"{NICHE},D:/Bit-Translate-data/gen_niche",
                   help="các thư mục chứa {mode}.jsonl, phân tách bằng dấu phẩy "
                        "(ổ E đầy nên data mới nằm ở ổ D)")
    a = p.parse_args()

    dirs = [Path(d) for d in a.niche_dirs.split(",")]

    # 1) đọc + QC niche
    niche, stats = [], Counter()
    for mode in a.modes.split(","):
        found = [d / f"{mode}.jsonl" for d in dirs if (d / f"{mode}.jsonl").exists()]
        if not found:
            print(f"  bỏ qua {mode} (không thấy trong {', '.join(str(d) for d in dirs)})")
            continue
        n = 0
        for f in found:
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_hong"] += 1
                    continue
                n += 1
                ja, vi = norm(o.get("ja", "")), norm(o.get("vi", ""))
                why = qc(mode, ja, vi, o)
                if not why and emoji_lech(ja, vi):
                    why = "emoji_chi_o_vi"
                if why:
                    stats[f"{mode}:{why}"] += 1
                    continue
                niche.append((ja, vi, mode))
        print(f"  {mode}: đọc {n:,} từ {len(found)} file")

    # dedup trong nội bộ niche
    seen, uniq = set(), []
    for ja, vi, mode in niche:
        h = dig(ja)
        if h in seen:
            stats["niche_trung_lap"] += 1
            continue
        seen.add(h)
        uniq.append((ja, vi, mode))
    per = Counter(x[2] for x in uniq)
    print(f"\nNiche qua QC + dedup: {len(uniq):,} cặp")
    for m, c in per.most_common():
        print(f"  {m:10s} {c:>8,}")

    # 2) stream corpus gốc ra out, loại câu ja trùng với niche (niche là bản tốt hơn)
    nbase = ndrop = 0
    with open(a.base, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fo:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if dig(o.get("ja", "")) in seen:
                ndrop += 1
                continue
            fo.write(line + "\n")
            nbase += 1

        # 2b) THÊM DATA THẬT (OPUS đã KD; sau này CC-100) — KHÔNG oversample.
        # Vòng 3 phần thêm toàn là data tổng hợp, và bench câu thật cho thấy cải
        # thiện KHÔNG khái quát (v3 66% = v2 66%). Data thật giữ model bám phân bố
        # tiếng Nhật thực tế thay vì phân bố do chính mình sinh ra.
        n_real = 0
        for rp in [x.strip() for x in a.extra_real.split(",") if x.strip()]:
            rpp = Path(rp)
            if not rpp.exists():
                print(f"  bỏ qua {rp} (không có)")
                continue
            got = 0
            for line in rpp.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ja, vi = norm(o.get("ja", "")), norm(o.get("vi", ""))
                if check(ja, vi) or emoji_lech(ja, vi):
                    stats["real:qc"] += 1
                    continue
                h = dig(ja)
                if h in seen:
                    stats["real:trung"] += 1
                    continue
                seen.add(h)
                fo.write(json.dumps({"id": f"real_{n_real}", "ja": ja, "vi": vi},
                                    ensure_ascii=False) + NL)
                n_real += 1
                got += 1
            print(f"  + data thật {rpp.name}: {got:,} cặp")

        # 3) ghi niche ×oversample (mỗi mode có thể có hệ số riêng qua --os-map)
        osmap = {}
        for kv in [x.strip() for x in a.os_map.split(",") if x.strip()]:
            m, _, v = kv.partition("=")
            osmap[m.strip()] = int(v)
        nout = 0
        per_out = Counter()
        for ja, vi, mode in uniq:
            for _ in range(osmap.get(mode, a.oversample)):
                fo.write(json.dumps({"id": f"{mode}_{nout}", "ja": ja, "vi": vi},
                                    ensure_ascii=False) + "\n")
                nout += 1
                per_out[mode] += 1

    tot = nbase + nout + n_real
    print(f"\n=== GỘP XONG ===")
    print(f"Corpus gốc giữ  : {nbase:,} (loại {ndrop:,} câu ja trùng niche)")
    print(f"Data THẬT thêm  : {n_real:,}")
    print(f"Niche (nhân bản): {nout:,}")
    for m, c in per_out.most_common():
        print(f"    {m:10s} x{osmap.get(m, a.oversample)} -> {c:>9,}")
    print(f"TỔNG            : {tot:,} | niche chiếm {100*nout/max(1,tot):.2f}%")
    if stats:
        print("Rớt/bỏ:")
        for k, v in stats.most_common(12):
            print(f"  {k:34s} {v:>7,}")
    print(f"Output          : {a.out}")


if __name__ == "__main__":
    main()
