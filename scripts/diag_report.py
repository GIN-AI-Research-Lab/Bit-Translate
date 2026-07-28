#!/usr/bin/env python3
"""Đọc kết quả mt_diagnose.py -> tỷ lệ lỗi theo ĐỘ DÀI + dựng mẫu review cho judge.

Hai việc:
 1. Bảng tỷ lệ cờ lỗi theo dải độ dài câu nguồn. Đây là thứ 10k câu mua được mà
    200 câu chấm tay không mua nổi: đường cong lỗi theo độ dài, sai số ±1%.
 2. Dựng MẪU PHÂN TẦNG cho judge. Cỡ mẫu chọn theo mục tiêu thống kê chứ không
    chọn bừa:
      - mỗi bộ dò lấy K câu CÓ cờ -> đo ĐỘ CHÍNH XÁC của bộ dò (bao nhiêu % cờ là
        lỗi thật). K=25 cho sai số ±19% ở p=0,5 — đủ phân biệt "bộ dò dùng được"
        (>70%) với "bộ dò toàn báo nhầm" (<40%).
      - lấy M câu KHÔNG cờ -> đo phần bộ dò BỎ SÓT. Đây mới là số quyết định:
        nếu câu không-cờ mà vẫn sai nhiều thì cả hệ thống dò là vô dụng.
    Ngưỡng tỷ lệ dài (roi_noi_dung / bia_them) tính từ PHÂN VỊ của chính mẻ này,
    không bịa số cứng — vì tỷ lệ dài VI/JA phụ thuộc cặp ngôn ngữ và tokenizer.

  python scripts/diag_report.py
  python scripts/diag_report.py --make-panel eval/judge_diag --per-flag 25 --clean 120
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BANDS = [(0, 40), (40, 60), (60, 80), (80, 110), (110, 150), (150, 10**9)]
FLAGS = ["sot_tieng_nhat", "lap_vong", "cut_cau", "lech_so", "roi_katakana",
         "mat_phu_dinh", "mat_cau_hoi", "roi_noi_dung", "bia_them", "rong"]

# --- BỘ DÒ TÍNH LẠI TỪ (ja, vi) --------------------------------------------
# Để ở đây chứ không ở mt_diagnose.py: cải tiến bộ dò thì chỉ cần chạy lại report,
# KHÔNG phải dịch lại 10k câu (mỗi lần dịch mất ~40 phút).
JA_CH = re.compile(r"[぀-ヿ一-鿿]")
KATA_RUN = re.compile(r"[ァ-ヶー]{4,}")
LATIN = re.compile(r"[A-Za-zÀ-ỹ]{3,}")
END = re.compile(r"[.!?…。]\s*$")
# Số: chữ số Ả Rập CỘNG số Hán. Bản đầu chỉ bắt chữ số nên `10人に1人`, `十分の一`,
# `二割` lọt hết — mà đó đúng là lớp lỗi [140] trên bench ("1/10" -> "không có ai").
KANJI_NUM = "〇一二三四五六七八九十百千万億兆"
NUM_ANY = re.compile(rf"[0-9]+|[{KANJI_NUM}]{{1,6}}(?=[人円年月日回倍割分％%個件名時])")
VI_NUM = re.compile(r"[0-9]+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|nghìn|triệu|tỷ",
                    re.I)
JA_NEG = re.compile(r"(ない|ません|なかった|ませんでした|ぬ[。、]|ず[に、。]|"
                    r"わけではあり|ではない|じゃない|不[可能足適]|未[だ定])")
VI_NEG = re.compile(r"(không|chẳng|chưa|đừng|chớ|phi |thiếu|bất )", re.I)
JA_Q = re.compile(r"(か[。？?]?$|でしょうか|ますか|ですか|のか[。？?]?$)")


def kanji_to_int(s):
    v, cur = 0, 0
    for ch in s:
        d = "〇一二三四五六七八九".find(ch)
        if d >= 0:
            cur = cur * 10 + d if cur else d
        elif ch == "十":
            cur = (cur or 1) * 10
        elif ch == "百":
            cur = (cur or 1) * 100
        elif ch == "千":
            cur = (cur or 1) * 1000
        elif ch in "万億兆":
            v += (cur or 1) * {"万": 10**4, "億": 10**8, "兆": 10**12}[ch]
            cur = 0
    return v + cur


def numbers_ja(s):
    out = set()
    for m in NUM_ANY.findall(s):
        out.add(int(m) if m.isdigit() else kanji_to_int(m))
    return {x for x in out if x}


def redetect(ja, vi):
    if not vi or vi.startswith("__LOI__"):
        return ["rong"]
    f = []
    if JA_CH.search(vi):
        f.append("sot_tieng_nhat")
    w = vi.split()
    if len(w) >= 12:
        c = Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
        if c and c.most_common(1)[0][1] >= 3:
            f.append("lap_vong")
    if not END.search(vi):
        f.append("cut_cau")
    nj = numbers_ja(ja)
    if nj:
        nv = {int(x) for x in re.findall(r"[0-9]+", vi)}
        # chấp nhận nếu số viết bằng chữ tiếng Việt (mười, một trăm…)
        if not nj <= nv and not (len(nj) == 1 and VI_NUM.search(vi)):
            f.append("lech_so")
    if KATA_RUN.search(ja) and not LATIN.search(vi):
        f.append("roi_katakana")
    # PHỦ ĐỊNH: câu Nhật có dấu phủ định mà câu Việt không có từ phủ định nào
    # -> gần như chắc chắn đảo nghĩa. Lớp lỗi này bench đã bắt được nhiều lần.
    if JA_NEG.search(ja) and not VI_NEG.search(vi):
        f.append("mat_phu_dinh")
    if JA_Q.search(ja.rstrip()) and "?" not in vi:
        f.append("mat_cau_hoi")
    return f


def band(n):
    for lo, hi in BANDS:
        if lo <= n < hi:
            return f"{lo}-{hi if hi < 10**9 else '∞'}"
    return "?"


def wilson(k, n):
    if not n:
        return 0.0, 0.0
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/Bit-Translate-data/diag_v4a5.jsonl")
    ap.add_argument("--make-panel", default="")
    ap.add_argument("--per-flag", type=int, default=25)
    ap.add_argument("--clean", type=int, default=120)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    for r in rows:                       # luôn tính lại, bỏ cờ lưu trong file
        r["flags"] = redetect(r["ja"], r["vi"])
    print(f"{len(rows):,} câu đã dịch\n")

    # --- ngưỡng tỷ lệ dài lấy từ phân vị của chính mẻ này ---
    ratios = sorted(r["lv"] / max(r["lj"], 1) for r in rows if r["vi"] and not r["vi"].startswith("__LOI__"))
    lo_th = ratios[int(len(ratios) * 0.05)]
    hi_th = ratios[int(len(ratios) * 0.97)]
    print(f"tỷ lệ dài VI/JA: p5={lo_th:.2f}  p50={ratios[len(ratios)//2]:.2f}  p97={hi_th:.2f}")
    print(f"  -> 'roi_noi_dung' nếu < {lo_th:.2f} | 'bia_them' nếu > {hi_th:.2f}\n")
    for r in rows:
        rt = r["lv"] / max(r["lj"], 1)
        if r["vi"] and not r["vi"].startswith("__LOI__"):
            if rt < lo_th:
                r["flags"] = list(set(r["flags"] + ["roi_noi_dung"]))
            elif rt > hi_th:
                r["flags"] = list(set(r["flags"] + ["bia_them"]))

    # --- bảng theo độ dài ---
    by = defaultdict(list)
    for r in rows:
        by[band(r["lj"])].append(r)
    order = [f"{lo}-{hi if hi < 10**9 else '∞'}" for lo, hi in BANDS]
    print(f"{'dải ký tự':>10} {'n':>7} {'CÓ ÍT NHẤT 1 CỜ':>17} {'KTC 95%':>14}")
    for b in order:
        v = by.get(b, [])
        if not v:
            continue
        k = sum(1 for r in v if r["flags"])
        lo, hi = wilson(k, len(v))
        print(f"{b:>10} {len(v):>7,} {100*k/len(v):16.1f}% {f'[{lo:.0f}-{hi:.0f}]':>14}")
    print()

    # --- từng cờ theo độ dài ---
    print(f"{'cờ lỗi':>16}" + "".join(f"{b:>10}" for b in order) + f"{'TỔNG':>8}")
    for f in FLAGS:
        cells, tot = [], 0
        for b in order:
            v = by.get(b, [])
            k = sum(1 for r in v if f in r["flags"])
            tot += k
            cells.append(f"{100*k/len(v):9.1f}%" if v else f"{'-':>10}")
        if tot:
            print(f"{f:>16}" + "".join(cells) + f"{tot:>8,}")
    print()

    cnt = Counter(f for r in rows for f in r["flags"])
    nflag = sum(1 for r in rows if r["flags"])
    print(f"CÓ cờ: {nflag:,} ({100*nflag/len(rows):.1f}%) | SẠCH: {len(rows)-nflag:,}")

    if not a.make_panel:
        print("\n(thêm --make-panel <thư mục> để dựng mẫu cho judge)")
        return

    rng = random.Random(a.seed)
    pick, why = {}, {}
    for f in FLAGS:
        v = [r for r in rows if f in r["flags"]]
        rng.shuffle(v)
        for r in v[:a.per_flag]:
            pick[r["ja"]] = r
            why.setdefault(r["ja"], []).append(f)
    clean = [r for r in rows if not r["flags"] and r["ja"] not in pick]
    rng.shuffle(clean)
    for r in clean[:a.clean]:
        pick[r["ja"]] = r
        why[r["ja"]] = ["SACH"]

    d = Path(a.make_panel)
    d.mkdir(parents=True, exist_ok=True)
    items = list(pick.values())
    rng.shuffle(items)
    with (d / "panel.jsonl").open("w", encoding="utf-8") as fo:
        for n, r in enumerate(items):
            fo.write(json.dumps({"i": n, "ja": r["ja"], "vi": r["vi"],
                                 "lj": r["lj"]}, ensure_ascii=False) + "\n")
    with (d / "key.json").open("w", encoding="utf-8") as fo:
        json.dump({str(n): why[r["ja"]] for n, r in enumerate(items)}, fo,
                  ensure_ascii=False, indent=1)
    print(f"\n-> {d}/panel.jsonl  {len(items)} câu (judge KHÔNG thấy cờ)")
    print(f"-> {d}/key.json     (cờ thật, để giải mã sau khi chấm)")
    print(f"   phân bố: {Counter(w for ws in why.values() for w in ws).most_common()}")


if __name__ == "__main__":
    main()
