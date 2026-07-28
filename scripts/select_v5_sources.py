#!/usr/bin/env python3
"""Chọn câu Nhật cho corpus vòng 5 — THEO NGÂN SÁCH TOKEN + THEO ĐỘ PHỦ ITEM.

Vì sao không lấy hết 14,6M câu đã thu: ngân sách GPU ($12,91 ≈ 13.000 step ×
131k token = 1,7 tỷ token) chỉ cho 0,86 epoch nếu train hết — chưa trọn một lượt,
mà lịch LR lại giảm dần nên data gặp muộn gần như không học được. Giữ ~1,9 epoch
(bằng vòng 4, đã biết là đủ) thì corpus phải ~900M token: 475M cũ + ~425M mới.

Vì sao không chỉ lọc theo ĐỘ DÀI: các lỗi đo được chia làm hai bản chất khác nhau
và cần chiến lược ngược nhau —

  KỸ NĂNG (cần BỀ RỘNG, mỗi ví dụ chỉ gặp 1-2 lần cũng được):
    - giữ cấu trúc câu dài   : bench long 52%, probe non-short 73%
    - chép/đổi số liệu       : lech_so 94% chính xác, tăng 3,8%->16,1% theo độ dài
  ITEM (cần LẶP >= ~50 lần, dưới ngưỡng là model bịa):
    - thành ngữ / khẩu ngữ   : 15/25 câu ngắn sai
    - tên riêng hay gặp      : 川端大臣 -> "Bộ trưởng Ngoại giao sông"

Ngưỡng 50 lấy từ đo đạc vòng 4: thuật ngữ đạt 78 lần trong corpus (~234 lần gặp
qua 3 epoch) thì được vá; còn <=5 lần thì dịch bậy. Xem HANDOFF §0-QUINQUIES.

Đầu ra là DANH SÁCH CÂU NHẬT để KD dịch — phải chọn TRƯỚC khi dịch, vì dịch 3M câu
mất 15 giờ còn dịch hết 14,6M mất 73 giờ.

  python scripts/select_v5_sources.py --budget-tokens 425000000
  # -> D:/Bit-Translate-data/raw/v5_select.txt  + báo cáo độ phủ item
"""
import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent
RAW = Path("D:/Bit-Translate-data/raw")
SP = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
# tỷ lệ token VI/JA = 1,38 (đo trên 3.000 cặp KD thật), +3 cho BOS/tag/EOS
VI_RATIO = 1.38

KANJI_NUM = "〇一二三四五六七八九十百千万億兆"
NUM = re.compile(rf"[0-9]+|[{KANJI_NUM}]{{1,6}}(?=[人円年月日回倍割分％%個件名時])")
# Tên riêng: chuỗi katakana dài (tên nước ngoài) hoặc kanji + hậu tố chức danh/địa danh
NAME = re.compile(r"[ァ-ヶー]{3,}|[一-鿿]{2,4}(?=(?:大臣|委員|議員|総理|長官|知事|"
                  r"党|省|庁|局|県|市|区|町|村|会社|銀行|大学))")
CLAUSE = re.compile(r"けれど|ものの|にもかかわらず|ながらも|とはいえ|ため、|ので、|から、|が、|"
                    r"という|ように|ことで|に対して|について|における")
IDIOM = re.compile(
    r"(手を[打貸借出焼染]|足を[引運洗踏]|目を[通見疑光]|耳を[傾貸澄]|口を[出挟揃]|"
    r"腹を[割立決探]|頭を[抱下悩痛]|気が[付済重引利]|気を[付使配遣]|水を[差向]|"
    r"猫の手|馬の耳|石の上|棚から|藪から|渡りに|背に腹|二の足|一石二鳥|"
    r"[一二三四五六七八九十百千]{1}[一-鿿]{1}[一二三四五六七八九十百千]{1}[一-鿿]{1})")
COLLOQ = re.compile(r"(ね[。！？]|よ[。！？]|じゃん|だろ[うっ]?|ちゃう|ちゃっ|なきゃ|"
                    r"やっぱ|めっちゃ|マジ|ヤバ|っけ|でしょ[。？])")


def pair_tokens(ja):
    return int(len(SP.encode(ja)) * (1 + VI_RATIO)) + 3


def read(p, lo=0, hi=10**9):
    if not Path(p).exists():
        return
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if lo <= len(s) < hi:
                yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-tokens", type=int, default=425_000_000)
    ap.add_argument("--out", default=str(RAW / "v5_select.txt"))
    ap.add_argument("--item-floor", type=int, default=50,
                    help="mỗi item (thành ngữ/tên) phải đạt tối thiểu bấy nhiêu lần")
    a = ap.parse_args()

    # Chia ngân sách theo BẰNG CHỨNG, không chia đều:
    #   - câu dài là lỗi nặng nhất (48% sai) -> phần lớn nhất
    #   - số liệu là lỗi đo được rõ nhất (bộ dò 94% chính xác) -> phần riêng
    #   - khẩu ngữ/thành ngữ tuy chỉ 15/25 câu ngắn nhưng token rẻ (53/cặp)
    PLAN = [
        ("caudai_kokkai", 0.46, RAW / "kokkai_ja.txt", 110, 220),
        ("soluong_kokkai", 0.19, RAW / "kokkai_ja.txt", 40, 110),
        ("khaungu_cc100", 0.23, RAW / "cc100_colloq.txt", 15, 60),
        ("chuyennganh_cc100", 0.12, RAW / "cc100_pick.txt", 60, 180),
    ]

    picked, seen = [], set()
    stats = defaultdict(lambda: {"n": 0, "tok": 0})
    item_cnt = Counter()

    for name, share, path, lo, hi in PLAN:
        budget = int(a.budget_tokens * share)
        used = 0
        # chấm điểm rồi lấy điểm cao trước — KHÔNG lấy tuần tự, vì phần đầu file
        # là năm 2010 (Quốc hội) hoặc một vùng crawl (CC-100), lấy tuần tự sẽ lệch.
        cand = []
        for s in read(path, lo, hi):
            if s in seen:
                continue
            nn = len(set(NUM.findall(s)))
            nm = len(set(NAME.findall(s)))
            nc = len(CLAUSE.findall(s))
            ni = len(IDIOM.findall(s))
            nk = len(COLLOQ.findall(s))
            if name == "caudai_kokkai":
                sc = 2.0 * min(4, nc) + 1.0 * min(3, nn) + 1.0 * min(3, nm)
            elif name == "soluong_kokkai":
                sc = 3.0 * min(4, nn) + 1.0 * min(2, nm)
                if nn == 0:
                    continue
            elif name == "khaungu_cc100":
                sc = 4.0 * min(2, ni) + 2.0 * min(3, nk)
                if ni == 0 and nk == 0:
                    continue
            else:
                sc = 1.5 * min(3, nc) + 1.0 * min(3, nn) + 1.5 * min(2, nm)
            cand.append((sc, s))
        cand.sort(key=lambda x: -x[0])
        for sc, s in cand:
            t = pair_tokens(s)
            if used + t > budget:
                break
            seen.add(s)
            picked.append(s)
            used += t
            stats[name]["n"] += 1
            stats[name]["tok"] += t
            for m in IDIOM.findall(s):
                item_cnt["idiom:" + (m if isinstance(m, str) else m[0])] += 1
            for m in NAME.findall(s):
                item_cnt["name:" + m] += 1
        print(f"  {name:20} {stats[name]['n']:>9,} câu | {used/1e6:>6.1f}M token "
              f"| ứng viên {len(cand):,}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in picked:
            f.write(s + "\n")

    tot_t = sum(v["tok"] for v in stats.values())
    print(f"\n=== CHỌN XONG ===")
    print(f"Câu   : {len(picked):,}")
    print(f"Token : {tot_t/1e6:.0f}M (ngân sách {a.budget_tokens/1e6:.0f}M)")
    print(f"Corpus v5 dự kiến: 475M (cũ) + {tot_t/1e6:.0f}M = {(475e6+tot_t)/1e6:.0f}M token")
    print(f"  -> 1.703M / {(475e6+tot_t)/1e6:.0f}M = {1703e6/(475e6+tot_t):.2f} epoch")

    # ĐỘ PHỦ ITEM — phần trả lời "số lần xuất hiện đủ để không bịa"
    idi = {k: v for k, v in item_cnt.items() if k.startswith("idiom:")}
    nam = {k: v for k, v in item_cnt.items() if k.startswith("name:")}
    print(f"\n=== ĐỘ PHỦ ITEM (ngưỡng {a.item_floor} lần) ===")
    for lbl, d in [("thành ngữ", idi), ("tên riêng", nam)]:
        if not d:
            continue
        v = sorted(d.values(), reverse=True)
        ok = sum(1 for x in v if x >= a.item_floor)
        print(f"  {lbl:10} {len(v):>7,} mục | đạt ngưỡng {ok:,} ({100*ok/len(v):.0f}%) "
              f"| trung vị {v[len(v)//2]} lần | top {v[0]}")
        print(f"             -> {len(v)-ok:,} mục DƯỚI ngưỡng, cần oversample hoặc sinh thêm")
    print(f"\nOutput: {a.out}")


if __name__ == "__main__":
    main()
