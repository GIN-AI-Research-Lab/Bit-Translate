#!/usr/bin/env python3
"""Lọc CC-100 ja CÓ CHỦ ĐÍCH — chỉ lấy câu trị đúng 2 điểm yếu đã đo được.

Bench câu thật (eval/bench_compare.html) cho thấy BitNet v3:
  - câu NGẮN: 4,35 — ngang/hơn Google (4,25)  => KHÔNG cần thêm câu ngắn
  - câu DÀI : 3,57 — thua Google 0,9 điểm     => cần câu dài
  - gãy ở THUẬT NGỮ, không phải cấu trúc      => cần câu giàu thuật ngữ
Nên script này KHÔNG lấy ngẫu nhiên mà CHẤM ĐIỂM từng câu rồi giữ câu điểm cao.

CC-100 là web crawl thô: phần lớn là blog đời thường, lẫn nhiều boilerplate/spam.
Đọc STREAM trực tiếp từ .xz (lzma) — KHÔNG giải nén (tiết kiệm ~70GB đĩa).

  python scripts/filter_cc100.py --target 4000000
  # -> D:/Bit-Translate-data/raw/cc100_pick.txt
"""
import argparse
import hashlib
import lzma
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = Path("D:/Bit-Translate-data/raw")

JA = re.compile(r"[぀-ヿ一-鿿]")
KANJI = re.compile(r"[一-鿿]")
KATA_RUN = re.compile(r"[ァ-ヶー]{4,}")
ASCII_ABBR = re.compile(r"\b[A-Z]{2,6}\b")
KANJI_RUN = re.compile(r"[一-鿿]{3,}")
NUM_UNIT = re.compile(r"[0-9０-９]+\s*(%|％|人|円|年|月|日|回|倍|度|kg|km|cm|mm|ppm|μm|GB|MB)")
CLAUSE = re.compile(r"けれど|ものの|にもかかわらず|ながらも|とはいえ|ため、|ので、|から、|が、")
# Rác web điển hình: menu, nút bấm, thông tin liên hệ, spam
JUNK = re.compile(
    r"^(ホーム|トップ|お問い合わせ|会社概要|プライバシー|サイトマップ|ログイン|検索|"
    r"詳しくは|続きを読む|コメント|カテゴリ|タグ|次のページ|前へ|次へ)")
SPAMMY = re.compile(r"(http|www\.|＜|＞|\|\s*\||★{2,}|☆{2,}|【.{0,4}】.*【|→.*→.*→)")
REPEAT = re.compile(r"(.)\1{5,}")

# Từ khóa LĨNH VỰC — cần thiết vì heuristic hình thức không phân biệt được
# katakana thuật ngữ (フラーレン) với katakana tên thương mại (バイオハザード).
# Thử nghiệm cho thấy lọc theo hình thức đơn thuần chỉ ra tên game/luật/tin tức.
SCI_KW = re.compile(
    r"研究|実験|観測|測定|解析|分析|検証|仮説|理論|論文|学会|"
    r"細胞|遺伝子|染色体|タンパク質|酵素|免疫|神経|脳|症状|診断|治療|臨床|患者|投与|"
    r"分子|原子|化合物|反応|触媒|溶液|濃度|結晶|"
    r"物理|量子|素粒子|重力|電子|光子|波長|エネルギー|質量|温度|"
    r"数学|方程式|関数|確率|統計|有意|平均値|アルゴリズム|定理|"
    r"地質|地層|気候|大気|海洋|生態系|環境|排出|"
    r"技術|装置|機構|設計|製造|材料|強度|効率|システム|回路|半導体|"
    r"経済|市場|需要|供給|金融|投資|資本|指数|成長率|"
    r"社会|制度|政策|法律|条約|統治|選挙|"
    r"医学|薬|ワクチン|感染|疫学|保健|"
    r"人工知能|機械学習|データ|ネットワーク|プログラム|計算")


def clean(s):
    s = unicodedata.normalize("NFKC", s).strip()
    return re.sub(r"\s+", " ", s)


def score(s):
    """Điểm ưu tiên: càng cao càng đáng lấy. Trả None nếu loại thẳng."""
    n = len(s)
    if not (30 <= n <= 180):
        return None
    ja = len(JA.findall(s))
    if ja / n < 0.45:                      # web hay lẫn nhiều latin/số
        return None
    if JUNK.match(s) or SPAMMY.search(s) or REPEAT.search(s):
        return None
    if not re.search(r"[。？！?!]$", s):    # phải là câu trọn vẹn
        return None
    if s.count("、") > 6 or s.count("・") > 3:
        return None

    # ĐIỀU KIỆN CỨNG: phải mang dấu hiệu chuyên ngành. Không có thì loại thẳng,
    # dù câu dài và đúng ngữ pháp. Lý do: thử ngưỡng điểm đơn thuần thì lọt toàn
    # blog đời thường / fanfic (câu dài + kanji đậm là đủ 5đ) — không trị được
    # điểm yếu thật là THUẬT NGỮ.
    n_kata = len(KATA_RUN.findall(s))
    n_krun = len(KANJI_RUN.findall(s))
    has_abbr = bool(ASCII_ABBR.search(s))
    has_num = bool(NUM_UNIT.search(s))
    n_kw = len(set(SCI_KW.findall(s)))
    # BẮT BUỘC có từ khóa lĩnh vực. Không có thì loại, dù câu dài và nhiều katakana
    # (nếu không sẽ đầy tên game/thương hiệu — đã kiểm chứng bằng thử nghiệm).
    if n_kw == 0:
        return None
    if not (n_kata >= 1 or has_abbr or n_krun >= 2 or has_num):
        return None

    sc = 3.0 * min(3, n_kw)                 # từ khóa lĩnh vực: trọng số CAO NHẤT
    if 60 <= n <= 150:                      # ĐÚNG dải câu dài đang yếu
        sc += 3.0
    elif n > 150:
        sc += 1.0
    sc += 2.0 * min(3, n_kata)              # thuật ngữ ngoại lai (trọng số cao nhất)
    if has_abbr:                            # viết tắt khoa học/kỹ thuật
        sc += 2.5
    sc += 1.2 * min(3, n_krun)              # cụm kanji = thuật ngữ hán-việt
    if has_num:                             # số + đơn vị
        sc += 1.5
    if CLAUSE.search(s):                    # nhiều mệnh đề
        sc += 1.0
    if len(KANJI.findall(s)) / n > 0.30:    # đậm đặc kanji = văn viết/chuyên môn
        sc += 1.0
    return sc


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RAW / "ja.txt.xz"))
    ap.add_argument("--out", default=str(RAW / "cc100_pick.txt"))
    ap.add_argument("--target", type=int, default=4_000_000)
    ap.add_argument("--min-score", type=float, default=5.0,
                    help="ngưỡng điểm tối thiểu (5.0 ~ câu dài + có thuật ngữ)")
    ap.add_argument("--max-lines", type=int, default=0, help="0 = đọc hết file")
    a = ap.parse_args()

    print("nạp câu đã dùng để loại trùng...", flush=True)
    old = set()
    for p in [ROOT / "data" / "full_11.88m_ja_clean.txt", RAW / "os_ja_new.txt"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    old.add(dig(line.strip()))
    # loại luôn câu trong bộ bench (không được để lọt vào train)
    bp = ROOT / "eval" / "bench_new.jsonl"
    if bp.exists():
        import json
        for line in bp.open(encoding="utf-8"):
            old.add(dig(json.loads(line)["src"]))
    print(f"  {len(old):,} hash đã dùng", flush=True)

    seen = set()
    n = kept = dup = lowq = 0
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(a.src, "rt", encoding="utf-8", errors="ignore") as fin, \
            open(a.out, "w", encoding="utf-8") as fo:
        for line in fin:
            n += 1
            if a.max_lines and n > a.max_lines:
                break
            s = clean(line)
            if not s:
                continue
            sc = score(s)
            if sc is None or sc < a.min_score:
                lowq += 1
                continue
            h = dig(s)
            if h in old or h in seen:
                dup += 1
                continue
            seen.add(h)
            fo.write(s + "\n")
            kept += 1
            if kept >= a.target:
                break
            if n % 5_000_000 == 0:
                el = time.time() - t0
                print(f"  ...{n:,} đọc | giữ {kept:,} | {el/60:.1f}p "
                      f"({n/max(el,1e-9)/1000:.0f}k dòng/s)", flush=True)

    el = time.time() - t0
    print(f"\n=== LỌC XONG ({el/60:.1f} phút) ===")
    print(f"Đọc        : {n:,}")
    print(f"GIỮ        : {kept:,} ({100*kept/max(1,n):.2f}%)")
    print(f"Điểm thấp  : {lowq:,} | trùng: {dup:,}")
    print(f"Output     : {a.out}")


if __name__ == "__main__":
    main()
