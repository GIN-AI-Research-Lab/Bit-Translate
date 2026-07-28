#!/usr/bin/env python3
"""Lọc CC-100 lượt HAI — nhắm câu NGẮN, KHẨU NGỮ, THÀNH NGỮ (lỗ hổng còn lại).

Vì sao cần lượt hai: `filter_cc100.py` đặt điều kiện CỨNG `n_kw == 0 -> loại` (bắt
buộc có từ khóa khoa học/kỹ thuật). Nó giữ 5,23M câu dài chuyên ngành — đúng cho
điểm yếu câu dài — nhưng đồng thời VỨT toàn bộ tầng văn nói: 453 triệu dòng.

Mà bench chấm tay cho thấy trong 25 câu NGẮN v4 dịch sai, 17 câu là Google dịch
đúng (tức vá được), và đặc trưng dồn vào slang 6 / question 5 / idiom 4. Không
nguồn nào hiện có lấp được: OPUS phụ đề đã dùng cạn (934.025/934.524 đã KD),
Quốc hội là văn nói TRANG TRỌNG, CC-100 lượt một đã loại hết văn đời thường.

Điểm ngược hẳn lượt một:
  - dài 15-60 ký tự (lượt một: 30-180)
  - BẮT BUỘC có dấu hiệu khẩu ngữ (tiểu từ cuối câu, dạng rút gọn, thán từ)
  - vẫn loại rác web y như lượt một (menu, spam, lặp ký tự)
  - dừng sớm khi đủ --target (khẩu ngữ phổ biến hơn thuật ngữ nhiều nên không
    phải đọc hết 15,9GB)

  python scripts/filter_cc100_colloq.py --target 2000000
  # -> D:/Bit-Translate-data/raw/cc100_colloq.txt
"""
import argparse
import hashlib
import lzma
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = Path("D:/Bit-Translate-data/raw")

JA = re.compile(r"[぀-ヿ一-鿿]")
KANJI4 = re.compile(r"[一-鿿]{4}")          # 四字熟語 — dấu hiệu thành ngữ
# Tiểu từ cuối câu + dạng rút gọn: chữ ký rõ nhất của văn nói tiếng Nhật
COLLOQ = re.compile(
    r"(ね[。！？!?]|よ[。！？!?]|な[あぁ]?[。！？!?]|さ[。！？!?]|ぞ[。！？!?]|ぜ[。！？!?]|"
    r"わ[。！？!?]|かな[。？?]|っけ|じゃん|だろ[うっ]?|でしょ[。？?]|"
    r"ちゃう|ちゃっ|じゃう|じゃっ|とい[てた]|[てで]る[。、！？]|[てで]た[。、！？]|"
    r"なきゃ|なくちゃ|しなよ|してよ|みたい[。、]|っぽい|すぎる|"
    r"やっぱ|めっちゃ|すごく|ちょっと|とりあえず|なんか|マジ|ヤバ)")
# Thán từ / mở đầu hội thoại
INTERJ = re.compile(r"^(え[えっ]?|あ[あっ]?|お[おっ]?|うん|はい|いや|まあ|ねえ|"
                    r"そう|でも|だから|ただ|実は|正直)[、。！？ 　]")
IDIOM_HINT = re.compile(
    r"(手を[打貸借出焼]|足を[引運洗]|目を[通見疑]|耳を[傾貸]|口を[出挟揃]|"
    r"腹を[割立決]|頭を[抱下悩]|気が[付済重引]|気を[付使配遣]|"
    r"猫の手|馬の耳|石の上|棚から|藪から|渡りに|背に腹|二の足|一石二鳥|"
    r"朝飯前|朝令暮改|以心伝心|臨機応変|一期一会|自業自得|十人十色)")

JUNK = re.compile(
    r"^(ホーム|トップ|お問い合わせ|会社概要|プライバシー|サイトマップ|ログイン|検索|"
    r"詳しくは|続きを読む|コメント|カテゴリ|タグ|次のページ|前へ|次へ|返信|引用|"
    r"投稿日|更新日|スポンサーリンク|関連記事)")
SPAMMY = re.compile(r"(http|www\.|＜|＞|\|\s*\||★{2,}|☆{2,}|→.*→|\d{3}-\d{4})")
REPEAT = re.compile(r"(.)\1{4,}")
# Câu chỉ có tên/nhãn, không có động từ-tính từ kết câu
NOVERB = re.compile(r"[。！？!?]$")


def clean(s):
    s = unicodedata.normalize("NFKC", s).strip()
    return re.sub(r"\s+", " ", s)


def score(s):
    n = len(s)
    if not (15 <= n <= 60):
        return None
    ja = len(JA.findall(s))
    if ja / n < 0.55:                       # câu ngắn thì phải đậm chữ Nhật
        return None
    if JUNK.match(s) or SPAMMY.search(s) or REPEAT.search(s):
        return None
    if not NOVERB.search(s):
        return None
    if s.count("、") > 2 or s.count("・") > 1:
        return None

    n_col = len(set(COLLOQ.findall(s)))
    has_int = bool(INTERJ.match(s))
    has_idi = bool(IDIOM_HINT.search(s))
    n_k4 = len(KANJI4.findall(s))
    # ĐIỀU KIỆN CỨNG: phải có dấu hiệu khẩu ngữ HOẶC thành ngữ. Không có thì loại,
    # nếu không sẽ đầy câu tin tức ngắn — thứ corpus đã thừa.
    if not (n_col or has_int or has_idi):
        return None

    sc = 2.0 * min(3, n_col)
    if has_idi:
        sc += 5.0                            # thành ngữ hiếm nhất -> trọng số cao nhất
    if has_int:
        sc += 2.0
    if n_k4:
        sc += 1.5 * min(2, n_k4)
    if 20 <= n <= 45:                        # dải câu thoại tự nhiên
        sc += 1.5
    if re.search(r"[「」『』]", s):           # có thoại trực tiếp
        sc += 1.0
    return sc


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(RAW / "ja.txt.xz"))
    ap.add_argument("--out", default=str(RAW / "cc100_colloq.txt"))
    ap.add_argument("--target", type=int, default=2_000_000)
    ap.add_argument("--min-score", type=float, default=6.0)
    ap.add_argument("--max-lines", type=int, default=0)
    a = ap.parse_args()

    print("nạp hash đã dùng để loại trùng...", flush=True)
    old = set()
    for p in [ROOT / "data" / "full_11.88m_ja_clean.txt", RAW / "os_ja_new.txt",
              RAW / "cc100_pick.txt", RAW / "kokkai_ja.txt"]:
        if p.exists():
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    old.add(dig(line.strip()))
    print(f"  {len(old):,} hash", flush=True)

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
                print(f"  ...{n:,} đọc | giữ {kept:,} ({100*kept/n:.2f}%) | "
                      f"{el/60:.1f}p", flush=True)

    el = time.time() - t0
    print(f"\n=== XONG ({el/60:.1f} phút) ===")
    print(f"Đọc  : {n:,}\nGIỮ  : {kept:,} ({100*kept/max(1,n):.2f}%)")
    print(f"Điểm thấp: {lowq:,} | trùng: {dup:,}\nOutput: {a.out}")


if __name__ == "__main__":
    main()
