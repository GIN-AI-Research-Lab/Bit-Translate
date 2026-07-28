#!/usr/bin/env python3
"""Thu thập câu tiếng Nhật từ BIÊN BẢN QUỐC HỘI (kokkai.ndl.go.jp) — nguồn câu DÀI.

Vì sao nguồn này: đo thực tế trên mẫu 400 phát biểu cho thấy **0 câu trùng** trong
17,4 triệu hash corpus cũ (OPUS + CC-100 + OpenSubtitles), và phân bố độ dài đảo
ngược hẳn so với corpus hiện có:

    corpus v4 : 69,7% chuỗi < 40 token, chỉ 3,4% >= 110 token
    Quốc hội  : p50 = 62 ký tự, 31% câu >= 80 ký tự

Đúng chỗ model đang gãy: bench câu thật cho v4 câu ngắn 75% / câu dài 52%, và
`grammar_probe.py` cho thấy cùng điểm ngữ pháp thì ngắn 96% / không-ngắn 73%.

CÁI PHẢI LỌC: biên bản đầy văn khuôn mẫu nghị trường ("賛成の諸君の起立を求めます")
— dài nhưng lặp và không đại diện tiếng Nhật thường. Lọc bằng PROC bên dưới.

Giấy phép: 会議録 là tài liệu công của Quốc hội Nhật; NDL cung cấp API mở để tái sử
dụng. Không phải tác phẩm có bản quyền theo 著作権法 13条.

  python scripts/harvest_kokkai.py --from 2015 --to 2025
  # -> D:/Bit-Translate-data/raw/kokkai_ja.txt  (mỗi dòng 1 câu, đã dedup + lọc)
"""
import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
API = "https://kokkai.ndl.go.jp/api/speech"
OUT = Path("D:/Bit-Translate-data/raw/kokkai_ja.txt")

# Đầu phát biểu: "○国務大臣（林芳正君）" — bỏ đi, không phải nội dung.
SPK = re.compile(r"^[○●][^（(]{0,24}[（(][^）)]{0,40}[）)]\s*")
# Chú thích sân khấu: 〔賛成者起立〕, （拍手）
BRACKET = re.compile(r"[〔\[][^〕\]]{0,60}[〕\]]|[（(](?:拍手|笑声|発言する者あり)[）)]")

# VĂN KHUÔN MẪU NGHỊ TRƯỜNG — dài nhưng vô giá trị để học dịch. Đây là phần chiếm
# tỷ lệ lớn nhất trong biên bản; không lọc thì model học giọng chủ tọa.
PROC = re.compile(
    r"起立|異議(なし|ありません)|賛成の諸君|閉会中審査|これより|本日は これにて|"
    r"散会|休憩|着席|御着席|発言を許します|質疑を許します|討論を許します|"
    r"ただいま議題となりました|採決|可決すべき|否決|記名投票|投票を行います|"
    r"委員長の報告|報告のとおり|次回は|公報をもって|announce|"
    r"速記を(止め|起こ)|理事会において協議|申出のとおり|会議を開きます")
# Câu chỉ là chào hỏi/thủ tục ngắn
GREET = re.compile(r"^(はい|そうです|わかりました|ありがとうございました?|"
                   r"よろしくお願い(いた)?します|以上です|失礼(いた)?しました)[。、]?$")
JA = re.compile(r"[぀-ヿ一-鿿]")
KANA = re.compile(r"[ぁ-ん]")
REPEAT = re.compile(r"(.)\1{5,}")


def clean(s):
    s = unicodedata.normalize("NFKC", s)
    s = SPK.sub("", s)
    s = BRACKET.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def dig(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def fetch(params, tries=4):
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                print(f"  [bỏ] {type(e).__name__} {str(e)[:70]}", flush=True)
                return None
            time.sleep(2 * (k + 1))
    return None


def months(y0, y1):
    # Dùng calendar chứ KHÔNG đoán ngày cuối tháng: "2010-02-29" không tồn tại,
    # API trả HTTP 400 và mất im lặng cả tháng đó (đã vấp lần đầu).
    import calendar
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            last = calendar.monthrange(y, m)[1]
            yield f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="y0", type=int, default=2015)
    ap.add_argument("--to", dest="y1", type=int, default=2025)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=220)
    ap.add_argument("--sleep", type=float, default=0.3, help="nghỉ giữa request (lịch sự)")
    a = ap.parse_args()

    print("nạp hash corpus cũ để loại trùng...", flush=True)
    seen = set()
    for p in [ROOT / "data" / "full_11.88m_ja_clean.txt",
              Path("D:/Bit-Translate-data/raw/os_ja_new.txt"),
              Path("D:/Bit-Translate-data/raw/cc100_pick.txt")]:
        if p.exists():
            with open(p, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    seen.add(dig(line.strip()))
    print(f"  {len(seen):,} hash\n", flush=True)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # resume: nạp lại phần đã thu
    if outp.exists():
        with outp.open(encoding="utf-8") as f:
            n0 = 0
            for line in f:
                seen.add(dig(line.strip()))
                n0 += 1
        print(f"[resume] đã có {n0:,} câu\n", flush=True)
    else:
        n0 = 0

    fo = outp.open("a", encoding="utf-8")
    kept = n0
    nspeech = ndrop_proc = ndrop_dup = ndrop_len = 0
    t0 = time.time()

    for m0, m1 in months(a.y0, a.y1):
        start = 1
        while True:
            d = fetch({"recordPacking": "json", "maximumRecords": 100,
                       "startRecord": start, "from": m0, "until": m1})
            if not d:
                break
            recs = d.get("speechRecord", [])
            if not recs:
                break
            for s in recs:
                nspeech += 1
                for x in re.split(r"(?<=[。？！])", clean(s.get("speech", ""))):
                    x = x.strip()
                    if not (a.min_len <= len(x) <= a.max_len):
                        ndrop_len += 1
                        continue
                    if not KANA.search(x) or len(JA.findall(x)) / len(x) < 0.4:
                        ndrop_len += 1
                        continue
                    if PROC.search(x) or GREET.match(x) or REPEAT.search(x):
                        ndrop_proc += 1
                        continue
                    h = dig(x)
                    if h in seen:
                        ndrop_dup += 1
                        continue
                    seen.add(h)
                    fo.write(x + "\n")
                    kept += 1
            nxt = d.get("nextRecordPosition")
            if not nxt:
                break
            start = nxt
            time.sleep(a.sleep)
        fo.flush()
        el = (time.time() - t0) / 60
        print(f"  {m0[:7]} | phát biểu {nspeech:,} | GIỮ {kept:,} câu | "
              f"loại: khuôn mẫu {ndrop_proc:,} trùng {ndrop_dup:,} độ dài {ndrop_len:,} "
              f"| {el:.0f}p", flush=True)

    fo.close()
    print(f"\n=== XONG ({(time.time()-t0)/60:.0f} phút) ===")
    print(f"Phát biểu đọc : {nspeech:,}")
    print(f"CÂU GIỮ       : {kept:,}")
    print(f"Loại khuôn mẫu: {ndrop_proc:,} | trùng: {ndrop_dup:,} | độ dài/không-JA: {ndrop_len:,}")
    print(f"Output        : {a.out}")


if __name__ == "__main__":
    main()
