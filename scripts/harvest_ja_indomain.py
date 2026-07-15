#!/usr/bin/env python3
"""Thu thập câu tiếng Nhật IN-DOMAIN IT (Qiita prose + GitHub issues/PR) làm nguồn
MONOLINGUAL cho back-translation (BT tự sinh vế VI → KHỎI cần align).

- Qiita: API public (60 req/h unauth). Lấy bài IT, tách câu prose (bỏ code/URL).
- GitHub: search issues/PR tiếng Nhật (cần GITHUB_TOKEN để có volume; unauth 60/h).

Xuất: data/synthetic/ja_indomain_bt.ja  (mỗi dòng 1 câu JA sạch, dedup)
Dùng: đưa file này lên GPU node -> model dịch ja->vi -> cặp (VI tổng hợp -> JA thật)
      -> train CHỈ chiều vi->ja (cứu chiều yếu + phủ hội thoại/prose IT thật).
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "synthetic"
JA = re.compile(r"[぀-ヿ㐀-鿿]")
CODEBLOCK = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`]*`")
MDLINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
HTML = re.compile(r"<[^>]+>")
MDMARK = re.compile(r"[#>*_~|]+")
SENT_SPLIT = re.compile(r"(?<=[。！？])")

QIITA_TAGS = ["Python", "JavaScript", "AWS", "Docker", "Java", "React",
              "Go", "Linux", "Git", "SQL", "kubernetes", "TypeScript"]
UA = {"User-Agent": "BitTranslate-harvest/1.0 (research)"}

seen, sents = set(), []


def clean_and_split(md):
    t = CODEBLOCK.sub(" ", md)
    t = INLINE.sub(" ", t)
    t = MDLINK.sub(r"\1", t)
    t = URL.sub(" ", t)
    t = HTML.sub(" ", t)
    t = MDMARK.sub(" ", t)
    out = []
    for line in t.split("\n"):
        for s in SENT_SPLIT.split(line):
            s = re.sub(r"\s+", " ", s).strip()
            if not s or not JA.search(s):
                continue
            if len(s) < 10 or len(s) > 160:
                continue
            # bỏ câu quá nhiều ký hiệu (còn sót code)
            if len(re.findall(r"[{}()=;<>/\\]", s)) > 4:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def harvest_qiita(pages_per_tag=1, per_page=100):
    for tag in QIITA_TAGS:
        try:
            for pg in range(1, pages_per_tag + 1):
                url = f"https://qiita.com/api/v2/tags/{tag}/items?page={pg}&per_page={per_page}"
                items = get(url)
                n0 = len(sents)
                for it in items:
                    sents.extend(clean_and_split(it.get("body", "")))
                print(f"[qiita:{tag} p{pg}] {len(items)} bài -> +{len(sents)-n0} câu", flush=True)
                time.sleep(1.2)  # lịch sự với API
        except Exception as e:
            print(f"[qiita:{tag}] lỗi: {str(e)[:70]}", flush=True)


def harvest_github(max_pages=3):
    tok = os.environ.get("GITHUB_TOKEN")
    hdr = dict(UA)
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    else:
        print("[github] không có GITHUB_TOKEN -> bỏ (unauth 60/h quá thấp). Set token để bật.", flush=True)
        return
    # issues/PR có nội dung tiếng Nhật, loại nhiễu
    q = "is:issue バグ OR エラー OR 修正 OR 実装 in:body"
    for pg in range(1, max_pages + 1):
        try:
            url = f"https://api.github.com/search/issues?q={urllib.parse.quote(q)}&per_page=100&page={pg}"
            data = get_hdr(url, hdr)
            n0 = len(sents)
            for it in data.get("items", []):
                sents.extend(clean_and_split(it.get("body") or ""))
            print(f"[github p{pg}] -> +{len(sents)-n0} câu", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"[github p{pg}] lỗi: {str(e)[:70]}", flush=True)
            break


def get_hdr(url, hdr):
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    import sys
    pages = int(os.environ.get("QIITA_PAGES", "1"))
    harvest_qiita(pages_per_tag=pages)
    if os.environ.get("GITHUB_TOKEN"):
        import urllib.parse
        harvest_github()
    (OUT / "ja_indomain_bt.ja").write_text("\n".join(sents) + "\n", encoding="utf-8")
    print(f"=== TỔNG: {len(sents)} câu JA in-domain -> data/synthetic/ja_indomain_bt.ja ===")
