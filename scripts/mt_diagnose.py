#!/usr/bin/env python3
"""Dịch KHỐI LỚN bằng bitnet.cpp rồi DÒ LỖI TỰ ĐỘNG — không cần judge.

Vì sao: chấm tay trần khoảng 200-400 câu/phiên, quá ít để vẽ đường cong tỷ lệ lỗi
theo độ dài. Mà bốn lớp lỗi user quan tâm (sai / thiếu / nhiễu / bịa) đều bắt được
bằng luật, nên chạy được trên 100k+ câu và cho sai số ±0,3%.

Bảy bộ dò, mỗi cái nhắm một lớp lỗi đã QUAN SÁT ĐƯỢC trên bench thật:
  sot_tieng_nhat  còn ký tự Nhật trong câu Việt          (chưa dịch)
  lap_vong        n-gram lặp >= 3 lần                    (nhiễu decoding)
  roi_noi_dung    tỷ lệ dài VI/JA thấp bất thường        (thiếu — lớp lỗi chủ đạo câu dài)
  bia_them        tỷ lệ đó cao bất thường                (bịa)
  lech_so         số trong JA không có trong VI          (lỗi [140] "1/10" -> "không có ai")
  roi_katakana    katakana dài trong JA, VI không có từ ngoại lai tương ứng
  cut_cau         không có dấu kết câu

QUAN TRỌNG — cách gọi bitnet.cpp: định dạng train là
    [BOS] >>vie<< <ja> [EOS] <vi> [EOS]
EOS là DẤU NGĂN CÁCH. `llama-cli -p ">>vie<< {src}"` thiếu EOS -> model ra rác
(v4_avg5 ra toàn dấu chấm). Phải truyền TOKEN ID qua llama-server. Bug này từng
làm sai mọi số đo GGUF cũ trong HANDOFF (§7 "292M GGUF 41,8 chrF").

  # 1) mở server trong WSL:
  ~/BitNet-test/build/bin/llama-server -m /mnt/d/.../v4_avg5_i2s.gguf \
      -c 512 -t 6 -np 4 --host 127.0.0.1 --port 8081
  # 2) chạy:
  python scripts/mt_diagnose.py --src D:/Bit-Translate-data/raw/kokkai_ja.txt -n 100000
"""
import argparse
import json
import random
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent
SP = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
BOS, EOS = SP.bos_id(), SP.eos_id()
VIE = SP.piece_to_id(">>vie<<")

JA_CH = re.compile(r"[぀-ヿ一-鿿]")
KATA_RUN = re.compile(r"[ァ-ヶー]{4,}")
NUM = re.compile(r"[0-9]+")
LATIN = re.compile(r"[A-Za-zÀ-ỹ]{3,}")
END = re.compile(r"[.!?…。]\s*$")


def norm_ja(s):
    return unicodedata.normalize("NFKC", s).strip()


def translate(src, url, n_predict=200):
    ids = [BOS, VIE] + SP.encode(norm_ja(src)) + [EOS]
    body = json.dumps({"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["content"].strip()


def ngram_repeat(s, n=4, k=3):
    w = s.split()
    if len(w) < n * k:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return c.most_common(1)[0][1] >= k if c else False


def detect(ja, vi):
    """Trả về danh sách cờ lỗi. Ngưỡng tỷ lệ dài lấy từ phân vị của chính mẻ dịch
    (xem calibrate) chứ không bịa số cứng."""
    f = []
    if not vi:
        return ["rong"]
    if JA_CH.search(vi):
        f.append("sot_tieng_nhat")
    if ngram_repeat(vi):
        f.append("lap_vong")
    if not END.search(vi):
        f.append("cut_cau")
    nj = set(NUM.findall(ja))
    if nj and not nj <= set(NUM.findall(vi)):
        f.append("lech_so")
    if KATA_RUN.search(ja) and not LATIN.search(vi):
        f.append("roi_katakana")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("-n", type=int, default=100000)
    ap.add_argument("--url", default="http://127.0.0.1:8081/completion")
    ap.add_argument("--workers", type=int, default=4, help="phải <= -np của llama-server")
    ap.add_argument("--out", default="D:/Bit-Translate-data/diag_v4a5.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    lines = []
    with open(a.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    print(f"nguồn: {len(lines):,} câu", flush=True)
    # LẤY MẪU PHÂN TẦNG theo độ dài — nếu lấy ngẫu nhiên thì dải dài (hiếm) không đủ
    # câu để có sai số hẹp, mà chính dải đó mới là chỗ cần đo.
    bands = [(0, 40), (40, 60), (60, 80), (80, 110), (110, 150), (150, 10**9)]
    by = defaultdict(list)
    for s in lines:
        for lo, hi in bands:
            if lo <= len(s) < hi:
                by[(lo, hi)].append(s)
                break
    rng = random.Random(a.seed)
    per = a.n // len(bands)
    pick = []
    for b in bands:
        v = by[b]
        rng.shuffle(v)
        pick += v[:per]
        print(f"  dải {b[0]:>3}-{b[1] if b[1]<10**9 else '∞':<4} có {len(v):>8,} -> lấy {min(per,len(v)):,}")
    rng.shuffle(pick)
    print(f"tổng lấy: {len(pick):,}\n", flush=True)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():
        for line in outp.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["ja"])
            except Exception:  # noqa: BLE001
                pass
        print(f"[resume] đã dịch {len(done):,}", flush=True)
    todo = [s for s in pick if s not in done]

    fo = outp.open("a", encoding="utf-8")
    t0 = time.time()
    n = 0

    def work(s):
        try:
            return s, translate(s, a.url)
        except Exception as e:  # noqa: BLE001
            return s, f"__LOI__ {type(e).__name__}"

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for ja, vi in ex.map(work, todo):
            n += 1
            fo.write(json.dumps({"ja": ja, "vi": vi, "flags": detect(ja, vi),
                                 "lj": len(ja), "lv": len(vi)}, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                fo.flush()
                el = time.time() - t0
                print(f"  {n:,}/{len(todo):,} | {n/el:.1f} câu/s | "
                      f"còn {(len(todo)-n)/max(n/el,1e-9)/60:.0f} phút", flush=True)
    fo.close()
    print(f"\nXONG {n:,} câu trong {(time.time()-t0)/60:.0f} phút -> {a.out}")


if __name__ == "__main__":
    main()
