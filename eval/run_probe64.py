#!/usr/bin/env python3
"""Chạy probe64 + glossary test-set trên 1 model i2_s qua llama-server.

  python run_probe64.py <gguf> <label> [--terms N]

- probe64: dịch 64 câu, chấm chrF từng câu so ref, gộp theo domain x chiều.
  (chrF từng câu là proxy khách quan; "ok" mắt người vẫn là chuẩn cuối.)
- glossary: lấy N term (mặc định 200, seed cố định) từ glossary_test.csv,
  dịch JA->VI, đếm % bản dịch chứa đúng nghĩa vi (so khớp không dấu hoa thường).
Kết quả in ra stdout + ghi eval/probe64_<label>.jsonl (từng câu, để đọc bằng mắt).
"""
import csv
import json
import os
import random
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import sacrebleu

SERVER = os.environ.get("LLAMA_SERVER", "/home/tuent/BitNet-test/build/bin/llama-server")

UP = "http://127.0.0.1:8080"
BOS, EOS, VIE, JPN = 2, 3, 4, 5
ROOT = Path(__file__).parent.parent
GGUF, LABEL = sys.argv[1], sys.argv[2]
N_TERMS = int(sys.argv[sys.argv.index("--terms") + 1]) if "--terms" in sys.argv else 200


def up(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wait_health(t=90):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(UP + "/health", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def translate(text, tag):
    ids = [BOS, tag] + up("/tokenize", {"content": text})["tokens"] + [EOS]
    out = up("/completion", {"prompt": ids, "n_predict": 140, "temperature": 0.0,
                             "cache_prompt": False})
    return (out.get("content") or "").strip()


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def main():
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(1.5)
    subprocess.Popen([SERVER, "-m", GGUF,
                      "--host", "127.0.0.1", "--port", "8080", "-t", "6", "-c", "256",
                      "--parallel", "1"],
                     stdout=open("/tmp/probe_server.log", "w"), stderr=subprocess.STDOUT)
    assert wait_health(), "server not up"

    # ---- probe64 ----
    probes = [json.loads(l) for l in (ROOT / "eval" / "probe64.jsonl").open(encoding="utf-8")]
    rows = []
    for p in probes:
        tag = JPN if p["dir"] == "vi2ja" else VIE
        hyp = translate(p["src"], tag)
        score = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
        rows.append({**p, "hyp": hyp, "chrf": round(score, 1)})
    outp = ROOT / "eval" / f"probe64_{LABEL}.jsonl"
    outp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")

    agg = {}
    for r in rows:
        agg.setdefault((r["domain"], r["dir"]), []).append(r["chrf"])
    print(f"=== probe64 [{LABEL}] — chrF trung bình theo domain x chiều ===", flush=True)
    for (dom, d), v in sorted(agg.items()):
        print(f"  {dom:10} {d:6} {sum(v)/len(v):5.1f}  (n={len(v)})", flush=True)
    for d in ("vi2ja", "ja2vi"):
        v = [r["chrf"] for r in rows if r["dir"] == d]
        print(f"  TB {d}: {sum(v)/len(v):.1f}", flush=True)

    # ---- glossary test-set ----
    terms = list(csv.DictReader((ROOT / "data" / "glossary" / "glossary_test.csv")
                                .open(encoding="utf-8-sig")))
    random.Random(42).shuffle(terms)
    terms = terms[:N_TERMS]
    hit = 0
    misses = []
    for t in terms:
        hyp = translate(t["ja"], VIE)
        if norm(t["vi"]) in norm(hyp):
            hit += 1
        elif len(misses) < 15:
            misses.append((t["ja"], t["vi"], hyp))
    print(f"=== glossary test [{LABEL}]: {hit}/{len(terms)} = {100*hit/len(terms):.0f}% "
          f"(JA term -> VI chứa đúng nghĩa) ===", flush=True)
    for ja, vi, hyp in misses[:8]:
        print(f"  MISS {ja!r} (đáp: {vi!r}) -> {hyp!r}", flush=True)
    print(f"done [{LABEL}] -> {outp}", flush=True)


if __name__ == "__main__":
    main()
