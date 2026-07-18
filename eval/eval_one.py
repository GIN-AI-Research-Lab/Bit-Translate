#!/usr/bin/env python3
"""chrF FLORES một model x một chiều (chạy ngắn, hợp foreground).
  python eval_one.py <gguf> <label> <vi2ja|ja2vi> [n]
Server tự bật/tắt. In 1 dòng kết quả + append vào eval/chrf_runs.tsv.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from text_norm import normalize_for_model  # noqa: E402  (NFKC — llama.cpp khong tu chuan hoa)

SERVER = os.environ.get("LLAMA_SERVER", "/home/tuent/BitNet-test/build/bin/llama-server")

UP = "http://127.0.0.1:8080"
BOS, EOS, VIE, JPN = 2, 3, 4, 5
ROOT = Path(__file__).parent.parent
FLORES = ROOT / "data" / "clean" / "flores"
GGUF, LABEL, DIR = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100


def up(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    # -x: chỉ khớp tên process — pkill -f giết cả shell cha có LLAMA_SERVER=... trong cmdline
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    time.sleep(1.5)
    subprocess.Popen([SERVER, "-m", GGUF,
                      "--host", "127.0.0.1", "--port", "8080", "-t", "6", "-c", "256",
                      "--parallel", "1"],
                     stdout=open("/tmp/eval_server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(UP + "/health", timeout=2)
            break
        except Exception:
            time.sleep(0.5)

    def read(p):
        return [l.strip() for l in open(p, encoding="utf-8").read().splitlines() if l.strip()][:N]

    vi, ja = read(FLORES / "flores.devtest.vi"), read(FLORES / "flores.devtest.ja")
    src, ref, tag = (vi, ja, JPN) if DIR == "vi2ja" else (ja, vi, VIE)
    t0 = time.time()
    hyps = []
    for s in src:
        s = normalize_for_model(s)
        ids = [BOS, tag] + up("/tokenize", {"content": s})["tokens"] + [EOS]
        out = up("/completion", {"prompt": ids, "n_predict": 120, "temperature": 0.0,
                                 "cache_prompt": False})
        hyps.append((out.get("content") or "").strip())
    chrf = sacrebleu.corpus_chrf(hyps, [ref]).score
    # -x: chỉ khớp tên process — pkill -f giết cả shell cha có LLAMA_SERVER=... trong cmdline
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    line = f"{LABEL}\t{DIR}\t{chrf:.2f}\t{len(src)}\t{time.time()-t0:.0f}s"
    print(f"RESULT\t{line}", flush=True)
    with (ROOT / "eval" / "chrf_runs.tsv").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
