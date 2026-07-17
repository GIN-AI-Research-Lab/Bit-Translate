#!/usr/bin/env python3
"""Chạy bộ đề hardbench trên 1 model i2_s qua llama-server (như run_probe64).

  LLAMA_SERVER=~/BitNet/build/bin/llama-server \
  python eval/run_hardbench.py <gguf> <label> [--probe eval/hardbench200.jsonl]

Ghi eval/hardbench_<label>.jsonl (từng câu: hyp + chrF so ref) + in tổng hợp
theo domain x chiều.
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
GGUF, LABEL = sys.argv[1], sys.argv[2]
PROBE = Path(sys.argv[sys.argv.index("--probe") + 1]) if "--probe" in sys.argv \
    else ROOT / "eval" / "hardbench200.jsonl"


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
    text = normalize_for_model(text)
    ids = [BOS, tag] + up("/tokenize", {"content": text})["tokens"] + [EOS]
    out = up("/completion", {"prompt": ids, "n_predict": 160, "temperature": 0.0,
                             "cache_prompt": False})
    return (out.get("content") or "").strip()


def main():
    # -x (khớp đúng tên process): pkill -f tự khớp cmdline cha chứa LLAMA_SERVER=.../llama-server
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)
    time.sleep(1.5)
    subprocess.Popen([SERVER, "-m", GGUF,
                      "--host", "127.0.0.1", "--port", "8080", "-t", "6", "-c", "320",
                      "--parallel", "1"],
                     stdout=open("/tmp/hardbench_server.log", "w"), stderr=subprocess.STDOUT)
    assert wait_health(), "server not up"

    probes = [json.loads(l) for l in PROBE.open(encoding="utf-8")]
    rows = []
    t0 = time.time()
    for i, p in enumerate(probes):
        tag = JPN if p["dir"] == "vi2ja" else VIE
        hyp = translate(p["src"], tag)
        score = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
        rows.append({**p, "hyp": hyp, "chrf": round(score, 1)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(probes)} ({time.time()-t0:.0f}s)", flush=True)
    subprocess.run(["pkill", "-x", "llama-server"], capture_output=True)

    outp = ROOT / "eval" / f"hardbench_{LABEL}.jsonl"
    outp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")

    agg = {}
    for r in rows:
        agg.setdefault((r["domain"], r["dir"]), []).append(r["chrf"])
    print(f"=== hardbench [{LABEL}] — chrF trung bình theo domain x chiều ===", flush=True)
    for (dom, d), v in sorted(agg.items()):
        print(f"  {dom:12} {d:6} {sum(v)/len(v):5.1f}  (n={len(v)})", flush=True)
    for d in ("vi2ja", "ja2vi"):
        v = [r["chrf"] for r in rows if r["dir"] == d]
        print(f"  TB {d}: {sum(v)/len(v):.1f}", flush=True)
    print(f"done [{LABEL}] -> {outp}", flush=True)


if __name__ == "__main__":
    main()
