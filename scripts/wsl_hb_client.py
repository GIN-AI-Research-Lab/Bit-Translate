#!/usr/bin/env python3
"""Client thuan stdlib (chay trong WSL) — query llama-server 127.0.0.1:8080 dich
100 cau ja2vi hardbench, ghi hyps ra file. Windows cham chrF sau (co sacrebleu).
"""
import json
import sys
import unicodedata
import urllib.request

UP = "http://127.0.0.1:8080"
BOS, EOS, VIE = 2, 3, 4
INP = sys.argv[1]
OUT = sys.argv[2]


def up(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    probes = [json.loads(l) for l in open(INP, encoding="utf-8")]
    out = open(OUT, "w", encoding="utf-8")
    for i, p in enumerate(probes):
        text = unicodedata.normalize("NFKC", p["src"])
        text = " ".join(text.split())
        ids = [BOS, VIE] + up("/tokenize", {"content": text})["tokens"] + [EOS]
        r = up("/completion", {"prompt": ids, "n_predict": 160, "temperature": 0.0,
                               "cache_prompt": False})
        hyp = (r.get("content") or "").strip()
        out.write(json.dumps({**p, "hyp": hyp}, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(probes)}", flush=True)
    out.close()
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
