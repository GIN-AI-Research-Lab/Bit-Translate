#!/usr/bin/env python3
"""Chạy hardbench ja2vi qua llama-server (i2_s GGUF chạy trong WSL), chấm chrF từ
Windows (có sacrebleu). So Google/Haiku. Server phải đang chạy ở 127.0.0.1:8080.

  python scripts/hardbench_server.py [label]
"""
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).parent))
from text_norm import normalize_for_model

ROOT = Path(__file__).parent.parent
UP = "http://127.0.0.1:8080"
BOS, EOS, VIE = 2, 3, 4
LABEL = sys.argv[1] if len(sys.argv) > 1 else "kd100m_i2s"


def up(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def translate(text):
    text = normalize_for_model(text)
    ids = [BOS, VIE] + up("/tokenize", {"content": text})["tokens"] + [EOS]
    out = up("/completion", {"prompt": ids, "n_predict": 160, "temperature": 0.0,
                             "cache_prompt": False})
    return (out.get("content") or "").strip()


def dom_agg(rows):
    a = defaultdict(list)
    for r in rows:
        a[r["domain"]].append(r["chrf"])
    return {k: sum(v) / len(v) for k, v in a.items()}


def main():
    probes = [json.loads(l) for l in (ROOT / "eval" / "hardbench_ja2vi.jsonl").open(encoding="utf-8")]
    rows = []
    for i, p in enumerate(probes):
        hyp = translate(p["src"])
        chrf = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
        rows.append({**p, "hyp": hyp, "chrf": round(chrf, 1)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100", flush=True)
    (ROOT / "eval" / f"hardbench_{LABEL}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    ours = dom_agg(rows)
    goog = dom_agg([json.loads(l) for l in (ROOT / "eval" / "hardbench_google.jsonl").open(encoding="utf-8")
                    if json.loads(l)["dir"] == "ja2vi"])
    haik = dom_agg([json.loads(l) for l in (ROOT / "eval" / "hardbench_haiku.jsonl").open(encoding="utf-8")
                    if json.loads(l)["dir"] == "ja2vi"])
    o = sum(r["chrf"] for r in rows) / len(rows)
    print(f"\n=== hardbench ja2vi chrF [{LABEL} — i2_s GGUF CPU] ===")
    print(f"{'domain':12} {'i2s':>7} {'Google':>7} {'Haiku':>7}")
    for d in sorted(ours):
        print(f"{d:12} {ours[d]:7.1f} {goog[d]:7.1f} {haik[d]:7.1f}")
    import statistics
    g = statistics.mean(json.loads(l)["chrf"] for l in (ROOT / "eval" / "hardbench_google.jsonl").open(encoding="utf-8") if json.loads(l)["dir"] == "ja2vi")
    h = statistics.mean(json.loads(l)["chrf"] for l in (ROOT / "eval" / "hardbench_haiku.jsonl").open(encoding="utf-8") if json.loads(l)["dir"] == "ja2vi")
    print(f"{'TỔNG TB':12} {o:7.1f} {g:7.1f} {h:7.1f}")
    print(f"(PyTorch step11000 chrF ja2vi = 43.3 — so xem i2_s có giữ chất lượng)")


if __name__ == "__main__":
    main()
