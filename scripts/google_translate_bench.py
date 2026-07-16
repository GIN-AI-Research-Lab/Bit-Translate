#!/usr/bin/env python3
"""Dịch bộ đề hardbench bằng Google Translate (endpoint web gtx, không cần key).

  python scripts/google_translate_bench.py [--probe eval/hardbench200.jsonl] [--label google]

Ghi eval/hardbench_<label>.jsonl cùng format với run_hardbench.py (hyp + chrF).
RESUME được: nếu file output đã có id nào thì bỏ qua id đó.
Lưu ý: gtx là endpoint không chính thức — đi chậm (0.8s/req) để khỏi bị chặn.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).parent.parent
PROBE = Path(sys.argv[sys.argv.index("--probe") + 1]) if "--probe" in sys.argv \
    else ROOT / "eval" / "hardbench200.jsonl"
LABEL = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "google"
OUT = ROOT / "eval" / f"hardbench_{LABEL}.jsonl"


def gtranslate(text, sl, tl, retries=4):
    q = urllib.parse.quote(text)
    url = (f"https://translate.googleapis.com/translate_a/single"
           f"?client=gtx&sl={sl}&tl={tl}&dt=t&q={q}")
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = json.loads(r.read())
            return "".join(seg[0] for seg in data[0] if seg and seg[0])
        except Exception as e:
            if a == retries - 1:
                raise
            time.sleep(3.0 * (a + 1))


def main():
    probes = [json.loads(l) for l in PROBE.open(encoding="utf-8")]
    done = set()
    rows = []
    if OUT.exists():
        rows = [json.loads(l) for l in OUT.open(encoding="utf-8")]
        done = {r["id"] for r in rows}
    for i, p in enumerate(probes):
        if p["id"] in done:
            continue
        sl, tl = ("vi", "ja") if p["dir"] == "vi2ja" else ("ja", "vi")
        hyp = gtranslate(p["src"], sl, tl).strip()
        score = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
        rows.append({**p, "hyp": hyp, "chrf": round(score, 1)})
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(probes)}", flush=True)
        time.sleep(0.8)

    agg = {}
    for r in rows:
        agg.setdefault((r["domain"], r["dir"]), []).append(r["chrf"])
    print(f"=== hardbench [{LABEL}] — chrF trung bình theo domain x chiều ===", flush=True)
    for (dom, d), v in sorted(agg.items()):
        print(f"  {dom:12} {d:6} {sum(v)/len(v):5.1f}  (n={len(v)})", flush=True)
    for d in ("vi2ja", "ja2vi"):
        v = [r["chrf"] for r in rows if r["dir"] == d]
        print(f"  TB {d}: {sum(v)/len(v):.1f}", flush=True)
    print(f"done [{LABEL}] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
