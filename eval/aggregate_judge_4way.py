#!/usr/bin/env python3
"""Tong hop diem trong tai 4 he tu cac file judge_p*_j*.jsonl + panel_key.json.

  python eval/aggregate_judge_4way.py <judges_dir> <panels_dir>

In bang tong (acc/nat theo chieu), theo domain, % dung duoc (acc>=4),
doi dau 110M vs Google va Haiku vs Google (theo diem acc trung binh/cau).
Ghi eval/hardbench_4way_scores.json (diem trung binh tung cau tung he).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
JD, PD = Path(sys.argv[1]), Path(sys.argv[2])

key = {int(k): v for k, v in json.loads((PD / "panel_key.json").read_text(encoding="utf-8")).items()}
base = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / "hardbench200.jsonl", encoding="utf-8"))}

# scores[id][system] = list of (acc, nat) tu cac giam khao
scores = defaultdict(lambda: defaultdict(list))
n_files = 0
for f in sorted(JD.glob("judge_p*_j*.jsonl")):
    n_files += 1
    for l in open(f, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        i = int(r["id"])
        for L in "ABCD":
            if L in r and L in key[i]:
                s = r[L]
                scores[i][key[i][L]].append((s["acc"], s["nat"]))
print(f"{n_files} file giam khao, {len(scores)} cau co diem")

# hệ lấy từ panel_key (hỗ trợ thay 110M bằng 292M...), giữ thứ tự ưu tiên hiển thị
_present = set(next(iter(key.values())).values())
SYS = [s for s in ["110M", "292M", "google", "fable", "haiku"] if s in _present] \
    + sorted(_present - {"110M", "292M", "google", "fable", "haiku"})
mean = lambda v: sum(v) / len(v) if v else float("nan")

# diem trung binh tung cau
per_item = {}
for i in scores:
    per_item[i] = {}
    for s in SYS:
        v = scores[i][s]
        per_item[i][s] = {"acc": mean([x[0] for x in v]), "nat": mean([x[1] for x in v]),
                          "n_judges": len(v)}

out = {"per_item": per_item}
Path(ROOT / "hardbench_4way_scores.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

print("\n=== TONG (n=%d cau, TB diem giam khao) ===" % len(per_item))
print(f"{'he':8} {'acc vi2ja':>9} {'acc ja2vi':>9} {'nat vi2ja':>9} {'nat ja2vi':>9} {'%acc>=4':>8}")
for s in SYS:
    row = []
    for m in ("acc", "nat"):
        for d in ("vi2ja", "ja2vi"):
            v = [per_item[i][s][m] for i in per_item if base[i]["dir"] == d]
            row.append(mean(v))
    ok = sum(1 for i in per_item if per_item[i][s]["acc"] >= 4) / len(per_item) * 100
    print(f"{s:8} {row[0]:9.2f} {row[1]:9.2f} {row[2]:9.2f} {row[3]:9.2f} {ok:7.1f}%")

print("\n=== ACC theo domain x chieu ===")
doms = sorted({(base[i]["domain"], base[i]["dir"]) for i in per_item})
print(f"{'domain':12} {'dir':6} " + " ".join(f"{s:>7}" for s in SYS))
for dom, d in doms:
    ids = [i for i in per_item if base[i]["domain"] == dom and base[i]["dir"] == d]
    vals = [mean([per_item[i][s]["acc"] for i in ids]) for s in SYS]
    print(f"{dom:12} {d:6} " + " ".join(f"{v:7.2f}" for v in vals))

for a, b in ((SYS[0], "google"), ("haiku", "google"), ("haiku", "fable")):
    w = sum(1 for i in per_item if per_item[i][a]["acc"] > per_item[i][b]["acc"])
    t = sum(1 for i in per_item if per_item[i][a]["acc"] == per_item[i][b]["acc"])
    print(f"\nDoi dau {a} vs {b} (acc TB/cau): thang {w} / hoa {t} / thua {len(per_item)-w-t}")
