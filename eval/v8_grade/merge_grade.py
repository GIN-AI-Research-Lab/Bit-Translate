#!/usr/bin/env python3
"""Gộp verdict chấm mù + giải mã key -> con số v8 vs v7a thật."""
import json
from collections import Counter
S = "C:/Users/ADMINI~1/AppData/Local/Temp/claude/F--Project-Ai-Bit-Translate/cb22c430-19c2-41f3-9d50-82854664bfd4/scratchpad"

def load(p): return json.load(open(f"{S}/{p}", encoding="utf-8"))

key64 = {k["id"]: k for k in load("grade_err64_key.json")}
key150 = {k["id"]: k for k in load("grade_rand150_key.json")}

v64 = load("verdict_ge64_c1.json") + load("verdict_ge64_c2.json")
v150 = load("verdict_gr150_c1.json") + load("verdict_gr150_c2.json") + load("verdict_gr150_c3.json")

def analyze(verdicts, keys, name):
    win = Counter()          # v8 / v7a / tie
    v8q = Counter(); v7aq = Counter()
    rows = []
    for v in verdicts:
        k = keys[v["id"]]
        a_sys, b_sys = k["A_sys"], k["B_sys"]
        w = v["winner"]
        winner_sys = "tie" if w == "tie" else (a_sys if w == "A" else b_sys)
        win[winner_sys] += 1
        v8_q = v["a_q"] if a_sys == "v8" else v["b_q"]
        v7a_q = v["a_q"] if a_sys == "v7a" else v["b_q"]
        v8q[v8_q] += 1; v7aq[v7a_q] += 1
        rows.append({"id": v["id"], "winner": winner_sys, "v8_q": v8_q,
                     "v7a_q": v7a_q, "domain": k.get("domain", ""),
                     "why": v.get("why", ""), "ja": k["ja"]})
    n = len(verdicts)
    print(f"\n===== {name} (n={n}) =====")
    print(f"  THẮNG:  v8={win['v8']}  v7a={win['v7a']}  tie={win['tie']}"
          f"  |  v8 win-rate (bỏ tie) = {win['v8']}/{win['v8']+win['v7a']}"
          f" = {100*win['v8']/max(win['v8']+win['v7a'],1):.0f}%")
    print(f"  Chất lượng v8 : good={v8q['good']}  ok={v8q['ok']}  bad={v8q['bad']}")
    print(f"  Chất lượng v7a: good={v7aq['good']}  ok={v7aq['ok']}  bad={v7aq['bad']}")
    return win, v8q, v7aq, rows

w64, v8q64, v7aq64, rows64 = analyze(v64, key64, "TẬP 64 CÂU v7a TỪNG SAI (trọng tâm)")
w150, v8q150, v7aq150, rows150 = analyze(v150, key150, "TẬP 150 CÂU RANDOM (no-regression)")

# Trên tập 64: v7a vốn SAI (bad/ok) hết -> đo v8 sửa được bao nhiêu = v8 'good'
fixed = [r for r in rows64 if r["v8_q"] == "good"]
still_bad = [r for r in rows64 if r["v8_q"] == "bad"]
print(f"\n>>> TẬP-64: v8 đạt 'good' (sửa dứt điểm) = {len(fixed)}/64 = {100*len(fixed)/64:.0f}%")
print(f">>> TẬP-64: v8 vẫn 'bad' (chưa sửa được)  = {len(still_bad)}/64")

# no-regression: câu v8 'bad' mà v7a không 'bad' = HỎNG THÊM (đáng lo)
regress = [r for r in rows150 if r["v8_q"] == "bad" and r["v7a_q"] != "bad"]
print(f"\n>>> TẬP-150: câu v8 làm HỎNG THÊM (v8 bad, v7a không bad) = {len(regress)}/150")
for r in regress:
    print(f"    id{r['id']}: {r['why']}")

# lưu bảng chi tiết
json.dump({"err64": rows64, "rand150": rows150},
          open(f"{S}/grade_merged.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\nchi tiết -> grade_merged.json")

# domain breakdown tập 64
print("\n--- Tập-64 theo domain (v8 win / v7a win / tie) ---")
dom = {}
for r in rows64:
    d = r["domain"] or "?"
    dom.setdefault(d, Counter())[r["winner"]] += 1
for d, c in sorted(dom.items()):
    print(f"  {d:24s} v8={c['v8']} v7a={c['v7a']} tie={c['tie']}")
