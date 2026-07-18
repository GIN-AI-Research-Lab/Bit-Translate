#!/usr/bin/env python3
"""Áp verdict Haiku review CÂU SINH vòng 3 → eval/vong3_review_blacklist.jsonl.

Nguồn verdict (eval/review_v3/):
  verdicts_sent.jsonl     — mẫu phân tầng 12 panel (panel_sent_*)
  verdicts_ct_full.jsonl  — full-review ct (panelF_ct_*) + confirm (panelC_turbo/max)
Map (panel, i) → cặp (ja, vi) qua chính file panel đã lưu, ghi blacklist theo "ja"
(filter_vong3.py đọc file này ở lần chạy sau). Chạy lại an toàn (ghi đè blacklist).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RV = ROOT / "eval" / "review_v3"
BLACKLIST = ROOT / "eval" / "vong3_review_blacklist.jsonl"

# panel prefix -> pattern file panel + cách đánh i (trong file panel, field "i" đã là khóa)
PANEL_GLOB = {
    "sent_": "panel_{p}_*.jsonl",     # verdicts_sent: panel="sent_vong3_ct_qwen-plus"
    "F_ct": "panelF_ct_*.jsonl",
    "C_turbo": "panelC_turbo_00.jsonl",
    "C_max": "panelC_max_00.jsonl",
}


def load_panel_items(pattern):
    items = {}
    for f in sorted(RV.glob(pattern)):
        for line in f.open(encoding="utf-8"):
            o = json.loads(line)
            items[o["i"]] = o
    return items


def main():
    rejects = []
    for vf in [RV / "verdicts_sent.jsonl", RV / "verdicts_ct_full.jsonl"]:
        if not vf.exists():
            continue
        for line in vf.open(encoding="utf-8"):
            v = json.loads(line)
            if v.get("ok") is not False:
                continue
            p = v.get("panel", "")
            if p.startswith("sent_"):
                pattern = f"panel_{p}_*.jsonl"
            elif p == "F_ct":
                pattern = "panelF_ct_*.jsonl"
            elif p in ("C_turbo", "C_max"):
                pattern = f"panelC_{p.split('_')[1]}_00.jsonl"
            else:
                continue
            rejects.append((pattern, v["i"], v.get("reason", "")))
    cache = {}
    out, missing = [], 0
    for pattern, i, reason in rejects:
        if pattern not in cache:
            cache[pattern] = load_panel_items(pattern)
        item = cache[pattern].get(i)
        if item is None:
            missing += 1
            continue
        out.append({"ja": item["ja"], "vi": item.get("vi", ""), "reason": reason,
                    "src": item.get("src", ""), "m": item.get("m", "")})
    # dedup theo (ja, vi)
    seen, uniq = set(), []
    for o in out:
        k = (o["ja"], o["vi"])
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    BLACKLIST.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in uniq),
                         encoding="utf-8")
    print(f"blacklist: {len(uniq)} cặp (từ {len(rejects)} verdict reject, {missing} không map được)")


if __name__ == "__main__":
    main()
