#!/usr/bin/env python3
"""Chạy hardbench200 qua một model API (OpenAI-compatible) — tuyển THẦY cho KD
(PLAN_KD_JA2VI Đợt 1): đo xem Gemini/Qwen/GLM free-quota đứng rank nào so
Google/Haiku trước khi chọn thầy dịch corpus.

  OPENAI_BASE_URL=... OPENAI_API_KEY=... MODEL=qwen-plus LABEL=qwen-plus RPM=30 \
  python3 eval/run_hardbench_api.py

Ghi eval/hardbench_<LABEL>.jsonl (fields y hệt run_hardbench.py: id domain dir
src ref hyp chrf) — dùng thẳng được cho build_judge_panels_4way.py.
Resume theo id đã có trong file output (an toàn chạy lại khi đứt quota).
Env: DIR=ja2vi|vi2ja|both (mặc định both), RPM (mặc định 20), MAX_RETRY=3,
     MAX=N (giới hạn số câu — smoke test),
     EXTRA_BODY='{"thinking":{"type":"disabled"}}' (param riêng provider — vd tắt
     thinking GLM-4.7-flash, không tắt thì reasoning ngốn hết max_tokens -> content rỗng).
"""
import json
import os
import re
import time
from pathlib import Path

import sacrebleu
from openai import OpenAI

ROOT = Path(__file__).parent.parent
MODEL = os.environ["MODEL"]
LABEL = os.environ.get("LABEL", MODEL)
ONLY_DIR = os.environ.get("DIR", "both")
MIN_GAP = 60.0 / float(os.environ.get("RPM", "20"))
MAX_RETRY = int(os.environ.get("MAX_RETRY", "3"))
EXTRA_BODY = json.loads(os.environ["EXTRA_BODY"]) if os.environ.get("EXTRA_BODY") else {}
OUTP = ROOT / "eval" / f"hardbench_{LABEL}.jsonl"

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

PROMPT = {
    "ja2vi": "Bạn là dịch giả chuyên nghiệp Nhật-Việt. Dịch câu tiếng Nhật sau sang "
             "tiếng Việt tự nhiên, giữ đúng nghĩa, sắc thái và văn phong. "
             "Chỉ trả về DUY NHẤT bản dịch, không giải thích.\n\n{src}",
    "vi2ja": "Bạn là dịch giả chuyên nghiệp Việt-Nhật. Dịch câu tiếng Việt sau sang "
             "tiếng Nhật tự nhiên, giữ đúng nghĩa, sắc thái và văn phong. "
             "Chỉ trả về DUY NHẤT bản dịch, không giải thích.\n\n{src}",
}


def clean(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)   # model thinking lộ tag
    txt = " ".join(txt.strip().split())                               # gộp newline/space
    # lột cặp ngoặc kép bọc toàn bộ (một số model tự thêm)
    for a, b in (('"', '"'), ("「", "」"), ("“", "”")):
        if txt.startswith(a) and txt.endswith(b) and len(txt) > 2:
            txt = txt[1:-1].strip()
    return txt


def translate(src, d):
    err = None
    for i in range(MAX_RETRY):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=1500,
                messages=[{"role": "user", "content": PROMPT[d].format(src=src)}],
                extra_body=EXTRA_BODY)
            hyp = clean(r.choices[0].message.content)
            if hyp:
                return hyp
            err = "empty"
        except Exception as e:  # noqa: BLE001
            err = str(e)[:120]
        time.sleep(2 * (i + 1))
    raise RuntimeError(err)


def main():
    probes = [json.loads(l) for l in (ROOT / "eval" / "hardbench200.jsonl").open(encoding="utf-8")]
    if ONLY_DIR != "both":
        probes = [p for p in probes if p["dir"] == ONLY_DIR]
    done = set()
    if OUTP.exists():
        done = {json.loads(l)["id"] for l in OUTP.open(encoding="utf-8") if l.strip()}
    todo = [p for p in probes if p["id"] not in done]
    if os.environ.get("MAX"):
        todo = todo[:int(os.environ["MAX"])]
    print(f"[{LABEL}] model={MODEL} dir={ONLY_DIR} | {len(done)} có sẵn, còn {len(todo)}", flush=True)

    last = 0.0
    with OUTP.open("a", encoding="utf-8") as f:
        for k, p in enumerate(todo):
            gap = MIN_GAP - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.time()
            try:
                hyp = translate(p["src"], p["dir"])
            except RuntimeError as e:
                print(f"  id={p['id']}: LỖI {e} — dừng (chạy lại để resume)", flush=True)
                break
            score = sacrebleu.sentence_chrf(hyp, [p["ref"]]).score
            f.write(json.dumps({**p, "hyp": hyp, "chrf": round(score, 1)},
                               ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(todo)}", flush=True)

    rows = [json.loads(l) for l in OUTP.open(encoding="utf-8")]
    print(f"=== [{LABEL}] {len(rows)} câu — chrF TB theo chiều ===", flush=True)
    for d in ("vi2ja", "ja2vi"):
        v = [r["chrf"] for r in rows if r["dir"] == d]
        if v:
            print(f"  {d}: {sum(v)/len(v):.1f} (n={len(v)})", flush=True)


if __name__ == "__main__":
    main()
