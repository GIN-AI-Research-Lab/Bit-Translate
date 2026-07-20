#!/usr/bin/env python3
"""Sinh corpus KD Đợt 1 (PLAN_KD_JA2VI): dịch ja→vi hàng loạt bằng 1 thầy (API),
từ file nguồn JA 1-câu/dòng. Dùng thẳng cùng prompt/clean() đã kiểm chứng ở
eval/run_hardbench_api.py (không dùng bản draft của 292M — dịch thẳng từ JA thật).

  OPENAI_BASE_URL=... OPENAI_API_KEY=... MODEL=qwen-plus LABEL=qwen-plus \
  SRC=data/synthetic/kd/ja_pool.txt START=0 COUNT=15000 RPM=30 \
  python3 scripts/gen_kd_corpus.py

Ghi data/synthetic/kd/raw_<LABEL>.jsonl {ja, vi, m: model} — RESUME theo số dòng
đã có (an toàn chạy lại khi đứt quota/mạng). START/COUNT chia shard không chồng
lấn giữa các model chạy song song trên CÙNG file nguồn.
Env thêm: EXTRA_BODY (json, vd tắt thinking GLM), MAX_RETRY=3.
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
MODEL = os.environ["MODEL"]
LABEL = os.environ.get("LABEL", MODEL)
SRC = Path(os.environ.get("SRC", "data/synthetic/kd/ja_pool.txt"))
START = int(os.environ.get("START", "0"))
COUNT = int(os.environ.get("COUNT", "0")) or None   # 0/unset = tới hết file
MIN_GAP = 60.0 / float(os.environ.get("RPM", "20"))
MAX_RETRY = int(os.environ.get("MAX_RETRY", "3"))
EXTRA_BODY = json.loads(os.environ["EXTRA_BODY"]) if os.environ.get("EXTRA_BODY") else {}

OUTDIR = ROOT / "data" / "synthetic" / "kd"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTP = OUTDIR / f"raw_{LABEL}.jsonl"

client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

PROMPT = ("Bạn là dịch giả chuyên nghiệp Nhật-Việt. Dịch câu tiếng Nhật sau sang "
          "tiếng Việt tự nhiên, giữ đúng nghĩa, sắc thái và văn phong. "
          "Chỉ trả về DUY NHẤT bản dịch, không giải thích.\n\n{src}")


def clean(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    txt = " ".join(txt.strip().split())
    for a, b in (('"', '"'), ("「", "」"), ("“", "”")):
        if txt.startswith(a) and txt.endswith(b) and len(txt) > 2:
            txt = txt[1:-1].strip()
    return txt


def translate(src):
    err = None
    for i in range(MAX_RETRY):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=1500,
                messages=[{"role": "user", "content": PROMPT.format(src=src)}],
                extra_body=EXTRA_BODY)
            hyp = clean(r.choices[0].message.content)
            if hyp:
                return hyp
            err = "empty"
        except Exception as e:  # noqa: BLE001
            err = str(e)[:150]
        time.sleep(2 * (i + 1))
    raise RuntimeError(err)


def main():
    all_lines = [l.strip() for l in SRC.open(encoding="utf-8") if l.strip()]
    end = START + COUNT if COUNT else len(all_lines)
    shard = all_lines[START:end]

    done = 0
    if OUTP.exists():
        done = sum(1 for _ in OUTP.open(encoding="utf-8"))
    todo = shard[done:]
    print(f"[{LABEL}] model={MODEL} shard=[{START}:{end}] ({len(shard)} câu) "
          f"| {done} đã có, còn {len(todo)}", flush=True)

    last = 0.0
    n_err = 0
    with OUTP.open("a", encoding="utf-8") as f:
        for k, ja in enumerate(todo):
            gap = MIN_GAP - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.time()
            try:
                vi = translate(ja)
            except RuntimeError as e:
                n_err += 1
                print(f"  #{done+k}: LỖI {e}", flush=True)
                if n_err >= 5:
                    print("  5 lỗi liên tiếp -- dừng (chạy lại script để resume)", flush=True)
                    break
                continue
            n_err = 0
            f.write(json.dumps({"ja": ja, "vi": vi, "m": MODEL}, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 200 == 0:
                print(f"  {done+k+1}/{len(shard)}", flush=True)

    total = sum(1 for _ in OUTP.open(encoding="utf-8")) if OUTP.exists() else 0
    print(f"[{LABEL}] XONG (hoặc dừng giữa chừng) — {total} câu trong {OUTP}", flush=True)


if __name__ == "__main__":
    main()
