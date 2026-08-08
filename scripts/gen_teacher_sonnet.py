#!/usr/bin/env python3
"""Thầy dịch KD ja->vi bằng Claude Sonnet 5 (PLAN_V8 §T3/§T4 — user chốt 2026-08-08).

Vai: nhận CÂU JA NGUỒN (mine corpus hoặc sinh) -> dịch VI tự nhiên + đúng nghĩa.
Khác gen_via_api.py (cái đó sinh CẢ ja lẫn vi từ list term). Ở đây câu JA cho sẵn,
thầy chỉ dịch -> tránh model đổi câu nguồn (in-out theo INDEX, không re-emit JA).

Vì sao Sonnet 5 (không phải OpenAI-shim): tiếng Việt tự nhiên hơn Google/Gemini-flash
-> giữ guardrail nat (điểm mạnh nhất của v7a: nat 94%). SDK chính thức `anthropic`,
KHÔNG dùng openai-compat. Sonnet 5: effort low (dịch không phải suy luận nặng),
cache glossary+system, CẤM truyền temperature/top_p (Sonnet 5 trả 400).

Glossary: quét câu nguồn theo eval/glossary_v8_candidates.csv, chèn gợi ý [ja=vi] để
thầy render đúng canonical (chặn lỗi 1-từ kiểu デフレ->lạm phát, ケース->trường hợp).
Register: PIN_VI/KEEP = pin cứng; AUDIENCE = ưu tiên VI trừ ngữ cảnh IT.

Kiểm inline: PIN_VI term có mặt trong bản dịch? (bắt sớm lỗi đảo nghĩa). LaBSE>=0.55
là GATE RIÊNG chạy sau (nợ sentence-transformers), không nhét vào đây.

Cách dùng:
  export ANTHROPIC_API_KEY=...        # hoặc `ant auth login`
  python scripts/gen_teacher_sonnet.py --src pilot300.ja --out pilot300.jsonl --effort low
  # pilot audit-300 TRƯỚC khi scale (luật dự án: audit thầy trước khi bung)

Cần: pip install anthropic
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
GLOSS = ROOT / "eval" / "glossary_v8_candidates.csv"

SYSTEM = """Bạn là biên dịch viên Nhật->Việt chuyên nghiệp. Dịch từng câu tiếng Nhật sang tiếng Việt \
TỰ NHIÊN như người Việt viết, KHÔNG dịch word-by-word, KHÔNG cứng nhắc.

Nguyên tắc:
- Giữ ĐÚNG nghĩa, đúng cực (khẳng định/phủ định), đúng thì, đúng hướng hành động.
- Câu hỏi tu từ (〜と思わない?, 〜じゃないか) giữ nguyên sắc thái tu từ, KHÔNG lật thành phủ định phẳng.
- Thành ngữ/tục ngữ: dịch theo NGHĨA hoặc thành ngữ Việt tương đương, KHÔNG dịch mặt chữ.
- Kính ngữ thương mại (査収/折り返す/ご清栄...) dịch đúng chuẩn văn phòng.
- Katakana: mặc định dùng từ Việt chuẩn cho người đọc phổ thông (ログイン->đăng nhập, \
バックアップ->sao lưu); chỉ giữ nguyên loanword đã Việt hoá (karaoke, ramen). \
KHÔNG giữ tiếng Anh quá tay làm câu kém tự nhiên.
- Nếu đầu câu có gợi ý dạng [từ_ja=nghĩa_vi], PHẢI render từ đó đúng theo nghĩa gợi ý.

Đầu vào là danh sách câu đánh số. Trả về JSONL, MỖI DÒNG một object {"i":<số>,"vi":"<bản dịch>"}. \
KHÔNG lặp lại câu tiếng Nhật, KHÔNG giải thích, KHÔNG markdown, KHÔNG bọc ```."""


def load_glossary():
    """-> list (ja_term, vi, register) sắp theo độ dài giảm dần (khớp cụm dài trước)."""
    rows = []
    with open(GLOSS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ja = (r.get("ja") or "").strip()
            vi = (r.get("vi") or "").strip()
            if ja and vi and not ja.isascii():
                rows.append((ja, vi, (r.get("register") or "").strip()))
    rows.sort(key=lambda x: -len(x[0]))
    return rows


def hints_for(ja, gloss):
    """Chèn gợi ý [term=vi] cho các term glossary xuất hiện trong câu (tối đa 3/câu)."""
    hs, pins = [], []
    for term, vi, reg in gloss:
        if term in ja:
            hs.append(f"[{term}={vi}]")
            if reg == "PIN_VI":
                pins.append((term, vi))
            if len(hs) >= 3:
                break
    return ("".join(hs) + " " if hs else ""), pins


def parse_out(txt):
    out = {}
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            p = json.loads(line)
            if "i" in p and p.get("vi"):
                out[int(p["i"])] = p["vi"].strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="file .ja (1 câu/dòng) HOẶC .jsonl có field ja")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="low cho câu thường; medium cho bucket idiom/phủ định khó")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="0 = tất cả")
    a = ap.parse_args()

    src = Path(a.src)
    lines = src.read_text(encoding="utf-8").splitlines()
    if src.suffix == ".jsonl":
        ja_list = [json.loads(l)["ja"] for l in lines if l.strip()]
    else:
        ja_list = [l.strip() for l in lines if l.strip()]
    if a.limit:
        ja_list = ja_list[:a.limit]

    outp = Path(a.out)
    done = set()
    if outp.exists():  # resume theo index đã ghi
        for l in outp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done.add(int(json.loads(l)["idx"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
    todo = [i for i in range(len(ja_list)) if i not in done]
    print(f"model={a.model} effort={a.effort} | {len(ja_list)} câu, còn {len(todo)}", flush=True)

    gloss = load_glossary()
    client = anthropic.Anthropic()
    fout = outp.open("a", encoding="utf-8")
    tin = tout = tcache = 0
    pin_miss = 0

    for b0 in range(0, len(todo), a.batch):
        idxs = todo[b0:b0 + a.batch]
        numbered, pin_map = [], {}
        for n, i in enumerate(idxs):
            hint, pins = hints_for(ja_list[i], gloss)
            numbered.append(f"{n}. {hint}{ja_list[i]}")
            if pins:
                pin_map[n] = pins
        user = "Dịch các câu sau sang tiếng Việt tự nhiên:\n\n" + "\n".join(numbered)
        try:
            r = client.messages.create(
                model=a.model, max_tokens=a.max_tokens,
                output_config={"effort": a.effort},
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  batch @{b0}: LỖI {str(e)[:100]} — bỏ, chạy lại resume", flush=True)
            continue
        tin += r.usage.input_tokens
        tout += r.usage.output_tokens
        tcache += getattr(r.usage, "cache_read_input_tokens", 0) or 0
        txt = "".join(c.text for c in r.content if c.type == "text")
        got = parse_out(txt)
        for n, i in enumerate(idxs):
            vi = got.get(n)
            if not vi:
                continue
            for term, cvi in pin_map.get(n, []):  # kiểm PIN_VI có mặt
                if cvi.split("/")[0].strip().lower() not in vi.lower():
                    pin_miss += 1
            fout.write(json.dumps({"idx": i, "ja": ja_list[i], "vi": vi,
                                   "dir": "ja2vi", "src": "sonnet5"},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        print(f"  @{b0}: +{len(got)}/{len(idxs)} | in {tin:,} out {tout:,} "
              f"cache {tcache:,} | pin_miss {pin_miss}", flush=True)
        time.sleep(0.3)

    fout.close()
    # giá intro Sonnet 5: $2/1M in, $10/1M out (đến 2026-08-31)
    cost = tin / 1e6 * 2 + tout / 1e6 * 10
    print(f"XONG. in {tin:,} out {tout:,} cache {tcache:,} | ~${cost:.2f} (giá intro) "
          f"| pin_miss {pin_miss} (kiểm tay các câu này)", flush=True)


if __name__ == "__main__":
    main()
