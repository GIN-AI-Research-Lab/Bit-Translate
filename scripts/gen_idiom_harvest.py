#!/usr/bin/env python3
"""Giai đoạn 1 — HARVEST danh sách quán ngữ tiếng Nhật + nghĩa Việt chuẩn (gloss).

Phủ tối đa quán ngữ thông dụng theo 12 nhóm chủ đề × nhiều batch. Mỗi request
xin 40 mục {ja, reading, vi (nghĩa chuẩn), freq 1-3 (1=rất hay gặp)}.
Dedup theo ja (bỏ cả 48 idiom đã có trong gen_vong2.ID_IDIOMS).
Kết quả: data/synthetic/gen/idiom_glosses.jsonl — để (a) Claude REVIEW nghĩa,
(b) gen_vong2 mode idh sinh câu.

Env: OPENAI_API_KEY, GEN_MODEL, GEN_RPM, GEN_REASONING (nên bật medium),
     HARVEST_ROUNDS (mặc định 2 — mỗi nhóm xin 2 lần, lần sau tránh trùng lần trước).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
OUTF = GEN / "idiom_glosses.jsonl"
MODEL = os.environ.get("GEN_MODEL", "gemini-2.5-flash-lite")
MIN_GAP = 60.0 / float(os.environ.get("GEN_RPM", "14"))
REASONING = os.environ.get("GEN_REASONING", "medium")
ROUNDS = int(os.environ.get("HARVEST_ROUNDS", "2"))
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL",
                                        "https://generativelanguage.googleapis.com/v1beta/openai/"),
                api_key=os.environ["OPENAI_API_KEY"])

sys.path.insert(0, str(ROOT / "scripts"))
from gen_vong2 import ID_IDIOMS  # noqa: E402 — 48 idiom đã có, tránh trùng

CATS = [
    "quán ngữ với bộ phận cơ thể: 目・耳・口・鼻・顔 (ví dụ 目が高い、口を挟む)",
    "quán ngữ với bộ phận cơ thể: 手・足・腕・肩・腰 (ví dụ 手を打つ、足が出る)",
    "quán ngữ với bộ phận cơ thể: 頭・胸・腹・心・気 (ví dụ 腹を括る、気が置けない)",
    "quán ngữ dùng trong CÔNG SỞ/business Nhật (ví dụ 根回し、たたき台、五月雨式)",
    "quán ngữ về tiền bạc/kinh doanh (ví dụ 足が出る、懐が寂しい、赤字続き)",
    "quán ngữ với động vật (ví dụ 猫の手も借りたい、馬の耳に念仏)",
    "quán ngữ với thiên nhiên/thời tiết/nước (ví dụ 水掛け論、雲をつかむよう)",
    "quán ngữ khẩu ngữ hội thoại hằng ngày (ví dụ 気が向く、腑抜け、けりをつける)",
    "四字熟語 THÔNG DỤNG trong nói/viết hằng ngày (ví dụ 一石二鳥、本末転倒、臨機応変)",
    "ことわざ thông dụng người Nhật hay dùng khi nói (ví dụ 石の上にも三年、急がば回れ)",
    "cách nói ẩn dụ về cảm xúc/tâm trạng (ví dụ 胸が痛む、頭に来る、気が滅入る)",
    "quán ngữ về thời gian/tiến độ/công việc (ví dụ 峠を越す、目処が立つ、追い込みをかける)",
]

PROMPT = """Liệt kê 40 quán ngữ/thành ngữ tiếng Nhật THÔNG DỤNG thuộc nhóm: {cat}.
Chỉ chọn loại người Nhật THẬT SỰ dùng trong nói/viết hiện đại (bỏ loại cổ/sách vở hiếm).
{avoid}
CHỈ trả JSONL, mỗi dòng: {{"ja":"<quán ngữ>","reading":"<hiragana>","vi":"<nghĩa tiếng Việt CHÍNH XÁC, tự nhiên, 3-10 từ>","freq":<1|2|3, 1=rất hay gặp>}}
Không markdown, không giải thích. Nghĩa vi phải là NGHĨA BÓNG (ví dụ 油を売る = "la cà lười biếng", KHÔNG phải "bán dầu")."""


def load_existing():
    seen = {x.split(" (")[0].strip() for x in ID_IDIOMS}
    if OUTF.exists():
        for line in OUTF.open(encoding="utf-8"):
            try:
                seen.add(json.loads(line)["ja"])
            except Exception:
                pass
    return seen


def main():
    seen = load_existing()
    fout = OUTF.open("a", encoding="utf-8", buffering=1)
    added, last = 0, 0.0
    extra = {"reasoning_effort": REASONING} if REASONING else {}
    for rnd in range(ROUNDS):
        for ci, cat in enumerate(CATS):
            gap = MIN_GAP - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.time()
            # tránh trùng: đưa các idiom đã có CÙNG NHÓM vào phần cấm (cắt 80 cái gần nhất cho gọn prompt)
            avoid = ""
            if seen:
                sample = list(seen)[-80:]
                avoid = "TRÁNH lặp các mục đã có: " + "、".join(sample)
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.9, max_tokens=6000,
                    messages=[{"role": "user", "content": PROMPT.format(cat=cat, avoid=avoid)}],
                    **extra)
            except Exception as e:  # noqa: BLE001
                print(f"  [r{rnd} c{ci}] LỖI {str(e)[:70]}", flush=True)
                continue
            txt = re.sub(r"^```[a-z]*\n?|```$", "",
                         (r.choices[0].message.content or "").strip(), flags=re.M)
            n0 = added
            for line in txt.splitlines():
                line = line.strip().rstrip(",")
                if not line.startswith("{"):
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ja, vi = (o.get("ja") or "").strip(), (o.get("vi") or "").strip()
                if not ja or not vi or ja in seen:
                    continue
                seen.add(ja)
                fout.write(json.dumps({"ja": ja, "reading": o.get("reading", ""), "vi": vi,
                                       "freq": o.get("freq", 2), "cat": ci},
                                      ensure_ascii=False) + "\n")
                added += 1
            print(f"  [r{rnd} c{ci}] +{added-n0} (tổng {added})", flush=True)
    fout.close()
    print(f"HARVEST XONG: +{added} idiom mới -> {OUTF}", flush=True)


if __name__ == "__main__":
    main()
