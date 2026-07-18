#!/usr/bin/env python3
"""Vòng 3 — HARVEST mở rộng từ điển thành ngữ/slang 2 CHIỀU (PLAN_RANKUP §2.1 + §8.2).

Đích: 476 mục reviewed (vòng 2) → ~2k mục. Hai phía:
  HARVEST_SIDE=ja  slang/khẩu ngữ Nhật HIỆN ĐẠI (vòng 2 mới phủ quán ngữ cổ điển)
                   → data/synthetic/gen/vong3_glosses_ja.jsonl
  HARVEST_SIDE=vi  thành ngữ/tục ngữ/slang VIỆT + cách nói Nhật tương đương (MỚI —
                   trị slang vi→ja, Google chỉ 3.3) → data/synthetic/gen/vong3_glosses_vi.jsonl

Dedup: trong file + reviewed 476 + ID_IDIOMS vòng 2. Mỗi record gắn "m"=model sinh
để review phân tầng theo provider. Chạy: bash scripts/gen_any.sh dashscope '' scripts/harvest_vong3.py
(todo không dùng — harvest tự lặp CATS × HARVEST_ROUNDS, resume = dedup theo mục).
Env: OPENAI_API_KEY/BASE_URL, GEN_MODEL, GEN_RPM, GEN_REASONING, HARVEST_SIDE, HARVEST_ROUNDS.
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
SIDE = os.environ.get("HARVEST_SIDE", "ja")
OUTF = GEN / f"vong3_glosses_{SIDE}.jsonl"
MODEL = os.environ.get("GEN_MODEL", "qwen-plus")
MIN_GAP = 60.0 / float(os.environ.get("GEN_RPM", "20"))
REASONING = os.environ.get("GEN_REASONING", "")
ROUNDS = int(os.environ.get("HARVEST_ROUNDS", "2"))
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

sys.path.insert(0, str(ROOT / "scripts"))
from gen_vong2 import ID_IDIOMS  # noqa: E402

CATS_JA = [
    "若者言葉/slang trẻ Nhật hiện đại người 20-30 tuổi dùng hằng ngày (ví dụ エモい、ガチで、詰んだ、沼る、推し、盛る)",
    "ネットスラング/từ lóng mạng xã hội Nhật còn thông dụng 2024-2026 (ví dụ 草、それな、ワンチャン、既読スルー)",
    "khẩu ngữ rút gọn hội thoại Nhật (ví dụ 〜っす、とりま、あーね、りょ、おけ)",
    "cách nói vòng/từ chối khéo/khiêm tốn kiểu Nhật hay gây hiểu lầm (ví dụ ちょっと難しい、考えておきます、前向きに検討)",
    "オノマトペ thông dụng trong hội thoại + công việc (ví dụ ざっくり、サクサク、ぐだぐだ、バタバタ、もやもや)",
    "tiếng lóng công sở/business Nhật hiện đại (ví dụ なるはや、リスケ、ペンディング、たたき台、ゴネる)",
    "tiếng lóng dev/IT hiện trường Nhật (ví dụ デグる、ポンコツコード、おまじない、神対応、椅子を温める)",
    "cách nói cảm thán/đưa đẩy hội thoại Nhật (ví dụ まさか、やっぱり、さすがに、どうりで、まじか)",
]
CATS_VI = [
    "thành ngữ/tục ngữ Việt thông dụng trong nói chuyện hằng ngày (ví dụ nước đến chân mới nhảy, mất bò mới lo làm chuồng, được voi đòi tiên)",
    "slang/khẩu ngữ giới trẻ Việt còn dùng 2024-2026 (ví dụ toang, cạn lời, gấu, chém gió, xu cà na, ăn hành, khum)",
    "cách nói ví von so sánh Việt (ví dụ đông như kiến, nghèo rớt mồng tơi, dễ như ăn kẹo, nói như máy khâu)",
    "khẩu ngữ đưa đẩy/cảm thán Việt (ví dụ thôi xong, hên xui, kệ đi, biết sao giờ, thế mới tài, ai dè)",
    "khẩu ngữ văn phòng/công việc Việt (ví dụ chạy deadline, ôm việc, đá bóng trách nhiệm, đội sổ, lươn lẹo)",
    "cách nói giảm nói tránh/mỉa mai Việt (ví dụ cũng bình thường thôi, hơi bị được, thế cũng gọi là, cạn nghĩ)",
]

P_JA = """Liệt kê 40 mục thuộc nhóm: {cat}.
Chỉ chọn mục người Nhật THẬT SỰ dùng hiện nay (2024-2026); bỏ loại đã lỗi thời (ナウい...).
{avoid}
CHỈ trả JSONL mỗi dòng: {{"ja":"<mục>","reading":"<hiragana>","vi":"<nghĩa/cách nói Việt tự nhiên tương đương, 2-10 từ>","freq":<1|2|3, 1=rất hay gặp>,"lit":"<nghĩa đen nếu KHÁC nghĩa dùng, không có thì bỏ trống>"}}
Không markdown. "vi" phải là điều người Việt NÓI trong tình huống đó (草 = "haha/chết cười", KHÔNG phải "cỏ")."""
P_VI = """Liệt kê 40 mục thuộc nhóm: {cat}.
Chỉ chọn mục người Việt THẬT SỰ dùng hiện nay; ưu tiên loại máy dịch hay dịch nghĩa đen sai.
{avoid}
CHỈ trả JSONL mỗi dòng: {{"vi":"<mục>","ja":"<cách nói Nhật TỰ NHIÊN tương đương người Nhật thật sự dùng, KHÔNG dịch từng chữ>","freq":<1|2|3>,"lit":"<nghĩa đen nếu KHÁC nghĩa dùng, không có thì bỏ trống>"}}
Không markdown. Ví dụ chuẩn: "gấu (người yêu)" → 彼氏/彼女・恋人 (KHÔNG phải クマ); "chém gió" → 話を盛る・ほらを吹く."""


def load_existing():
    # ja: dedup với mọi kho JA đã có. vi: CHỈ dedup với chính OUTF —
    # trường "vi" của kho JA là NGHĨA tiếng Việt, không phải mục từ VI (nhét vào
    # avoid-list sẽ làm model né nhầm hàng loạt mục hợp lệ).
    seen = set()
    key = "ja" if SIDE == "ja" else "vi"
    files = [GEN / "idiom_glosses_reviewed.jsonl", GEN / "idiom_glosses.jsonl", OUTF] \
        if SIDE == "ja" else [OUTF]
    for f in files:
        if f.exists():
            for line in f.open(encoding="utf-8"):
                try:
                    seen.add(json.loads(line)[key].strip())
                except Exception:
                    pass
    if SIDE == "ja":
        seen |= {x.split(" (")[0].strip() for x in ID_IDIOMS}
    return seen


def main():
    cats, prompt = (CATS_JA, P_JA) if SIDE == "ja" else (CATS_VI, P_VI)
    key = "ja" if SIDE == "ja" else "vi"
    seen = load_existing()
    print(f"harvest_vong3 side={SIDE} model={MODEL} | đã có {len(seen)} mục (dedup)", flush=True)
    fout = OUTF.open("a", encoding="utf-8", buffering=1)
    added, last = 0, 0.0
    extra = {"reasoning_effort": REASONING} if REASONING else {}
    for rnd in range(ROUNDS):
        for ci, cat in enumerate(cats):
            gap = MIN_GAP - (time.time() - last)
            if gap > 0:
                time.sleep(gap)
            last = time.time()
            avoid = ""
            if seen:
                avoid = "TRÁNH lặp mục đã có: " + "、".join(list(seen)[-80:])
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.9, max_tokens=6000,
                    messages=[{"role": "user", "content": prompt.format(cat=cat, avoid=avoid)}],
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
                k, v = (o.get(key) or "").strip(), (o.get("vi" if SIDE == "ja" else "ja") or "").strip()
                if not k or not v or k in seen:
                    continue
                seen.add(k)
                o = {kk: vv for kk, vv in o.items() if vv}
                o.update({"cat": ci, "m": MODEL})
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                added += 1
            print(f"  [r{rnd} c{ci}] +{added-n0} (tổng {added})", flush=True)
    fout.close()
    print(f"HARVEST {SIDE} XONG: +{added} mục mới -> {OUTF}", flush=True)


if __name__ == "__main__":
    main()
