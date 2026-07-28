#!/usr/bin/env python3
"""Phân loại NGUYÊN NHÂN 69 câu v5 dịch hỏng — để biết vòng 6 phải chữa cái gì.

Vì sao cần: kế hoạch vòng 6 ban đầu chỉ nhắm ĐỘ PHỦ THUẬT NGỮ. Nhưng đọc tay 22 câu
hỏng thấy KHÔNG CÓ câu nào hỏng vì thiếu thuật ngữ chuyên ngành — toàn lỗi ẩn chủ ngữ,
cấu trúc câu, katakana. Nếu đúng vậy thì vòng 6 theo kế hoạch cũ sẽ không nhích được
bench phổ thông chút nào. Phải đo trước khi tiêu tiền train.

Chấm bằng Gemini (bên thứ ba, không phải tôi tự đọc) trên đúng những câu ĐÃ được chấm
mù là hỏng, và có bản Google dịch đúng làm đối chứng -> lỗi là do model, không phải câu
bất khả dịch.

  python scripts/error_taxonomy_v5.py
"""
import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent

CATS = {
    "an_chu_ngu": "Ẩn chủ ngữ: tiếng Nhật lược chủ ngữ, bản dịch chọn SAI ngôi (thêm 'tôi' khi phải là 'họ/anh ấy/bạn', hoặc đổi ngôi giữa câu)",
    "cau_truc_cau": "Sai cấu trúc: phân tích sai quan hệ ngữ pháp — câu chẻ のは...です, đảo chủ-vị, gán sai trợ từ を/が/に, mệnh đề lồng",
    "phu_dinh_modality": "Sai phủ định hoặc thái độ: phủ định kép, ～ずして, ～のに, ～ばいい, mức độ chắc chắn/mong muốn bị đổi",
    "bi_dong_sai_khien_thu_nhan": "Sai thể bị động/sai khiến/cho-nhận: ～られる ～させる ～てもらう ～てくれる — đảo ai làm cho ai",
    "katakana_ngoai_lai": "Katakana ngoại lai dịch sai: dịch nghĩa đen thay vì nhận ra từ mượn (マジックテープ→Velcro, không phải 'băng dính ma thuật')",
    "ten_rieng": "Tên riêng sai: tên người/địa danh/thương hiệu bị dịch nghĩa hoặc đọc sai (2パック→'hai cái pack' thay vì 2Pac)",
    "quan_ngu_thanh_ngu": "Quán ngữ/thành ngữ/cụm 4 chữ dịch nghĩa đen mất ý (両者両得)",
    "thuat_ngu_chuyen_nganh": "Thiếu thuật ngữ chuyên ngành: từ chuyên môn của một lĩnh vực bị dịch sai vì model chưa từng gặp",
    "them_bot_nghia": "Thêm hoặc bỏ nội dung: chèn ý không có trong nguồn, hoặc bỏ sót vế/thông tin",
    "tu_vung_thong_thuong": "Chọn sai nghĩa từ thông thường (không chuyên ngành): 微妙, 靴ヒモ dịch thành từ khác nghĩa",
    "van_phong_tu_nhien": "Nghĩa đúng nhưng diễn đạt tiếng Việt cứng/không tự nhiên",
}

PROMPT = """Bạn chấm lỗi dịch Nhật→Việt. Dưới đây là câu tiếng Nhật, bản dịch SAI của model, và bản dịch ĐÚNG để đối chiếu.

NHẬT   : {src}
MODEL  : {hyp}
ĐÚNG   : {ref}

Các loại lỗi:
{cats}

Nhiệm vụ: chỉ ra loại lỗi CHÍNH khiến bản dịch model không dùng được, và loại phụ nếu có.
Chọn đúng mã loại trong danh sách trên. Nếu nhiều lỗi, đặt lỗi NGHIÊM TRỌNG NHẤT làm chính.

Trả về DUY NHẤT JSON:
{{"chinh":"ma_loai","phu":["ma_loai"],"giai_thich":"một câu ngắn"}}"""


def load_keys():
    ks = []
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*gemini_key_(\d+)\s*=\s*(\S+)", line)
        if m:
            ks.append(m.group(2))
    return ks


async def main_async(a):
    keys = load_keys()
    fails = json.loads(Path(a.fails).read_text(encoding="utf-8"))
    fails = [f for f in fails if f.get("google")]
    cats = "\n".join(f"  {k}: {v}" for k, v in CATS.items())
    print(f"{len(fails)} câu hỏng | {len(keys)} key", flush=True)

    out = {}
    sem = asyncio.Semaphore(6)

    async def one(i, f):
        async with sem:
            c = genai.Client(api_key=keys[i % len(keys)])
            for att in range(3):
                try:
                    r = await c.aio.models.generate_content(
                        model="gemini-flash-latest",
                        contents=PROMPT.format(src=f["src"], hyp=f["v5"],
                                               ref=f["google"], cats=cats),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json", temperature=0.1))
                    t = re.sub(r"^```[a-z]*\s*|```\s*$", "", (r.text or "").strip(),
                               flags=re.I | re.M)
                    o = json.loads(t)
                    if o.get("chinh") in CATS:
                        out[f["id"]] = {**o, **f}
                        return
                except Exception:
                    await asyncio.sleep(2 + att * 4)

    await asyncio.gather(*[one(i, f) for i, f in enumerate(fails)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fails", default="C:/Users/OS/.claude/jobs/6fdeecab/tmp/v5_fails.json")
    ap.add_argument("--out", default="eval/error_taxonomy_v5.json")
    a = ap.parse_args()

    res = asyncio.run(main_async(a))
    n = len(res)
    main_c = Counter(v["chinh"] for v in res.values())
    any_c = Counter()
    for v in res.values():
        any_c[v["chinh"]] += 1
        for p in v.get("phu", []) or []:
            if p in CATS and p != v["chinh"]:
                any_c[p] += 1

    print(f"\n=== NGUYÊN NHÂN {n} CÂU v5 HỎNG (Google dịch được) ===\n")
    print(f"{'loại lỗi':<32}{'là lỗi CHÍNH':>14}{'  xuất hiện (kể cả phụ)':>24}")
    print("-" * 72)
    for k, _ in main_c.most_common():
        print(f"{k:<32}{main_c[k]:>7} ({100*main_c[k]/n:>3.0f}%){any_c[k]:>15} ({100*any_c[k]/n:>3.0f}%)")
    for k in CATS:
        if k not in main_c:
            print(f"{k:<32}{0:>7} (  0%){any_c[k]:>15} ({100*any_c[k]/n:>3.0f}%)")

    Path(a.out).write_text(json.dumps(
        {"n": n, "main": dict(main_c), "any": dict(any_c), "rows": list(res.values())},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
