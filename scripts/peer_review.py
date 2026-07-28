#!/usr/bin/env python3
"""Gửi một hồ sơ/plan cho model NGOÀI phản biện, rồi lưu lại để đối chiếu.

Vì sao cần: suốt vòng 5-6 chỉ có MỘT người phân tích (Claude), và nó đã sai ít nhất
2 lần mà không ai bắt được — đọc dốc loss trên một cửa sổ 1000 step nhiễu rồi kết luận
"giảm nhanh hơn v5 5,5 lần" (hồi quy 4000 step thật ra chỉ −0,0052), và trừ chrF của
bench này với điểm judge của bench khác. Một model khác họ, không dính vào lập luận cũ,
bắt được những chỗ đó.

⚠️ REVIEW CỦA NÓ PHẢI ĐƯỢC KIỂM LẠI, KHÔNG ĐỌC THẲNG. Lần chạy đầu (2026-07-28) nó
bắt 5 lỗi: 3 cái kiểm ra ĐÚNG (train.py không eval dev, không log tỉ lệ ternary,
init_round bỏ optimizer state), 2 cái SAI — nó so loss v6-trên-corpus-v6 với
loss v5-trên-corpus-v5 rồi kết luận "quên kiến thức cũ" (khác corpus, không so được),
và gọi việc bỏ optimizer state là bug trong khi docstring init_round.py ghi rõ là chủ ý.

Model: nemotron-3-ultra-550b (free, context 1M) là bản dùng được. Đã thử và loại:
gpt-oss-20b:free (20B, ra văn bản lẫn tiếng Nga), gpt-4.1 qua GitHub Models (bịa số).
GPT-5 KHÔNG gọi được: OpenAI API cần nạp tiền, GitHub Models trả unavailable_model,
OpenRouter chỉ free cho nhóm ':free'.

  python scripts/peer_review.py <file_hoso.md> [--model M] [--out F]

Dựng hồ sơ: đưa SỐ LIỆU THÔ + văn bản plan nguyên văn + liệt kê kết luận của mình
thành các mệnh đề để nó công phá. Đừng chỉ đưa bản tóm tắt — review sẽ bị chính cách
mình diễn giải dẫn dắt.
"""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
SYS = ("Bạn là chuyên gia machine translation độc lập, được thuê để PHẢN BIỆN. "
       "Mặc định là chuỗi lập luận trong hồ sơ CÓ LỖI — hãy tìm ra. Ưu tiên chỉ rõ "
       "SỐ LIỆU NÀO bác KẾT LUẬN NÀO; nếu một kết luận không có số liệu chống lưng "
       "thì nói thẳng là thiếu bằng chứng. Viết tiếng Việt, ngắn gọn, có cấu trúc, "
       "không khách sáo, không nhắc lại hồ sơ.")


def env(name):
    for ln in open(ROOT / ".env", encoding="utf-8"):
        ln = ln.strip()
        if ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        if k.strip() == name:
            return v.strip()
    raise SystemExit(f"thiếu {name} trong .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", help="file hồ sơ (.md) cần review")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-tokens", type=int, default=9000)
    a = ap.parse_args()

    text = Path(a.packet).read_text(encoding="utf-8")
    print(f"hồ sơ {a.packet}: {len(text):,} ký tự (~{len(text)//3:,} token)")

    body = {"model": a.model,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": text}],
            "max_tokens": a.max_tokens}
    r = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}",
                 "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(r, timeout=900))
    except urllib.error.HTTPError as e:
        try:
            raise SystemExit("LỖI: " + json.loads(e.read().decode())["error"]["message"])
        except (KeyError, ValueError):
            raise SystemExit(f"LỖI HTTP {e.code}")

    # OpenRouter trả lỗi (rate-limit, model đang nghẽn) KÈM HTTP 200 và không có
    # "choices" -> phải bắt ở đây, không thì KeyError che mất thông báo thật.
    if "choices" not in d:
        raise SystemExit("API không trả choices. Nguyên văn: "
                         + json.dumps(d, ensure_ascii=False)[:500])
    msg = d["choices"][0]["message"]
    # model reasoning có thể trả rỗng ở content và đặt nội dung ở 'reasoning'
    out = (msg.get("content") or "") or (msg.get("reasoning") or "")
    if not out.strip():
        raise SystemExit("model trả về rỗng — thử lại hoặc giảm --max-tokens")

    dst = Path(a.out) if a.out else (ROOT / "eval" /
                                     f"peer_review_{Path(a.packet).stem}.md")
    dst.parent.mkdir(parents=True, exist_ok=True)
    u = d.get("usage", {})
    dst.write_text(f"<!-- {a.model} | in {u.get('prompt_tokens')} "
                   f"out {u.get('completion_tokens')} -->\n\n" + out, encoding="utf-8")
    print(f"-> {dst}  ({len(out):,} ký tự)")
    print("\n⚠️  KIỂM TỪNG KHẲNG ĐỊNH trước khi dùng — lần đầu nó sai 2/5.")


if __name__ == "__main__":
    main()
