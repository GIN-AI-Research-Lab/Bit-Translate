#!/usr/bin/env python3
"""Sinh câu glossary qua API OpenAI-compatible (OpenRouter / DeepSeek / Qwen / Kimi...).

Lặp các batch trong data/synthetic/gen/todo_remaining.json, mỗi batch gọi 1 request,
ghi kết quả ra data/synthetic/gen/out_NNN.jsonl. RESUME được (bỏ batch đã có out),
tôn trọng rate limit (mặc định 20 req/phút cho free tier).

Cách dùng (ví dụ OpenRouter free, Qwen3 Coder):
  export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
  export OPENAI_API_KEY="sk-or-..."
  export GEN_MODEL="qwen/qwen3-coder:free"
  python scripts/gen_via_api.py
DeepSeek: BASE=https://api.deepseek.com  MODEL=deepseek-chat
Qwen/DashScope: BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1  MODEL=qwen-plus
Kimi: BASE=https://api.moonshot.cn/v1  MODEL=moonshot-v1-8k

Cần: pip install openai
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
MODEL = os.environ.get("GEN_MODEL", "qwen/qwen3-coder:free")
RPM = float(os.environ.get("GEN_RPM", "20"))          # request/phút (free tier ~20)
N_SENT = int(os.environ.get("GEN_SENT", "6"))          # câu/term
MIN_GAP = 60.0 / RPM

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)

PROMPT = """Bạn sinh DATA HUẤN LUYỆN dịch máy Việt–Nhật. Dưới đây là danh sách thuật ngữ (JSON, mỗi mục {{ja, vi, en}}):

{terms}

Với MỖI thuật ngữ, viết {n} cặp câu (câu tiếng Nhật tự nhiên + bản dịch tiếng Việt trung thực) DÙNG thuật ngữ đó đúng nghĩa trong câu.
Yêu cầu: JA bản ngữ tự nhiên; VI khớp nghĩa CẢ câu; đa dạng trần thuật/hỏi/nhờ lịch sự (〜てください/〜ていただけますか) + lịch sự lẫn khẩu ngữ; ngữ cảnh IT/văn phòng/đời thường; giữ katakana nếu term là katakana; câu 5-25 từ, không lặp khuôn.

CHỈ trả về JSONL, MỖI DÒNG một object {{"ja":"...","vi":"...","term":"<từ ja>"}}. Không giải thích, không markdown, không bọc ```."""


def gen_batch(terms):
    r = client.chat.completions.create(
        model=MODEL, temperature=0.7,
        max_tokens=int(os.environ.get("GEN_MAXTOK", "16000")),  # tránh cắt cụt 180 cặp/batch
        messages=[{"role": "user", "content": PROMPT.format(terms=json.dumps(terms, ensure_ascii=False), n=N_SENT)}],
    )
    txt = r.choices[0].message.content or ""
    txt = re.sub(r"^```[a-z]*\n?|```$", "", txt.strip(), flags=re.M)  # bỏ rào code nếu có
    out = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            p = json.loads(line)
            if p.get("ja") and p.get("vi"):
                out.append({"ja": p["ja"], "vi": p["vi"], "term": p.get("term")})
        except json.JSONDecodeError:
            pass
    return out


def main():
    todo = json.loads((GEN / os.environ.get("GEN_TODO", "todo_remaining.json")).read_text())
    todo = [i for i in todo if not (GEN / f"out_{i:03d}.jsonl").exists()]  # resume
    print(f"model={MODEL} | còn {len(todo)} batch (đã bỏ batch có out)")
    last = 0.0
    done = 0
    for i in todo:
        gap = MIN_GAP - (time.time() - last)
        if gap > 0:
            time.sleep(gap)
        last = time.time()
        terms = json.loads((GEN / f"batch_{i:03d}.json").read_text())
        try:
            pairs = gen_batch(terms)
        except Exception as e:  # noqa: BLE001 - log rồi đi tiếp
            print(f"  batch {i}: LỖI {str(e)[:80]} — bỏ qua, chạy lại sau")
            continue
        if not pairs:
            print(f"  batch {i}: 0 cặp (parse fail?) — bỏ qua")
            continue
        (GEN / f"out_{i:03d}.jsonl").write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        done += 1
        print(f"  batch {i}: +{len(pairs)} cặp -> out_{i:03d}.jsonl  ({done}/{len(todo)})")
    print(f"XONG {done} batch. Chạy: python scripts/merge_glossary_sents.py để gộp.")


if __name__ == "__main__":
    main()
