#!/usr/bin/env python3
"""
Test Gemini 3 Flash Live (gemini-3.1-flash-live-preview) trên 10 domain khó của hardbench200 (ja2vi).
In chi tiết bản dịch thu được và so sánh với Reference gold standard.
"""

import asyncio
import os
import sys
import json
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

API_KEY = os.environ.get("gemini_key_1") or os.environ.get("gemini_key_2") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("❌ Không tìm thấy API Key Gemini")
    sys.exit(1)

MODEL_ID = "gemini-3.1-flash-live-preview"

# Đọc sample 10 câu đại diện 10 domain ja2vi từ eval/hardbench200.jsonl
BENCH_FILE = "eval/hardbench200.jsonl"
samples_by_domain = {}

with open(BENCH_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("dir") == "ja2vi":
            dom = item["domain"]
            if dom not in samples_by_domain:
                samples_by_domain[dom] = item

SYSTEM_PROMPT = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu."
    "Hãy dịch câu tiếng Nhật sau sang tiếng Việt cực kỳ tự nhiên, chính xác, thoát ý và chuẩn văn phong tiếng Việt."
    "Chỉ trả về duy nhất bản dịch tiếng Việt, không kèm lời giải thích hay chú thích."
)

async def translate_sentence(session, ja_text: str) -> str:
    prompt = f"Dịch sang tiếng Việt: {ja_text}"
    await session.send_client_content(
        turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
        turn_complete=True
    )
    texts = []
    async for response in session.receive():
        sc = response.server_content
        if sc:
            if hasattr(sc, "output_transcription") and sc.output_transcription:
                t = getattr(sc.output_transcription, "text", "")
                if t:
                    texts.append(t)
            if sc.turn_complete:
                break
    return "".join(texts).strip()

async def main():
    client = genai.Client(api_key=API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    )

    print(f"🚀 TEST HARDBENCH 10 DOMAIN JA→VI VỚI MODEL: {MODEL_ID}\n" + "="*80)

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            count = 0
            t_total = time.time()
            for dom, item in samples_by_domain.items():
                count += 1
                ja = item["src"]
                ref = item["ref"]
                
                t0 = time.time()
                vi = await translate_sentence(session, ja)
                dt = time.time() - t0

                print(f"📌 [{count}/10] DOMAIN: [{dom.upper()}] (⏱ {dt:.2f}s)")
                print(f"   🇯🇵 JA (Gốc) : {ja}")
                print(f"   🇻🇳 Ref Gold  : {ref}")
                print(f"   ✨ Gemini3   : {vi}")
                print("-" * 80)

            total_time = time.time() - t_total
            print(f"\n📊 TỔNG KẾT: Đã dịch 10 domain khó trong {total_time:.2f} giây (Trung bình {total_time/10:.2f}s/câu)")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    asyncio.run(main())
