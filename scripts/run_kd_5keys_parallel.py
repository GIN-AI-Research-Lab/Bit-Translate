#!/usr/bin/env python3
"""
Script dịch KD tự động với N Gemini API Keys chạy song song WebSocket.
Hỗ trợ Auto-Reconnect, lưu checkpoint tự động, resuming nếu đứt kết nối.

Cách dùng:
    python3 scripts/run_kd_5keys_parallel.py
"""

import asyncio
import os
import sys
import json
import time
from glob import glob

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# 1. Gom tất cả Gemini API Keys trong .env
KEYS = []
for k, v in os.environ.items():
    if k.lower().startswith("gemini_key_") or k == "GEMINI_API_KEY":
        if v.strip() and v.strip() not in KEYS:
            KEYS.append(v.strip())

if not KEYS:
    print("❌ Không tìm thấy Gemini API Key nào trong .env!")
    print(" Vui lòng tạo file .env với gemini_key_1=..., gemini_key_2=...")
    sys.exit(1)

print(f"🔑 Đã tìm thấy {len(KEYS)} Gemini API Key(s) khả dụng.")

MODEL_ID = "gemini-3.1-flash-live-preview"
INPUT_FILE = "data/full_11.88m_ja_clean.txt"
OUTPUT_FILE = "data/synthetic/kd_clean/kd_gemini3_final.jsonl"
CHECKPOINT_FILE = "data/synthetic/kd_clean/kd_checkpoint.json"

os.makedirs("data/synthetic/kd_clean", exist_ok=True)

SYSTEM_PROMPT = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu."
    "Hãy dịch câu tiếng Nhật sau sang tiếng Việt cực kỳ tự nhiên, chính xác và thoát ý."
    "Chỉ trả về duy nhất bản dịch tiếng Việt, không kèm lời giải thích hay chú thích."
)

async def worker(worker_id: int, api_key: str, queue: asyncio.Queue, out_f, lock: asyncio.Lock, stats: dict):
    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    )

    while not queue.empty():
        try:
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                while not queue.empty():
                    item = await queue.get()
                    idx, ja_text = item

                    t0 = time.time()
                    try:
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
                        
                        vi_text = "".join(texts).strip()

                        if vi_text and len(vi_text) >= 2:
                            record = {"id": idx, "ja": ja_text, "vi": vi_text}
                            async with lock:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                out_f.flush()
                                stats["success"] += 1
                        
                        queue.task_done()
                        await asyncio.sleep(0.01) # Tránh nghẽn rate limit 65k token

                    except Exception as e:
                        print(f"⚠️ [Luồng {worker_id}] Lỗi câu {idx}, reconnecting WebSocket... ({e})")
                        await queue.put(item) # Đưa lại câu bị đứt vào hàng đợi
                        queue.task_done()
                        break # Thoát session để reconnect WebSocket mới

        except Exception as conn_err:
            print(f"❌ [Luồng {worker_id}] Lỗi kết nối API: {conn_err}. Thử lại sau 5s...")
            await asyncio.sleep(5)

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Chưa có file dữ liệu {INPUT_FILE}. Đang chờ giải nén...")
        return

    # Nạp danh sách câu đã dịch (để resume)
    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(r["id"])
                except:
                    pass
    print(f"📑 Đã hoàn thành từ trước: {len(done_ids):,d} câu.")

    # Đọc input
    print(f"📖 Đang nạp danh sách câu từ {INPUT_FILE}...")
    queue = asyncio.Queue()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx not in done_ids:
                queue.put_nowait((idx, line.strip()))

    total_todo = queue.qsize()
    print(f"🎯 Cần dịch tiếp: {total_todo:,d} câu với {len(KEYS)} luồng song song.")

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    lock = asyncio.Lock()
    stats = {"success": 0}

    tasks = []
    for i, key in enumerate(KEYS, 1):
        t = asyncio.create_task(worker(i, key, queue, out_f, lock, stats))
        tasks.append(t)

    await queue.join()
    for t in tasks:
        t.cancel()

    out_f.close()
    print(f"\n🎉 HOÀN THÀNH DISTILLATION FULL DATASET! Đã dịch {stats['success']:,d} câu.")

if __name__ == "__main__":
    asyncio.run(main())
