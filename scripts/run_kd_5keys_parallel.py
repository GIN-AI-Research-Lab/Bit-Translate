#!/usr/bin/env python3
"""
Script dịch KD tự động với N Gemini API Keys chạy song song WebSocket.
Hỗ trợ Auto-Reconnect, lưu checkpoint tự động, resuming nếu đứt kết nối.

Cách dùng:
    python3 scripts/run_kd_5keys_parallel.py
"""

import asyncio
import os
import random
import sys
import json
import time
from glob import glob

from dotenv import load_dotenv, dotenv_values
load_dotenv()

from google import genai
from google.genai import types

# 1. Gom Gemini API Keys — CHỈ đọc từ file .env, tên biến dạng gemini_key_1, gemini_key_2...
# (Không quét toàn bộ os.environ nữa: biến hệ thống như GOOGLE_API_KEY cũ
#  từng bị bắt nhầm làm key → chạy thừa luồng với key rác.)
KEYS = []
env_file = dotenv_values()  # chỉ nội dung file .env
for k in sorted(env_file.keys()):
    if k.lower().startswith("gemini"):
        val = (env_file[k] or "").strip()
        if val and val not in KEYS and len(val) >= 20:
            KEYS.append(val)
            print(f"   • Dùng key từ .env: {k} ({val[:8]}...)")

if not KEYS:
    print("❌ Không tìm thấy Gemini API Key nào trong .env!")
    print(" Vui lòng tạo file .env với dạng:")
    print(" gemini_key_1=AIzaSy...")
    print(" gemini_key_2=AIzaSy...")
    sys.exit(1)

# Số session WebSocket song song trên MỖI key — Live API cho phép 1 key mở nhiều
# session, giới hạn thực tế là quota concurrent-session của project (không phải per key).
WORKERS_PER_KEY = max(1, int(os.getenv("WORKERS_PER_KEY", "1")))

print(f"🔑 Đã tìm thấy {len(KEYS)} Gemini API Key(s) khả dụng × {WORKERS_PER_KEY} session/key = {len(KEYS) * WORKERS_PER_KEY} luồng.")

MODEL_ID = "gemini-3.1-flash-live-preview"
INPUT_FILE = "data/full_11.88m_ja_clean.txt"
OUTPUT_FILE = "data/synthetic/kd_clean/kd_gemini3_final.jsonl"

RECV_TIMEOUT = 60   # giây — chống luồng treo vĩnh viễn khi kết nối chết "câm"
MAX_ATTEMPTS = 5    # số lần thử tối đa cho 1 câu trước khi bỏ qua hẳn

os.makedirs("data/synthetic/kd_clean", exist_ok=True)

SYSTEM_PROMPT = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu. "
    "Hãy dịch câu tiếng Nhật sau sang tiếng Việt cực kỳ tự nhiên, chính xác và thoát ý. "
    "Quy tắc về từ nước ngoài: GIỮ NGUYÊN không dịch các tên riêng, tên thương hiệu, tên sản phẩm, "
    "và thuật ngữ tiếng Anh (kể cả khi viết bằng katakana) nếu người Việt thường dùng nguyên dạng tiếng Anh "
    "(ví dụ: video, API, backend, marketing, download). "
    "Chỉ dịch từ katakana sang tiếng Việt khi có từ tiếng Việt thông dụng tương đương (ví dụ: テーブル → bàn). "
    "Chỉ trả về duy nhất bản dịch tiếng Việt, không kèm lời giải thích hay chú thích."
)

async def worker(worker_id: int, api_key: str, queue: asyncio.Queue, out_f, lock: asyncio.Lock, stats: dict):
    # Khởi động so le: tránh 60 luồng cùng mở handshake WebSocket một lúc
    # (gây "timed out during opening handshake" hàng loạt lúc start)
    await asyncio.sleep((worker_id - 1) * 0.5)

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
                    idx, ja_text, attempts = await queue.get()

                    try:
                        prompt = f"Dịch sang tiếng Việt: {ja_text}"
                        await session.send_client_content(
                            turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                            turn_complete=True
                        )

                        texts = []

                        async def _receive_turn():
                            async for response in session.receive():
                                sc = response.server_content
                                if sc:
                                    if hasattr(sc, "output_transcription") and sc.output_transcription:
                                        t = getattr(sc.output_transcription, "text", "")
                                        if t:
                                            texts.append(t)
                                    if sc.turn_complete:
                                        return

                        # Timeout chống kết nối chết "câm" làm treo luồng vô hạn
                        await asyncio.wait_for(_receive_turn(), timeout=RECV_TIMEOUT)

                        vi_text = "".join(texts).strip()

                        if vi_text and len(vi_text) >= 2:
                            record = {"id": idx, "ja": ja_text, "vi": vi_text}
                            async with lock:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                out_f.flush()
                                stats["success"] += 1
                                if stats["success"] % 100 == 0:
                                    print(f"✅ Đã dịch {stats['success']:,d} câu...")
                            queue.task_done()
                            await asyncio.sleep(0.01) # Tránh nghẽn rate limit 65k token
                        else:
                            # Session đóng êm (GoAway) hoặc model trả rỗng:
                            # đẩy lại queue thay vì nuốt mất câu, rồi reconnect
                            if attempts + 1 >= MAX_ATTEMPTS:
                                print(f"🚫 [Luồng {worker_id}] Câu {idx} rỗng sau {MAX_ATTEMPTS} lần thử — bỏ qua.")
                            else:
                                await queue.put((idx, ja_text, attempts + 1))
                            queue.task_done()
                            break

                    except Exception as e:
                        if attempts + 1 >= MAX_ATTEMPTS:
                            print(f"🚫 [Luồng {worker_id}] Câu {idx} lỗi sau {MAX_ATTEMPTS} lần thử — bỏ qua. ({e})")
                        else:
                            print(f"⚠️ [Luồng {worker_id}] Lỗi câu {idx}, reconnecting WebSocket... ({e})")
                            await queue.put((idx, ja_text, attempts + 1))
                        queue.task_done()
                        break

        except Exception as conn_err:
            # Jitter ngẫu nhiên để các luồng không cùng retry một thời điểm (bão reconnect)
            delay = 5 + random.uniform(0, 5)
            print(f"❌ [Luồng {worker_id}] Lỗi kết nối API: {conn_err}. Thử lại sau {delay:.1f}s...")
            await asyncio.sleep(delay)

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Chưa thấy file dữ liệu {INPUT_FILE}.")
        print(f" Vui lòng tải file từ Release về bằng lệnh: wget https://github.com/trituenguyen97/Bit-Translate/releases/download/raw-corpus-11.88m/full_11.88m_ja_clean.txt -P data/")
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
                queue.put_nowait((idx, line.strip(), 0))

    total_todo = queue.qsize()
    n_workers = len(KEYS) * WORKERS_PER_KEY
    print(f"🎯 Cần dịch tiếp: {total_todo:,d} câu với {n_workers} luồng song song.")

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    lock = asyncio.Lock()
    stats = {"success": 0}

    tasks = []
    wid = 0
    for key in KEYS:
        for _ in range(WORKERS_PER_KEY):
            wid += 1
            t = asyncio.create_task(worker(wid, key, queue, out_f, lock, stats))
            tasks.append(t)

    await queue.join()
    for t in tasks:
        t.cancel()

    out_f.close()
    print(f"\n🎉 HOÀN THÀNH DISTILLATION FULL DATASET! Đã dịch {stats['success']:,d} câu.")

if __name__ == "__main__":
    asyncio.run(main())
