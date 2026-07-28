#!/usr/bin/env python3
"""
Benchmark Live API: đo throughput & latency theo số worker song song trên 1 key.

Trả lời câu hỏi: tăng worker thì tốc độ tổng có tăng theo không,
hay server chỉ trả response đều (throttle) khiến thêm worker vô nghĩa?

Cách dùng:
    python3 scripts/bench_live_workers.py            # mặc định test 1, 5, 10, 20 worker
    python3 scripts/bench_live_workers.py 5 20 40    # tự chọn các mức worker
"""

import asyncio
import statistics
import sys
import time

from dotenv import dotenv_values
from google import genai
from google.genai import types

MODEL_ID = "gemini-3.1-flash-live-preview"
INPUT_FILE = "data/full_11.88m_ja_clean.txt"
RECV_TIMEOUT = 45

WORKER_COUNTS = [int(x) for x in sys.argv[1:]] or [1, 5, 10, 20]

env = dotenv_values()
KEY = None
for k in sorted(env.keys()):
    if k.lower().startswith("gemini"):
        v = (env[k] or "").strip()
        if len(v) >= 20:
            KEY = v
            break
if not KEY:
    print("❌ Không thấy gemini key trong .env")
    sys.exit(1)

SYSTEM_PROMPT = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu."
    "Hãy dịch câu tiếng Nhật sau sang tiếng Việt cực kỳ tự nhiên, chính xác và thoát ý."
    "Chỉ trả về duy nhất bản dịch tiếng Việt, không kèm lời giải thích hay chú thích."
)


async def bench_worker(queue: asyncio.Queue, latencies: list, errors: list):
    client = genai.Client(api_key=KEY)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
    )
    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            while True:
                try:
                    ja = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                t0 = time.perf_counter()
                try:
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=f"Dịch sang tiếng Việt: {ja}")]),
                        turn_complete=True,
                    )
                    texts = []

                    async def _recv():
                        async for r in session.receive():
                            sc = r.server_content
                            if sc:
                                ot = getattr(sc, "output_transcription", None)
                                if ot and getattr(ot, "text", ""):
                                    texts.append(ot.text)
                                if sc.turn_complete:
                                    return

                    await asyncio.wait_for(_recv(), timeout=RECV_TIMEOUT)
                    if "".join(texts).strip():
                        latencies.append(time.perf_counter() - t0)
                    else:
                        errors.append("empty")
                except Exception as e:
                    # Bench ngắn nên không reconnect — ghi lỗi rồi dừng worker
                    errors.append(str(e)[:70])
                    return
    except Exception as e:
        errors.append("CONNECT: " + str(e)[:80])


async def run_config(n_workers: int, sentences: list):
    queue = asyncio.Queue()
    for s in sentences:
        queue.put_nowait(s)
    latencies, errors = [], []
    t0 = time.perf_counter()
    await asyncio.gather(*(bench_worker(queue, latencies, errors) for _ in range(n_workers)))
    return time.perf_counter() - t0, latencies, errors


async def main():
    # Lấy pool câu ngắn vừa phải (10–60 ký tự) để các config so sánh được với nhau
    pool = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if 10 <= len(s) <= 60:
                pool.append(s)
            if len(pool) >= 2000:
                break

    print(f"🔬 Benchmark 1 key ({KEY[:8]}...), model {MODEL_ID}")
    print(f"{'worker':>6} | {'câu OK':>6} | {'wall(s)':>7} | {'câu/s':>6} | {'s/câu hiệu dụng':>15} | {'latency TB':>10} | {'p50':>5} | lỗi")
    print("-" * 95)

    offset = 0
    for n in WORKER_COUNTS:
        cnt = min(240, max(15, 12 * n))
        sentences = pool[offset:offset + cnt]
        offset += cnt
        wall, lat, errs = await run_config(n, sentences)
        done = len(lat)
        thr = done / wall if wall > 0 and done else 0.0
        eff = (1 / thr) if thr else float("inf")
        avg = statistics.mean(lat) if lat else 0.0
        p50 = statistics.median(lat) if lat else 0.0
        print(f"{n:>6} | {done:>6} | {wall:>7.1f} | {thr:>6.2f} | {eff:>15.3f} | {avg:>10.2f} | {p50:>5.2f} | {len(errs)}")
        if errs:
            print(f"       ↳ ví dụ lỗi: {errs[:3]}")
        await asyncio.sleep(2)

    print("\nĐọc kết quả: nếu 'câu/s' tăng ~tuyến tính theo worker và 'latency TB' giữ nguyên")
    print("→ tăng worker CÓ ý nghĩa. Nếu latency phình theo số worker còn câu/s đứng yên")
    print("→ server đang throttle tổng tốc độ, thêm worker vô nghĩa.")


if __name__ == "__main__":
    asyncio.run(main())
