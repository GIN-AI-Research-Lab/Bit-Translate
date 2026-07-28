#!/usr/bin/env python3
"""
Benchmark gom N cau/request tren Live API: do throughput + ty le LECH HANG.

So sanh 1 vs 2 vs 3 cau/request. Moi cau danh so [1]/[2]/[3] de gan lai;
dem so lan model tra ve SAI so dong (lech hang -> phai vut ca batch).

Cach dung:
    python3 scripts/bench_batch_size.py
    python3 scripts/bench_batch_size.py 1 2 3 5   # tuy chon batch sizes

Dung 1 KEY DAU trong .env (khong dung het de khong dun production).
"""

import asyncio
import re
import statistics
import sys
import time

from dotenv import dotenv_values
from google import genai
from google.genai import types

MODEL_ID = "gemini-3.1-flash-live-preview"
INPUT_FILE = "data/full_11.88m_ja_clean.txt"
RECV_TIMEOUT = 60
N_REQUESTS = 40          # so request moi cau hinh
BATCH_SIZES = [int(x) for x in sys.argv[1:]] or [1, 2, 3]

env = dotenv_values()
KEY = None
for k in sorted(env.keys()):
    if k.lower().startswith("gemini"):
        v = (env[k] or "").strip()
        if len(v) >= 20:
            KEY = v
            break
if not KEY:
    print("Khong thay gemini key trong .env"); sys.exit(1)

SYSTEM_PROMPT_SINGLE = (
    "Ban la dich gia tieng Nhat sang tieng Viet. Dich cau sau that tu nhien, "
    "chi tra ve ban dich, khong giai thich."
)
SYSTEM_PROMPT_BATCH = (
    "Ban la dich gia tieng Nhat sang tieng Viet. Ban se nhan nhieu cau, moi cau "
    "danh so dang [1] [2] [3]. Dich TUNG cau that tu nhien va tra ve dung dinh dang: "
    "moi dong bat dau bang so thu tu trong ngoac vuong roi ban dich, vi du:\n"
    "[1] ban dich cau 1\n[2] ban dich cau 2\n"
    "TUYET DOI giu dung so luong dong = so cau nhan duoc, khong gop, khong tach, khong giai thich."
)


def parse_batch(text, n):
    """Tach ban dich theo [i]. Tra ve list n phan tu hoac None neu lech hang."""
    parts = {}
    for m in re.finditer(r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|$)', text, flags=re.S):
        i = int(m.group(1)); v = m.group(2).strip()
        if 1 <= i <= n and v:
            parts[i] = v
    if len(parts) != n:
        return None
    return [parts[i] for i in range(1, n + 1)]


async def run_config(batch_size, sentences):
    client = genai.Client(api_key=KEY)
    is_batch = batch_size > 1
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(
            text=SYSTEM_PROMPT_BATCH if is_batch else SYSTEM_PROMPT_SINGLE)]),
    )
    latencies = []
    sent_cnt = 0        # tong cau da gui
    ok_cnt = 0          # tong cau lay duoc ban dich (khong lech)
    misalign = 0        # so batch bi lech hang
    errors = 0

    idx = 0
    t_start = time.perf_counter()
    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        for _ in range(N_REQUESTS):
            batch = sentences[idx:idx + batch_size]
            idx += batch_size
            if len(batch) < batch_size:
                break
            if is_batch:
                prompt = "Dich cac cau sau sang tieng Viet:\n" + \
                         "\n".join(f"[{i+1}] {s}" for i, s in enumerate(batch))
            else:
                prompt = f"Dich sang tieng Viet: {batch[0]}"

            t0 = time.perf_counter()
            try:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
                    turn_complete=True)
                texts = []

                async def recv():
                    async for r in session.receive():
                        sc = r.server_content
                        if sc:
                            ot = getattr(sc, "output_transcription", None)
                            if ot and getattr(ot, "text", ""):
                                texts.append(ot.text)
                            if sc.turn_complete:
                                return
                await asyncio.wait_for(recv(), timeout=RECV_TIMEOUT)
                out = "".join(texts).strip()
                sent_cnt += batch_size
                latencies.append(time.perf_counter() - t0)

                if is_batch:
                    parsed = parse_batch(out, batch_size)
                    if parsed is None:
                        misalign += 1          # ca batch bi vut
                    else:
                        ok_cnt += batch_size
                else:
                    if out:
                        ok_cnt += 1
            except Exception as e:
                errors += 1
    wall = time.perf_counter() - t_start
    return dict(batch=batch_size, sent=sent_cnt, ok=ok_cnt, misalign=misalign,
                errors=errors, wall=wall,
                lat=statistics.mean(latencies) if latencies else 0,
                thr=ok_cnt / wall if wall else 0)


async def main():
    pool = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if 10 <= len(s) <= 70:
                pool.append(s)
            if len(pool) >= 2000:
                break

    print(f"Benchmark GOM CAU tren 1 key ({KEY[:8]}...), model {MODEL_ID}")
    print(f"{N_REQUESTS} request moi cau hinh. Cau/request: {BATCH_SIZES}\n")
    print(f"{'cau/req':>7} | {'cau OK':>6} | {'lech hang':>9} | {'loi':>4} | "
          f"{'wall(s)':>7} | {'cau/s':>6} | {'latency/req':>11}")
    print("-" * 78)
    off = 0
    results = []
    for bs in BATCH_SIZES:
        need = N_REQUESTS * bs
        r = await run_config(bs, pool[off:off + need])
        off += need
        results.append(r)
        mis_pct = f"{r['misalign']}/{N_REQUESTS}" if bs > 1 else "-"
        print(f"{r['batch']:>7} | {r['ok']:>6} | {mis_pct:>9} | {r['errors']:>4} | "
              f"{r['wall']:>7.1f} | {r['thr']:>6.2f} | {r['lat']:>9.2f}s")
        await asyncio.sleep(2)

    base = next((x for x in results if x['batch'] == 1), None)
    print("\n--- Nhan xet ---")
    for r in results:
        if r['batch'] == 1 or not base or base['thr'] == 0:
            continue
        speedup = r['thr'] / base['thr']
        mis_rate = r['misalign'] / N_REQUESTS * 100
        print(f"  {r['batch']} cau/req: nhanh {speedup:.2f}x so voi 1 cau, "
              f"ty le lech hang {mis_rate:.1f}% (cau trong batch lech BI VUT het)")


if __name__ == "__main__":
    asyncio.run(main())
