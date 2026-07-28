#!/usr/bin/env python3
"""Kiểm tra từng gemini_key_* trong .env còn sống không (Live API).

Vì sao cần: key chết trả 1008 "The bound service account is deleted or disabled",
mà `run_kd_batch.py` vẫn gán đủ luồng cho nó rồi quay vòng retry vô ích. Đo thực tế
2026-07-26: 3/10 key chết = 24/80 luồng lãng phí.

⚠️ CHẠY KHI KHÔNG CÓ JOB KD NÀO ĐANG CHẠY. Nếu job chính đang chiếm kết nối thì key
sống cũng báo "timed out during opening handshake" — không phân biệt được chết/bận.
Chỉ mã 1008 mới là bằng chứng key chết thật.

  python scripts/check_gemini_keys.py
  # rồi: KD_SKIP_KEYS=gemini_key_6,gemini_key_9 python -u scripts/run_kd_batch.py
"""
import asyncio
import re
import sys
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
MODEL = "gemini-3.1-flash-live-preview"


def load():
    keys = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(gemini_key_\d+)\s*=\s*(\S+)", line)
        if m:
            keys[m.group(1)] = m.group(2)
    return keys


async def test(name, k, cfg):
    try:
        c = genai.Client(api_key=k)
        async with c.aio.live.connect(model=MODEL, config=cfg) as s:
            await s.send_client_content(turns=types.Content(
                role="user", parts=[types.Part(text="Trả lời đúng một chữ: OK")]))
            async for r in s.receive():
                if r.server_content and r.server_content.turn_complete:
                    break
        return name, "SỐNG", ""
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        st = "CHẾT" if "1008" in msg else "?bận/timeout"
        return name, st, msg[:70]


async def main():
    keys = load()
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig())
    res = await asyncio.gather(*[test(n, k, cfg) for n, k in keys.items()])
    dead = []
    for name, st, msg in sorted(res):
        print(f"  {name:16} {st:14} {msg}")
        if st == "CHẾT":
            dead.append(name)
    print(f"\n{sum(1 for _,s,_ in res if s=='SỐNG')}/{len(res)} key sống")
    if dead:
        print(f"KD_SKIP_KEYS={','.join(dead)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
