#!/usr/bin/env python3
import asyncio, os, time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

API_KEY = os.environ.get('gemini_key_1') or os.environ.get('gemini_key_2')
MODEL_ID = 'gemini-3.1-flash-live-preview'

TEST_CASES = [
    ('IT', 'このコードは非同期処理でWebsocket通信を行っています。'),
    ('Keigo', '恐れ入りますが、至急ご確認のほどよろしくお願い申し上げます。'),
    ('Hội thoại', 'マジで？昨日言ってた話、本当だったんだ。'),
    ('Thành ngữ', '一石二鳥の対策を考えなければならない。'),
    ('Phủ định', '彼が犯人ではないとは言い切れない。')
]

async def translate_one(session, ja):
    prompt = f"Dịch sang tiếng Việt: {ja}"
    await session.send_client_content(
        turns=types.Content(role='user', parts=[types.Part(text=prompt)]),
        turn_complete=True
    )
    texts = []
    async for r in session.receive():
        sc = r.server_content
        if sc:
            # Check direct part text (if any)
            if sc.model_turn:
                for p in sc.model_turn.parts:
                    if hasattr(p, 'text') and p.text:
                        texts.append(p.text)
            # Check output_transcription (đúng chỗ chứa text khi dùng AUDIO mode)
            if hasattr(sc, 'output_transcription') and sc.output_transcription:
                t = getattr(sc.output_transcription, 'text', '')
                if t:
                    texts.append(t)
            if sc.turn_complete:
                break
    return "".join(texts).strip()

async def main():
    client = genai.Client(api_key=API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=['AUDIO'],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text='Bạn là dịch giả Việt-Nhật. Dịch chính xác câu tiếng Nhật sang tiếng Việt, chỉ trả về bản dịch ngắn gọn.')])
    )
    async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
        for cat, ja in TEST_CASES:
            t0 = time.time()
            vi = await translate_one(session, ja)
            print(f"[{cat}] {ja}")
            print(f" ➔ {vi} ({time.time()-t0:.2f}s)\n")

if __name__ == '__main__':
    asyncio.run(main())
