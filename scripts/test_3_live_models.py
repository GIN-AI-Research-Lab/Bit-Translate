#!/usr/bin/env python3
import asyncio, os, time
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

API_KEY = os.environ.get('gemini_key_1') or os.environ.get('gemini_key_2')

MODELS_TO_TEST = [
    ("Gemini 2.5 Flash Native Audio Dialog", "gemini-2.5-flash-native-audio-latest"),
    ("Gemini 3 Flash Live", "gemini-3.1-flash-live-preview"),
    ("Gemini 3.5 Live Translate", "gemini-3.5-live-translate-preview")
]

TEST_SENTENCE = "猫が好きです。"

async def test_model(display_name, model_id):
    print(f"\n==================================================")
    print(f"🧪 Testing: {display_name} ({model_id})")
    print(f"==================================================")
    client = genai.Client(api_key=API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=['AUDIO'],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text='Bạn là dịch giả. Dịch câu tiếng Nhật sang tiếng Việt, chỉ trả về bản dịch ngắn gọn.')])
    )
    t0 = time.time()
    try:
        async with client.aio.live.connect(model=model_id, config=config) as session:
            await session.send_client_content(
                turns=types.Content(role='user', parts=[types.Part(text=f"Dịch sang tiếng Việt: {TEST_SENTENCE}")]),
                turn_complete=True
            )
            texts = []
            async for r in session.receive():
                sc = r.server_content
                if sc:
                    if hasattr(sc, 'output_transcription') and sc.output_transcription:
                        t = getattr(sc.output_transcription, 'text', '')
                        if t:
                            texts.append(t)
                    if sc.turn_complete:
                        break
            dt = time.time() - t0
            result = "".join(texts).strip()
            print(f"✅ HOẠT ĐỘNG THÀNH CÔNG!")
            print(f"   JA: {TEST_SENTENCE}")
            print(f"   VI: {result}")
            print(f"   ⏱ Thời gian: {dt:.2f}s")
            return True
    except Exception as e:
        print(f"❌ LỖI: {e}")
        return False

async def main():
    for name, mid in MODELS_TO_TEST:
        await test_model(name, mid)

if __name__ == '__main__':
    asyncio.run(main())
