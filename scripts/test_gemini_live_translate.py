#!/usr/bin/env python3
"""
Test Gemini 3.1 Flash Live Preview — Text Input, Audio Output + Transcript

Model này CHỈ hỗ trợ response_modalities=["AUDIO"], KHÔNG hỗ trợ ["TEXT"].
Nhưng bật output_audio_transcription=True sẽ trả về text transcript kèm audio.
Ta lấy transcript text làm bản dịch, bỏ qua audio data.

Rate Limit (Free): Unlimited RPM, 65K token/phút, Unlimited RPD → CỰC MẠNH!

Cách chạy:
    python3 scripts/test_gemini_live_translate.py
"""

import asyncio
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Lấy key từ .env
API_KEY = os.environ.get("gemini_key_1") or os.environ.get("gemini_key_2") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("❌ Không tìm thấy API key Gemini trong .env")
    sys.exit(1)

MODEL_ID = "gemini-3.1-flash-live-preview"

# Câu test tiếng Nhật
TEST_SENTENCES = [
    "猫が好きです。",
    "明日の会議は午後3時に変更になりました。",
    "このライブラリをインストールしてください。",
    "彼女は雨が降っているにもかかわらず、傘を持たずに出かけた。",
    "お忙しいところ恐れ入りますが、ご確認いただけますでしょうか。",
]

SYSTEM_PROMPT = (
    "あなたは日本語→ベトナム語の翻訳者です。"
    "入力された日本語の文を、自然で正確なベトナム語に翻訳してください。"
    "翻訳文のみを出力し、説明や注釈は一切付けないでください。"
)


async def translate_one(session, ja_text: str) -> str:
    """Gửi 1 câu JA và nhận transcript dịch VI."""
    prompt = f"翻訳してください: {ja_text}"
    await session.send_client_content(
        turns=types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        ),
        turn_complete=True
    )

    transcript_parts = []
    async for response in session.receive():
        sc = response.server_content
        if sc is not None:
            # Lấy text transcript (không phải audio bytes)
            if sc.model_turn is not None:
                for part in sc.model_turn.parts:
                    if hasattr(part, 'text') and part.text:
                        transcript_parts.append(part.text)
            # Kiểm tra output_transcription (nếu SDK trả riêng)
            if hasattr(sc, 'output_transcription') and sc.output_transcription:
                if hasattr(sc.output_transcription, 'text') and sc.output_transcription.text:
                    transcript_parts.append(sc.output_transcription.text)
            if sc.turn_complete:
                break

    return "".join(transcript_parts).strip()


async def main():
    client = genai.Client(api_key=API_KEY)

    # Config: AUDIO modality + bật transcript để nhận text
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)]
        ),
    )

    print(f"🔌 Đang kết nối Live API: {MODEL_ID}")
    print(f"   API Key: ...{API_KEY[-8:]}")
    print(f"   Mode: AUDIO + output_audio_transcription (lấy text từ transcript)")
    print()

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("✅ Kết nối WebSocket thành công!\n")

            # === TEST 1: Dịch từng câu ===
            print("=" * 60)
            print("TEST 1: DỊCH TỪNG CÂU (1 câu / 1 request)")
            print("=" * 60)

            t0 = time.time()
            results = []
            for i, ja in enumerate(TEST_SENTENCES, 1):
                t1 = time.time()
                vi = await translate_one(session, ja)
                dt = time.time() - t1
                results.append((ja, vi, dt))
                print(f"\n  [{i}] JA: {ja}")
                print(f"      VI: {vi}")
                print(f"      ⏱ {dt:.2f}s")

            total_1 = time.time() - t0
            print(f"\n  📊 Tổng: {total_1:.2f}s cho {len(TEST_SENTENCES)} câu")
            print(f"     TB: {total_1/len(TEST_SENTENCES):.2f}s / câu")

            # Kiểm tra xem có nhận được text không
            got_text = sum(1 for _, vi, _ in results if vi and len(vi) > 2)

            # === TEST 2: Stress test 10 request liên tục ===
            print()
            print("=" * 60)
            print("TEST 2: STRESS TEST (10 request liên tục)")
            print("=" * 60)

            stress_ja = "このシステムは本番環境にデプロイされています。"
            t2 = time.time()
            success = 0
            for i in range(10):
                try:
                    vi = await translate_one(session, stress_ja)
                    if vi and len(vi) > 2:
                        success += 1
                    print(f"  [{i+1:2d}/10] {'✅' if vi else '⚠️'} {vi[:60] if vi else '(trống)'}")
                except Exception as e:
                    print(f"  [{i+1:2d}/10] ❌ {e}")
            dt2 = time.time() - t2
            print(f"\n  📊 {success}/10 thành công, tổng {dt2:.2f}s, TB {dt2/10:.2f}s/req")

            # === KẾT LUẬN ===
            print()
            print("=" * 60)
            print("KẾT LUẬN")
            print("=" * 60)
            if got_text >= 3 and success >= 7:
                avg_time = dt2 / 10
                rpm = 60 / avg_time if avg_time > 0 else 0
                tok_per_min = 65_000
                avg_tok = 60  # ~30 in + ~30 out
                sents_per_min = min(rpm, tok_per_min / avg_tok)
                sents_per_day = sents_per_min * 60 * 24

                print(f"  ✅ LIVE API TEXT TRANSLATION: HOẠT ĐỘNG!")
                print(f"  ✅ Rate: Unlimited RPM, 65K token/phút")
                print(f"  📊 Ước tính sản lượng KD (FREE 100%):")
                print(f"     ~{sents_per_min:.0f} câu/phút")
                print(f"     ~{sents_per_min*60:.0f} câu/giờ")
                print(f"     ~{sents_per_day:,.0f} câu/ngày")
                print(f"  🎯 200k câu → {200_000/(sents_per_min*60):.1f} giờ")
                print(f"  🎯 11.5M câu → {11_500_000/sents_per_day:.1f} ngày")
            elif got_text == 0:
                print(f"  ⚠️ Transcript trống — model trả audio nhưng KHÔNG trả text.")
                print(f"     Cần thử cách khác (ví dụ: dùng gemini-2.0-flash-live-001)")
            else:
                print(f"  ⚠️ Hoạt động không ổn định ({got_text}/5 câu có text, {success}/10 stress)")

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
