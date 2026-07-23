#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv
load_dotenv()
from google import genai

key = os.environ.get('gemini_key_1') or os.environ.get('gemini_key_2')
if not key:
    print("No key found")
    sys.exit(1)

client = genai.Client(api_key=key)

models = ['gemini-3-flash-preview', 'gemini-3.1-flash-live-preview', 'gemini-2.5-flash', 'gemini-flash-lite-latest']
for m in models:
    try:
        res = client.models.generate_content(model=m, contents='Dịch sang tiếng Việt ngắn gọn: 猫が好きです。')
        print(f"✅ REST {m}: {res.text.strip()}")
    except Exception as e:
        print(f"❌ REST {m}: {e}")
