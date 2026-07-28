#!/usr/bin/env python3
"""
KIEM CHUNG CAN CHINH khi gom 2 cau/request tren dat THAT (co ground-truth).

Muc dich: chung minh gom cau KHONG lam lech hang, truoc khi dua vao production.

Phuong phap:
  1. Lay N cap cau THAT tu corpus.
  2. Dich TUNG cau rieng le  -> ground truth (a, b).
  3. Dich gom 2 cau/request  -> (x, y) sau khi parse theo ID.
  4. Kiem tra NHIEM CHEO bang do giong (difflib):
       can chinh dung  <=> x giong a hon giong b  VA  y giong b hon giong a.
       neu batch bi hoan doi -> x se giong b hon -> BAT DUOC.
  5. Thu 2 kieu ID de chon kieu dang tin nhat qua duong audio:
       - simple : [1] [2]
       - random : [4-digit ngau nhien]  (chong model tu che thu tu + tranh trung [so] trong cau)

Cach dung:
    python3 scripts/verify_batch_alignment.py [N_BATCH]   (mac dinh 30)
"""
import asyncio
import difflib
import re
import sys

from dotenv import dotenv_values
from google import genai
from google.genai import types

MODEL_ID = "gemini-3.1-flash-live-preview"
INPUT_FILE = "data/full_11.88m_ja_clean.txt"
RECV_TIMEOUT = 60
N_BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 30

env = dotenv_values()
KEY = next((env[k].strip() for k in sorted(env)
            if k.lower().startswith("gemini") and len((env[k] or "").strip()) >= 20), None)
if not KEY:
    print("Khong thay gemini key trong .env"); sys.exit(1)

SYS_SINGLE = ("Ban la dich gia tieng Nhat sang tieng Viet. Dich that tu nhien, "
              "chi tra ve ban dich, khong giai thich.")
SYS_BATCH = ("Ban la dich gia tieng Nhat sang tieng Viet. Ban se nhan nhieu cau, "
             "moi cau co ma so dang [ma]. Dich TUNG cau, tra ve moi dong dang: "
             "[ma] ban dich. Giu DUNG cac ma so da cho, dung so luong dong = so cau, "
             "khong gop khong tach khong giai thich.")

JP = re.compile(r'[぀-ヿ一-鿿]')


def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


async def translate_single(session, ja):
    prompt = f"Dich sang tieng Viet: {ja}"
    await session.send_client_content(
        turns=types.Content(role="user", parts=[types.Part(text=prompt)]), turn_complete=True)
    texts = []
    async def recv():
        async for r in session.receive():
            sc = r.server_content
            if sc:
                ot = getattr(sc, "output_transcription", None)
                if ot and getattr(ot, "text", ""): texts.append(ot.text)
                if sc.turn_complete: return
    await asyncio.wait_for(recv(), timeout=RECV_TIMEOUT)
    return "".join(texts).strip()


async def translate_batch(session, pairs_ids):
    """pairs_ids: list of (id_str, ja). Tra ve dict {id_str: vi} hoac None neu lech."""
    lines = "\n".join(f"[{i}] {ja}" for i, ja in pairs_ids)
    prompt = "Dich cac cau sau sang tieng Viet:\n" + lines
    await session.send_client_content(
        turns=types.Content(role="user", parts=[types.Part(text=prompt)]), turn_complete=True)
    texts = []
    async def recv():
        async for r in session.receive():
            sc = r.server_content
            if sc:
                ot = getattr(sc, "output_transcription", None)
                if ot and getattr(ot, "text", ""): texts.append(ot.text)
                if sc.turn_complete: return
    await asyncio.wait_for(recv(), timeout=RECV_TIMEOUT)
    out = "".join(texts).strip()
    want_ids = [i for i, _ in pairs_ids]
    got = {}
    for m in re.finditer(r'\[([0-9]+)\]\s*(.*?)(?=\[[0-9]+\]|$)', out, flags=re.S):
        i, v = m.group(1), m.group(2).strip()
        if i in want_ids and v and i not in got:
            got[i] = v
    # PHAI khop DUNG tap id (khong thieu khong thua), moi ban dich khong rong
    if set(got.keys()) != set(want_ids):
        return None, out
    return got, out


async def run_scheme(scheme, sentences):
    client = genai.Client(api_key=KEY)
    def cfg(sysp):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(parts=[types.Part(text=sysp)]))

    parse_fail = 0
    swap = 0            # nhiem cheo: ban dich gan nham cau
    jp_leak = 0
    checked = 0
    rid = 1000

    async with client.aio.live.connect(model=MODEL_ID, config=cfg(SYS_SINGLE)) as s_single, \
               client.aio.live.connect(model=MODEL_ID, config=cfg(SYS_BATCH)) as s_batch:
        for b in range(N_BATCH):
            jaA, jaB = sentences[2 * b], sentences[2 * b + 1]
            # ground truth: dich rieng tung cau
            try:
                a = await translate_single(s_single, jaA)
                bt = await translate_single(s_single, jaB)
            except Exception:
                continue
            if not a or not bt:
                continue

            if scheme == "simple":
                idA, idB = "1", "2"
            else:
                idA, idB = str(rid), str(rid + 37); rid += 100

            try:
                got, raw = await translate_batch(s_batch, [(idA, jaA), (idB, jaB)])
            except Exception:
                parse_fail += 1
                continue
            if got is None:
                parse_fail += 1
                continue

            x, y = got[idA], got[idB]
            checked += 1
            if JP.search(x) or JP.search(y):
                jp_leak += 1
            # can chinh dung neu x giong a hon giong b, va y giong b hon giong a
            aligned = (ratio(x, a) >= ratio(x, bt)) and (ratio(y, bt) >= ratio(y, a))
            if not aligned:
                swap += 1
                if swap <= 3:
                    print(f"    [SWAP scheme={scheme}] A={jaA[:25]} | single_a={a[:25]} | batch_x={x[:25]}")
                    print(f"                             B={jaB[:25]} | single_b={bt[:25]} | batch_y={y[:25]}")

    return dict(scheme=scheme, checked=checked, parse_fail=parse_fail, swap=swap, jp_leak=jp_leak)


async def main():
    pool = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if 12 <= len(s) <= 80:
                pool.append(s)
            if len(pool) >= 4 * N_BATCH + 20:
                break

    print(f"Kiem chung can chinh gom-2-cau tren {N_BATCH} batch/kieu, model {MODEL_ID}")
    print("(dich rieng = chuan vang; so nhiem cheo bang do giong difflib)\n")
    r1 = await run_scheme("simple", pool[:2 * N_BATCH])
    r2 = await run_scheme("random", pool[2 * N_BATCH:4 * N_BATCH])

    print(f"\n{'kieu ID':>8} | {'kiem tra':>8} | {'parse fail':>10} | {'NHIEM CHEO':>10} | {'lot chu Nhat':>12}")
    print("-" * 62)
    for r in (r1, r2):
        print(f"{r['scheme']:>8} | {r['checked']:>8} | {r['parse_fail']:>10} | "
              f"{r['swap']:>10} | {r['jp_leak']:>12}")
    print("\nKet luan: kieu nao NHIEM CHEO = 0 va parse fail thap la an toan de dung.")
    print("Neu ca hai deu co nhiem cheo > 0 -> gom cau qua audio KHONG an toan, giu 1 cau/request.")


if __name__ == "__main__":
    asyncio.run(main())
