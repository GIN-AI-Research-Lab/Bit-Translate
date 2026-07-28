#!/usr/bin/env python3
"""
Dich KD gom N cau/request (mac dinh 2) — tang toc ~1.7x so voi 1 cau/request.

AN TOAN LA UU TIEN SO 1. Da kiem chung 180 batch tren dat that (co ground-truth)
= 0 nhiem cheo (scripts/verify_batch_alignment.py). Ngoai ra co 4 lop bao ve +
fail an toan: bat ky nghi ngo nao -> cau do quay ve dich 1-cau (duong da chung minh
dung), roi LaBSE gac cong cuoi o buoc loc. => Gom cau chi CO THE nhanh hon, khong
the lam hong data.

Cach dung (giong het script cu, ghi CUNG file output, resume tuong thich):
    WORKERS_PER_KEY=20 BATCH_SIZE=2 python3 scripts/run_kd_batch.py
    BATCH_SIZE=1 python3 scripts/run_kd_batch.py     # = hanh vi script cu

4 lop chong lech hang:
  1. ID NGAU NHIEN 4 chu so [xxxx] — marker 3-5 chu so gan nhu khong xuat hien
     tu nhien trong tieng Viet => cau nguon chua [1]/[2] khong pha duoc parse.
  2. KHOP DUNG TAP ID — tra ve phai dung tap id da gui (khong thieu/thua/lap).
  3. SANITY tung cap — lot chu Nhat, hoac ty le do dai bat thuong -> loai.
  4. FAIL AN TOAN — batch/cap nao khong dat -> tung cau quay ve hang doi dich 1-cau.
"""
import asyncio
import os
import random
import re
import sys
import json
import time

from dotenv import load_dotenv, dotenv_values
load_dotenv()

from google import genai
from google.genai import types

# ---- Keys tu .env ----
# KD_SKIP_KEYS: bo qua key da chet MA KHONG phai sua .env (file credential cua user).
# Key chet tra ve 1008 "The bound service account is deleted or disabled" — luong gan
# vao no chi quay vong retry, an mat cong suat. Do thuc te 2026-07-26: 3/10 key chet
# -> 24/80 luong lang phi. Kiem tra key bang scripts/check_gemini_keys.py.
#   KD_SKIP_KEYS=gemini_key_6,gemini_key_9,gemini_key_10 python -u scripts/run_kd_batch.py
SKIP = {s.strip().lower() for s in os.getenv("KD_SKIP_KEYS", "").split(",") if s.strip()}
KEYS = []
env_file = dotenv_values()
for k in sorted(env_file.keys()):
    if k.lower().startswith("gemini"):
        if k.lower() in SKIP:
            print(f"   • BỎ QUA {k} (KD_SKIP_KEYS)")
            continue
        val = (env_file[k] or "").strip()
        if val and val not in KEYS and len(val) >= 20:
            KEYS.append(val)
            print(f"   • Dùng key từ .env: {k} ({val[:8]}...)")
if not KEYS:
    print("❌ Không tìm thấy Gemini API Key nào trong .env!"); sys.exit(1)

WORKERS_PER_KEY = max(1, int(os.getenv("WORKERS_PER_KEY", "1")))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "2")))

print(f"🔑 {len(KEYS)} key × {WORKERS_PER_KEY} session = {len(KEYS) * WORKERS_PER_KEY} luồng "
      f"| gom {BATCH_SIZE} câu/request")

MODEL_ID = "gemini-3.1-flash-live-preview"
# Doi qua env de dich nguon MOI (vong 4) ma khong dung toi corpus cu:
#   KD_INPUT=D:/Bit-Translate-data/raw/os_ja_new.txt \
#   KD_OUTPUT=D:/Bit-Translate-data/raw/kd_os_new.jsonl \
#   WORKERS_PER_KEY=6 BATCH_SIZE=1 python3 scripts/run_kd_batch.py
# Resume van hoat dong: script doc lai OUTPUT de bo qua id da dich.
INPUT_FILE = os.getenv("KD_INPUT", "data/full_11.88m_ja_clean.txt")
OUTPUT_FILE = os.getenv("KD_OUTPUT", "data/synthetic/kd_clean/kd_gemini3_final.jsonl")

RECV_TIMEOUT = 60
MAX_ATTEMPTS = 5
MAX_CONCURRENT_HANDSHAKE = int(os.getenv("MAX_HANDSHAKE", "12"))  # so bat tay dong thoi toi da
CONN_GATE = None  # asyncio.Semaphore, khoi tao trong main() (phai o trong event loop)
JP_RE = re.compile(r'[぀-ヿ一-鿿]')            # hiragana/katakana/kanji lot vao ban dich VI
MARKER_RE = re.compile(r'\[([0-9]{3,5})\]\s*(.*?)(?=\[[0-9]{3,5}\]|$)', re.S)

os.makedirs("data/synthetic/kd_clean", exist_ok=True)

SYSTEM_PROMPT_SINGLE = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu. "
    "Hãy dịch câu tiếng Nhật sau sang tiếng Việt cực kỳ tự nhiên, chính xác và thoát ý. "
    "Quy tắc về từ nước ngoài: GIỮ NGUYÊN không dịch các tên riêng, tên thương hiệu, tên sản phẩm, "
    "và thuật ngữ tiếng Anh (kể cả khi viết bằng katakana) nếu người Việt thường dùng nguyên dạng tiếng Anh "
    "(ví dụ: video, API, backend, marketing, download). "
    "Chỉ dịch từ katakana sang tiếng Việt khi có từ tiếng Việt thông dụng tương đương (ví dụ: テーブル → bàn). "
    "Chỉ trả về duy nhất bản dịch tiếng Việt, không kèm lời giải thích hay chú thích."
)
SYSTEM_PROMPT_BATCH = (
    "Bạn là một dịch giả tiếng Nhật sang tiếng Việt hàng đầu. "
    "Bạn sẽ nhận NHIỀU câu, mỗi câu có mã số dạng [mã] ở đầu. "
    "Hãy dịch TỪNG câu sang tiếng Việt cực kỳ tự nhiên, chính xác và thoát ý, rồi trả về "
    "mỗi câu trên MỘT dòng đúng định dạng: [mã] bản dịch. "
    "GIỮ ĐÚNG các mã số đã cho, đúng số lượng dòng bằng số câu nhận được, "
    "KHÔNG gộp, KHÔNG tách, KHÔNG thêm chú thích. "
    "Quy tắc về từ nước ngoài: GIỮ NGUYÊN tên riêng, thương hiệu, sản phẩm, thuật ngữ tiếng Anh "
    "(kể cả viết bằng katakana) nếu người Việt thường dùng nguyên dạng (video, API, backend...). "
    "Chỉ dịch katakana khi có từ tiếng Việt thông dụng tương đương (テーブル → bàn)."
)


def sanity_ok(ja, vi):
    """Lop 3: loai cap dang ngo (lot chu Nhat / ty le do dai bat thuong)."""
    if not vi or len(vi) < 2:
        return False
    if JP_RE.search(vi):
        return False
    # ty le do dai long: VI thuong 0.5x–4x do dai JA; ra ngoai la nghi lech/gop
    if len(vi) > 6 * len(ja) + 20 or len(vi) * 6 + 20 < len(ja):
        return False
    return True


async def _recv_text(session):
    texts = []
    async def _r():
        async for response in session.receive():
            sc = response.server_content
            if sc:
                ot = getattr(sc, "output_transcription", None)
                if ot and getattr(ot, "text", ""):
                    texts.append(ot.text)
                if sc.turn_complete:
                    return
    await asyncio.wait_for(_r(), timeout=RECV_TIMEOUT)
    return "".join(texts).strip()


def new_ids(n):
    """n mã 4 chữ số duy nhất, cách xa nhau."""
    return [str(x) for x in random.sample(range(1000, 9999), n)]


async def write_ok(idx, ja, vi, out_f, lock, stats):
    record = {"id": idx, "ja": ja, "vi": vi}
    async with lock:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        stats["success"] += 1
        if stats["success"] % 100 == 0:
            print(f"✅ Đã dịch {stats['success']:,d} câu...")


async def requeue_single(items, queue):
    """Fail an toan: day tung cau ve hang doi, ep dich 1-cau (force_single=True)."""
    for (idx, ja, attempts, _fs) in items:
        if attempts + 1 < MAX_ATTEMPTS:
            await queue.put((idx, ja, attempts + 1, True))
        else:
            print(f"🚫 Câu {idx} bỏ qua sau {MAX_ATTEMPTS} lần thử.")


async def worker(worker_id, api_key, queue, out_f, lock, stats):
    # So le nhe (cong CONN_GATE moi la thu chinh chong bao handshake)
    await asyncio.sleep((worker_id - 1) * 0.05)
    client = genai.Client(api_key=api_key)
    cfg_single = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT_SINGLE)]))
    cfg_batch = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT_BATCH)]))

    # Mo ket noi CO CONG: chi cho <= MAX_CONCURRENT_HANDSHAKE bat tay dong thoi
    # (chong "bao handshake" khi 120+ luong cung mo mot luc -> timeout hang loat).
    # Cong duoc GIU trong luc bat tay, NHA NGAY sau khi ket noi thanh cong -> moi
    # session van chay song song binh thuong, chi rai deu giai doan bat tay.
    # cm duoc GIU tham chieu de khong bi GC dong session (bug da gap truoc do).
    async def gated_connect(cfg):
        await CONN_GATE.acquire()
        try:
            cm = client.aio.live.connect(model=MODEL_ID, config=cfg)
            sess = await cm.__aenter__()
        except BaseException:
            CONN_GATE.release()
            raise
        CONN_GATE.release()
        return cm, sess

    # Duong single-fallback: mo 1 session GON theo tung lan can (hiem, chi khi batch
    # lech/sanity fail hoac con le 1 cau) — moi worker mac dinh chi giu 1 session batch.
    async def translate_single(ja):
        cm, ss = await gated_connect(cfg_single)
        try:
            await ss.send_client_content(
                turns=types.Content(role="user",
                    parts=[types.Part(text=f"Dịch sang tiếng Việt: {ja}")]),
                turn_complete=True)
            return await _recv_text(ss)
        finally:
            await cm.__aexit__(None, None, None)

    while not queue.empty():
        try:
            cm_batch, s_batch = await gated_connect(cfg_batch)
            try:
                while not queue.empty():
                    # Gom toi da BATCH_SIZE item KHONG force_single vao 1 request.
                    first = await queue.get()
                    batch = [first]
                    if BATCH_SIZE > 1 and not first[3]:
                        while len(batch) < BATCH_SIZE:
                            try:
                                nxt = queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if nxt[3]:            # force_single -> tra lai, khong gom
                                await queue.put(nxt)
                                break
                            batch.append(nxt)

                    single_mode = (len(batch) == 1) or BATCH_SIZE == 1 or batch[0][3]

                    try:
                        if single_mode:
                            # ----- Duong 1-cau (da chung minh dung) -----
                            for item in batch:      # thuong chi 1 item
                                idx, ja, attempts, _fs = item
                                vi = await translate_single(ja)
                                if sanity_ok(ja, vi):
                                    await write_ok(idx, ja, vi, out_f, lock, stats)
                                    queue.task_done()
                                else:
                                    if attempts + 1 < MAX_ATTEMPTS:
                                        await queue.put((idx, ja, attempts + 1, True))
                                    else:
                                        print(f"🚫 Câu {idx} bỏ qua sau {MAX_ATTEMPTS} lần thử.")
                                    queue.task_done()
                                await asyncio.sleep(0.01)
                        else:
                            # ----- Duong gom cau -----
                            ids = new_ids(len(batch))
                            id2item = {ids[i]: batch[i] for i in range(len(batch))}
                            lines = "\n".join(f"[{ids[i]}] {batch[i][1]}" for i in range(len(batch)))
                            await s_batch.send_client_content(
                                turns=types.Content(role="user",
                                    parts=[types.Part(text="Dịch các câu sau sang tiếng Việt:\n" + lines)]),
                                turn_complete=True)
                            out = await _recv_text(s_batch)

                            # Lop 1+2: parse marker 3-5 chu so, khop DUNG tap id
                            got = {}
                            for m in MARKER_RE.finditer(out):
                                i, v = m.group(1), m.group(2).strip()
                                if i in id2item and v and i not in got:
                                    got[i] = v

                            if set(got.keys()) != set(id2item.keys()):
                                # Lech hang -> vut CA batch, dich lai tung cau (fail an toan)
                                await requeue_single(batch, queue)
                                for _ in batch:
                                    queue.task_done()
                            else:
                                # Lop 3: sanity tung cap; cap nao ngo -> ve dich 1-cau
                                bad = []
                                for i, item in id2item.items():
                                    idx, ja, attempts, _fs = item
                                    vi = got[i]
                                    if sanity_ok(ja, vi):
                                        await write_ok(idx, ja, vi, out_f, lock, stats)
                                        queue.task_done()
                                    else:
                                        bad.append(item)
                                if bad:
                                    await requeue_single(bad, queue)
                                    for _ in bad:
                                        queue.task_done()
                            await asyncio.sleep(0.01)

                    except Exception as e:
                        # Loi giua chung -> tra ca batch ve hang doi (giu nguyen force_single flag), reconnect
                        for (idx, ja, attempts, fs) in batch:
                            if attempts + 1 < MAX_ATTEMPTS:
                                await queue.put((idx, ja, attempts + 1, fs))
                            else:
                                print(f"🚫 Câu {idx} bỏ qua sau {MAX_ATTEMPTS} lần thử. ({e})")
                            queue.task_done()
                        print(f"⚠️ [Luồng {worker_id}] reconnecting... ({str(e)[:80]})")
                        break
            finally:
                try:
                    await cm_batch.__aexit__(None, None, None)
                except Exception:
                    pass

        except Exception as conn_err:
            delay = 5 + random.uniform(0, 5)
            print(f"❌ [Luồng {worker_id}] Lỗi kết nối API: {str(conn_err)[:90]}. Thử lại sau {delay:.1f}s...")
            await asyncio.sleep(delay)


async def main():
    global CONN_GATE
    CONN_GATE = asyncio.Semaphore(MAX_CONCURRENT_HANDSHAKE)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Chưa thấy file dữ liệu {INPUT_FILE}."); return

    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except:
                    pass
    print(f"📑 Đã hoàn thành từ trước: {len(done_ids):,d} câu.")

    print(f"📖 Đang nạp danh sách câu từ {INPUT_FILE}...")
    queue = asyncio.Queue()
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx not in done_ids:
                queue.put_nowait((idx, line.strip(), 0, False))  # (id, ja, attempts, force_single)

    total_todo = queue.qsize()
    n_workers = len(KEYS) * WORKERS_PER_KEY
    print(f"🎯 Cần dịch tiếp: {total_todo:,d} câu với {n_workers} luồng, gom {BATCH_SIZE} câu/request.")

    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")
    lock = asyncio.Lock()
    stats = {"success": 0}

    tasks = []
    wid = 0
    for key in KEYS:
        for _ in range(WORKERS_PER_KEY):
            wid += 1
            tasks.append(asyncio.create_task(worker(wid, key, queue, out_f, lock, stats)))

    await queue.join()
    for t in tasks:
        t.cancel()
    out_f.close()
    print(f"\n🎉 HOÀN THÀNH! Đã dịch {stats['success']:,d} câu.")


if __name__ == "__main__":
    asyncio.run(main())
