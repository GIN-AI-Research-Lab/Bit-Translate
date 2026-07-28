#!/usr/bin/env python3
"""Sinh câu cho QUÁN NGỮ THIẾU — theo DANH SÁCH ĐO ĐƯỢC, không sinh theo chủ đề.

Đo được (scripts tính trong logs/idiom_gap2.log): trong 4.651 quán ngữ của
`data/synthetic/gen/idiom_glosses.jsonl`, có **3.168 mục xuất hiện < 50 lần** trong
corpus v4 (12,5M câu) CỘNG phần chọn cho vòng 5 (3,35M câu) — và **2.098 mục xuất
hiện ĐÚNG 0 LẦN**. Trung vị chỉ 8 lần.

Vì sao thu thật không lấp được: biên bản Quốc hội quá trang trọng, còn nhóm khẩu ngữ
CC-100 chỉ chiếm 15% ngân sách token. Đây là lớp data DUY NHẤT mà sinh tổng hợp có
bằng chứng hiệu quả trong dự án này (vòng 2: 1.241 quán ngữ -> +0,8 chrF; vòng 3:
4.691 quán ngữ -> +1,7).

Ngưỡng 50: lấy từ đo đạc vòng 4 — thuật ngữ đạt 78 lần/corpus (~234 lần gặp qua 3
epoch) thì được vá, còn <=5 lần thì model dịch bậy.

  set PYTHONUTF8=1
  python scripts/gen_idiom_gap.py --per-item 20
  # -> D:/Bit-Translate-data/gen_niche/idiomgap.jsonl
"""
import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from gen_niche_kd import load_keys, parse_array, VI_TONE_STRICT, VI_WORD_IN_JA  # noqa: E402
from filter_kd_full import check, norm  # noqa: E402

GAP = Path("D:/Bit-Translate-data/idiom_gap.json")
OUT = Path("D:/Bit-Translate-data/gen_niche/idiomgap.jsonl")

# Quán ngữ sống trong VĂN NÓI ĐỜI THƯỜNG — đó là chỗ corpus đang trống. Cảnh KHÔNG
# lấy nghị trường/công sở trang trọng vì Quốc hội đã phủ dày phần đó rồi.
SCENES = [
    "hai người bạn thân nói chuyện phiếm",
    "gia đình trò chuyện trong bữa cơm",
    "đồng nghiệp tán gẫu giờ nghỉ trưa",
    "anh chị em cãi nhau rồi làm lành",
    "hàng xóm chào hỏi nhau ngoài ngõ",
    "bình luận trên mạng xã hội",
    "kể lại chuyện vừa xảy ra cho bạn nghe",
    "than thở về công việc với người thân",
    "khuyên nhủ một người bạn đang phân vân",
    "trêu chọc nhau giữa nhóm bạn",
    "nhật ký/blog cá nhân",
    "phỏng vấn đời thường trên truyền hình",
]

PROMPT = """Bạn là dịch giả Nhật-Việt, viết data huấn luyện cho model dịch.

Với MỖI quán ngữ dưới đây, viết {k} câu tiếng Nhật TỰ NHIÊN có dùng quán ngữ đó,
trong ngữ cảnh: {scene}. Sau đó dịch sang tiếng Việt.

Quán ngữ (kèm nghĩa tiếng Việt):
{items}

QUY TẮC BẮT BUỘC:
- Quán ngữ phải xuất hiện trong câu Nhật, được PHÉP chia đuôi cho hợp ngữ pháp.
- Câu Việt phải diễn đạt ĐÚNG NGHĨA BÓNG của quán ngữ, KHÔNG dịch từng chữ theo
  nghĩa đen. Ví dụ 猫の手も借りたい = "bận tối mắt tối mũi", KHÔNG phải "muốn mượn
  cả tay mèo".
- Không bắt buộc dùng đúng cụm tiếng Việt đã cho — miễn đúng nghĩa và tự nhiên.
- Câu Nhật 15-70 ký tự, giọng ĐỜI THƯỜNG (dùng だ/だよ/けど/じゃん... khi hợp),
  KHÔNG dùng văn phong hành chính.
- Mỗi câu một tình huống KHÁC nhau.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","idiom":"<quán ngữ đã dùng>"}}]"""


def stem(s):
    return s[:-1] if len(s) > 3 and s[-1] in "るうくつすぶむぐぬいな" else s


class Gen:
    def __init__(self, a, keys, gap):
        self.a, self.keys, self.gap = a, keys, gap
        self.stems = {k: stem(k) for k in gap}
        self.lock = asyncio.Lock()
        self.seen, self.kept, self.calls, self.errs = set(), 0, 0, 0
        self.covered = set()
        self.rej = {}
        self.t0 = time.time()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if OUT.exists():
            for line in OUT.open(encoding="utf-8"):
                try:
                    o = json.loads(line)
                    self.seen.add(o["ja"])
                    self.covered.add(o.get("idiom", ""))
                except Exception:  # noqa: BLE001
                    pass
            self.kept = len(self.seen)
            print(f"[resume] {self.kept:,} câu, phủ {len(self.covered):,}", flush=True)
        self.fo = OUT.open("a", encoding="utf-8")

    def qc(self, ja, vi, idiom):
        why = check(ja, vi)
        if why:
            return why
        if not (12 <= len(ja) <= 120):
            return "do_dai_ja"
        if VI_TONE_STRICT.search(ja) or VI_WORD_IN_JA.search(ja):
            return "ja_lan_tieng_viet"
        if self.stems[idiom] not in ja:
            return "quan_ngu_khong_trong_ja"
        return None

    def tasks(self):
        rng = random.Random(7)
        items = list(self.gap.items())
        rng.shuffle(items)
        g, rnd = self.a.group, 0
        while True:
            for s in range(0, len(items), g):
                chunk = [(k, d["vi"]) for k, d in items[s:s + g]]
                if chunk:
                    yield {"items": chunk, "scene": SCENES[(rnd + s // g) % len(SCENES)]}
            rnd += 1
            rng.shuffle(items)

    def prompt(self, t):
        items = "\n".join(f"- {ja}  =  {vi}" for ja, vi in t["items"])
        return PROMPT.format(k=self.a.per_item, scene=t["scene"], items=items)

    async def worker(self, wid, key, q, gate):
        client = genai.Client(api_key=key)
        cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig())
        cm = sess = None
        turns = 0
        await asyncio.sleep(wid * 0.05)

        async def close():
            nonlocal cm, sess, turns
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except BaseException:  # noqa: BLE001
                    pass
            cm = sess = None
            turns = 0

        while len(self.covered) < len(self.gap):
            try:
                t = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            tries = t.get("_try", 0)
            try:
                if sess is None or turns >= 8:
                    await close()
                    async with gate:
                        cm = client.aio.live.connect(model=self.a.model, config=cfg)
                        sess = await cm.__aenter__()
                    turns = 0
                await sess.send_client_content(turns=types.Content(
                    role="user", parts=[types.Part(text=self.prompt(t))]))
                buf = ""
                async for r in sess.receive():
                    sc = r.server_content
                    if not sc:
                        continue
                    ot = getattr(sc, "output_transcription", None)
                    if ot and ot.text:
                        buf += ot.text
                    if sc.turn_complete:
                        break
                turns += 1
                await self.absorb(t, parse_array(buf))
            except Exception as e:  # noqa: BLE001
                async with self.lock:
                    self.errs += 1
                    n = self.errs
                if n % 200 == 1:
                    print(f"  [w{wid}] {str(e)[:90]}", flush=True)
                await close()
                if tries + 1 < 4:
                    t["_try"] = tries + 1
                    q.put_nowait(t)
                await asyncio.sleep(min(1.5 * (tries + 1), 10))
        await close()

    async def absorb(self, t, recs):
        want = {k: v for k, v in t["items"]}
        async with self.lock:
            self.calls += 1
            for o in recs:
                if not isinstance(o, dict):
                    continue
                ja, vi = norm(str(o.get("ja") or "")), norm(str(o.get("vi") or ""))
                idm = str(o.get("idiom") or "").strip()
                if idm not in want:
                    idm = next((k for k in want if self.stems[k] in ja), "")
                if not idm:
                    self.rej["idiom_ngoai_ds"] = self.rej.get("idiom_ngoai_ds", 0) + 1
                    continue
                why = self.qc(ja, vi, idm)
                if why:
                    self.rej[why] = self.rej.get(why, 0) + 1
                    continue
                if ja in self.seen:
                    self.rej["trung_lap"] = self.rej.get("trung_lap", 0) + 1
                    continue
                self.seen.add(ja)
                self.covered.add(idm)
                self.fo.write(json.dumps(
                    {"ja": ja, "vi": vi, "src": "idiomgap", "idiom": idm,
                     "vi_idiom": want[idm], "scene": t["scene"]},
                    ensure_ascii=False) + "\n")
                self.kept += 1
            if self.calls % 20 == 0:
                self.fo.flush()
            if self.calls % 10 == 0:
                print(f"  {self.calls} req | {self.kept:,} câu | PHỦ "
                      f"{len(self.covered):,}/{len(self.gap):,} "
                      f"({100*len(self.covered)/len(self.gap):.0f}%) | lỗi {self.errs} "
                      f"| {(time.time()-self.t0)/60:.1f}p", flush=True)

    async def run(self):
        nreq = int(len(self.gap) / max(1, self.a.group) * 2.2 + 50)
        q = asyncio.Queue()
        gen = self.tasks()
        for _ in range(nreq):
            q.put_nowait(next(gen))
        nw = self.a.workers_per_key * len(self.keys)
        gate = asyncio.Semaphore(12)
        print(f"[run] {len(self.gap):,} quán ngữ x {self.a.per_item} câu | "
              f"{nw} session | {nreq} request\n", flush=True)
        await asyncio.gather(*[self.worker(i, self.keys[i % len(self.keys)], q, gate)
                               for i in range(nw)])
        self.fo.flush()
        self.fo.close()
        print(f"\n=== XONG ===\nCâu: {self.kept:,}")
        print(f"PHỦ: {len(self.covered):,}/{len(self.gap):,} "
              f"({100*len(self.covered)/len(self.gap):.1f}%)")
        print(f"Request {self.calls} | lỗi {self.errs} | {(time.time()-self.t0)/60:.1f} phút")
        if self.rej:
            print("Rớt QC:")
            for k, v in sorted(self.rej.items(), key=lambda x: -x[1])[:8]:
                print(f"  {k:28s} {v:>7,}")
        print(f"Output: {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gap", default=str(GAP))
    p.add_argument("--per-item", type=int, default=20)
    p.add_argument("--group", type=int, default=5)
    p.add_argument("--model", default="gemini-3.1-flash-live-preview")
    p.add_argument("--workers-per-key", type=int, default=2,
                   help="thấp vì chạy SONG SONG với job KD chính (chung hạn ngạch TPM)")
    a = p.parse_args()
    gap = json.load(open(a.gap, encoding="utf-8"))
    print(f"nạp {len(gap):,} quán ngữ thiếu", flush=True)
    asyncio.run(Gen(a, load_keys(), gap).run())


if __name__ == "__main__":
    main()
