#!/usr/bin/env python3
"""HARVEST danh sách quán ngữ Nhật + nghĩa Việt, qua Live API (thay bản cũ đã hỏng).

Vì sao cần: sau vòng enrich, `thanhngu` chỉ lên 1,8 -> 2,6 (vẫn thấp nhất) dù đã
sinh 30.778 cặp câu. Nguyên nhân KHÔNG phải thiếu câu mà thiếu ĐỘ PHỦ: seed chỉ có
1.241 quán ngữ, không trùng quán ngữ xuất hiện trong hardbench. Cần harvest thêm
vài nghìn quán ngữ MỚI rồi mới sinh câu cho chúng.

Khác bản cũ (`gen_idiom_harvest.py`): dùng Live API (`gemini-3.1-flash-live-preview`)
thay endpoint OpenAI-compat — bản cũ cần OPENAI_API_KEY (không có trong .env) và
model `gemini-2.5-flash-lite` đã 404.

  set PYTHONUTF8=1
  python scripts/harvest_idioms_live.py --target 5000
  # -> data/synthetic/gen/idiom_glosses.jsonl (append, dedup theo ja)
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
from gen_niche_kd import load_keys, parse_array  # noqa: E402

OUT = ROOT / "data" / "synthetic" / "gen" / "idiom_glosses.jsonl"

# 30 nhóm — rộng hơn hẳn 12 nhóm của bản cũ, thêm những mảng chưa từng harvest:
# オノマトペ (từ tượng thanh/tượng hình) là điểm mù kinh điển của dịch máy Nhật-Việt.
CATS = [
    "慣用句 với 目・耳 (目を疑う、耳が痛い…)",
    "慣用句 với 口・鼻・舌 (口が滑る、鼻が高い…)",
    "慣用句 với 手・指 (手を焼く、指をくわえる…)",
    "慣用句 với 足・腰 (足を洗う、腰が重い…)",
    "慣用句 với 頭・顔 (頭が下がる、顔が広い…)",
    "慣用句 với 胸・腹・肝 (胸を張る、肝を冷やす…)",
    "慣用句 với 気 (気が引ける、気を揉む…)",
    "慣用句 với 心・魂 (心を鬼にする、魂が抜ける…)",
    "慣用句 với 骨・血・肌 (骨が折れる、血の気が引く…)",
    "慣用句 công sở Nhật (顔を立てる、話を詰める、揉む…)",
    "慣用句 tiền bạc/kinh doanh (財布の紐、元手、算盤が合う…)",
    "慣用句 động vật (猿も木から落ちる、雀の涙…)",
    "慣用句 thực vật/hoa (根に持つ、実を結ぶ…)",
    "慣用句 thiên nhiên/thời tiết (雲行きが怪しい、日の目を見る…)",
    "慣用句 đồ vật/dụng cụ (釘を刺す、油を差す、匙を投げる…)",
    "慣用句 thức ăn (油を売る、味噌をつける、煮え湯を飲まされる…)",
    "慣用句 màu sắc và số đếm (白を切る、二の舞、三行半…)",
    "四字熟語 dùng trong hội thoại (自業自得、右往左往…)",
    "四字熟語 dùng trong văn viết/báo chí (前代未聞、暗中模索…)",
    "ことわざ về nỗ lực và thời gian (継続は力なり、待てば海路の日和あり…)",
    "ことわざ về quan hệ người với người (情けは人の為ならず…)",
    "ことわざ về thất bại và bài học (失敗は成功のもと、覆水盆に返らず…)",
    "オノマトペ mô tả cảm xúc (ドキドキ、イライラ、モヤモヤ、ワクワク…)",
    "オノマトペ mô tả trạng thái cơ thể (ぐったり、へとへと、ふらふら、ズキズキ…)",
    "オノマトペ mô tả cách làm việc (てきぱき、だらだら、こつこつ、ばたばた…)",
    "オノマトペ mô tả âm thanh/chuyển động (ざわざわ、しとしと、ぐるぐる…)",
    "cách nói uyển ngữ/né tránh trong tiếng Nhật (お察しください、遠慮させて…)",
    "quán ngữ hội thoại hằng ngày (それはそうと、とはいえ、なんだかんだ…)",
    "quán ngữ mô tả tính cách con người (腰が低い、目が利く、口が堅い…)",
    "quán ngữ về sức khỏe, tuổi tác, đời sống (歳には勝てない、無理がきかない…)",
]

PROMPT = """Liệt kê {k} quán ngữ / thành ngữ / cách nói cố định tiếng Nhật thuộc nhóm:
{cat}

YÊU CẦU:
- Chỉ chọn loại người Nhật THẬT SỰ dùng trong nói hoặc viết hiện đại. Bỏ loại cổ,
  chỉ có trong từ điển, không ai dùng.
- Nghĩa tiếng Việt phải là NGHĨA BÓNG chính xác, tự nhiên, 3-12 từ.
  Ví dụ 油を売る = "la cà lười biếng" (KHÔNG phải "bán dầu").
- Nếu tiếng Việt có thành ngữ tương đương thì ưu tiên dùng (猫に小判 = "đàn gảy tai trâu").
{avoid}
CHỈ trả JSON array, không markdown, không giải thích:
[{{"ja":"<quán ngữ>","reading":"<hiragana>","vi":"<nghĩa tiếng Việt>","freq":<1|2|3, 1=rất hay gặp>}}]"""

JA_OK = re.compile(r"[぀-ヿ一-鿿]")
VI_TONE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


class Harvest:
    def __init__(self, a, keys):
        self.a, self.keys = a, keys
        self.lock = asyncio.Lock()
        self.seen, self.added, self.calls, self.errs = set(), 0, 0, 0
        self.t0 = time.time()
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if OUT.exists():
            for line in OUT.open(encoding="utf-8"):
                try:
                    self.seen.add(json.loads(line)["ja"])
                except Exception:  # noqa: BLE001
                    pass
        self.start = len(self.seen)
        print(f"[resume] đã có {self.start:,} quán ngữ", flush=True)
        self.fo = OUT.open("a", encoding="utf-8")

    def ok(self, ja, vi):
        if not (2 <= len(ja) <= 24) or not JA_OK.search(ja):
            return False
        if not (3 <= len(vi) <= 90) or not VI_TONE.search(vi):
            return False
        return JA_OK.search(vi) is None       # nghĩa Việt không được lẫn chữ Nhật

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

        while self.added < self.a.target:
            try:
                cat = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if sess is None or turns >= 8:
                    await close()
                    async with gate:
                        cm = client.aio.live.connect(model=self.a.model, config=cfg)
                        sess = await cm.__aenter__()
                    turns = 0
                # đưa mẫu ĐÃ CÓ vào prompt để model tránh lặp (cắt ngẫu nhiên 60 cái)
                async with self.lock:
                    pool = list(self.seen)
                avoid = ""
                if pool:
                    s = random.sample(pool, min(60, len(pool)))
                    avoid = "- TRÁNH các mục đã có: " + "、".join(s) + "\n"
                await sess.send_client_content(turns=types.Content(
                    role="user", parts=[types.Part(text=PROMPT.format(
                        k=self.a.per_req, cat=cat, avoid=avoid))]))
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
                await self.absorb(cat, parse_array(buf))
            except Exception as e:  # noqa: BLE001
                async with self.lock:
                    self.errs += 1
                    n = self.errs
                if n % 50 == 1:
                    print(f"  [w{wid}] {str(e)[:100]}", flush=True)
                await close()
                await asyncio.sleep(1.5)
        await close()

    async def absorb(self, cat, recs):
        async with self.lock:
            self.calls += 1
            n0 = self.added
            for o in recs:
                if not isinstance(o, dict):
                    continue
                ja = str(o.get("ja") or "").strip()
                vi = str(o.get("vi") or "").strip()
                if not self.ok(ja, vi) or ja in self.seen:
                    continue
                self.seen.add(ja)
                self.fo.write(json.dumps(
                    {"ja": ja, "reading": str(o.get("reading") or "").strip(),
                     "vi": vi, "freq": o.get("freq", 2), "cat": cat[:40]},
                    ensure_ascii=False) + "\n")
                self.added += 1
            if self.calls % 20 == 0:
                self.fo.flush()
            if self.calls % 10 == 0:
                el = (time.time() - self.t0) / 60
                print(f"  {self.calls} req | +{self.added:,}/{self.a.target:,} mới "
                      f"(tổng {len(self.seen):,}) | {self.added/max(el,1e-9):.0f}/phút "
                      f"| lỗi {self.errs} | {el:.1f}p", flush=True)
            _ = n0

    async def run(self):
        rng = random.Random(7)
        nreq = self.a.max_requests or int(self.a.target / max(1, self.a.per_req) * 4 + 40)
        q = asyncio.Queue()
        pool = CATS * (nreq // len(CATS) + 1)
        rng.shuffle(pool)
        for c in pool[:nreq]:
            q.put_nowait(c)
        nw = self.a.workers_per_key * len(self.keys)
        gate = asyncio.Semaphore(12)
        print(f"[run] {len(CATS)} nhóm | {len(self.keys)} key x {self.a.workers_per_key}"
              f" = {nw} session | {nreq} request | target +{self.a.target:,}\n", flush=True)
        await asyncio.gather(*[self.worker(i, self.keys[i % len(self.keys)], q, gate)
                               for i in range(nw)])
        self.fo.flush()
        self.fo.close()
        print(f"\n=== HARVEST XONG ===")
        print(f"Quán ngữ mới : +{self.added:,} (tổng {len(self.seen):,}, trước {self.start:,})")
        print(f"Request      : {self.calls} | lỗi {self.errs} | {(time.time()-self.t0)/60:.1f} phút")
        print(f"Output       : {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=4000, help="số quán ngữ MỚI cần thêm")
    p.add_argument("--model", default="gemini-3.1-flash-live-preview")
    p.add_argument("--workers-per-key", type=int, default=6)
    p.add_argument("--per-req", type=int, default=40)
    p.add_argument("--max-requests", type=int, default=0)
    a = p.parse_args()
    asyncio.run(Harvest(a, load_keys()).run())


if __name__ == "__main__":
    main()
