#!/usr/bin/env python3
"""Sinh câu cho THUẬT NGỮ THIẾU — data theo DANH SÁCH PHỦ, không sinh ngẫu nhiên.

Vì sao khác các mode trong gen_niche_kd.py: những mode đó sinh theo chủ đề tự nghĩ
ra, xong mới biết phủ được gì. Kết quả đo được: vòng 3 sinh 241k cặp -> hardbench
+18 điểm nhưng bench câu THẬT 0 điểm.

Script này đi từ DANH SÁCH ĐO ĐƯỢC (`measure_term_coverage.py` -> term_gap.json):
8.041/11.933 thuật ngữ glossary xuất hiện <= 5 lần trong corpus, trong đó 2.343
thuật ngữ CHƯA TỪNG xuất hiện. Mỗi thuật ngữ được sinh K câu trong ngữ cảnh khác
nhau, và bản dịch tiếng Việt LẤY TỪ GLOSSARY chứ không để model tự đoán.

  set PYTHONUTF8=1
  python scripts/gen_term_gap.py --per-term 9
  # -> D:/Bit-Translate-data/gen_niche/termgap.jsonl
  # Sau đó đo lại: python scripts/measure_term_coverage.py
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

GAP = Path("D:/Bit-Translate-data/term_gap.json")
OUT = Path("D:/Bit-Translate-data/gen_niche/termgap.jsonl")

# Ngữ cảnh để cùng một thuật ngữ ra nhiều câu khác nhau — nghiêng về nghiệp vụ
# vì glossary của dự án thiên về ERP/sản xuất (ピッキングリスト, ファントム品, 製番…).
SCENES = [
    "email trao đổi công việc giữa hai bộ phận",
    "màn hình/hướng dẫn sử dụng phần mềm nghiệp vụ",
    "biên bản họp dự án",
    "báo cáo tiến độ gửi cấp trên",
    "trao đổi với khách hàng về yêu cầu hệ thống",
    "tài liệu định nghĩa yêu cầu (要件定義書)",
    "phiếu báo lỗi / yêu cầu sửa đổi",
    "hội thoại tại hiện trường sản xuất",
    "quy trình vận hành chuẩn (SOP)",
    "trao đổi nhanh trên chat nội bộ",
    "bài viết giải thích khái niệm cho người mới",
    "câu hỏi và trả lời trong buổi đào tạo",
]

PROMPT = """Bạn là dịch giả Nhật-Việt chuyên ngành CNTT/nghiệp vụ sản xuất, viết data
huấn luyện cho model dịch.

Với MỖI thuật ngữ dưới đây, viết {k} câu tiếng Nhật TỰ NHIÊN có dùng thuật ngữ đó,
trong ngữ cảnh: {scene}. Sau đó dịch sang tiếng Việt.

Thuật ngữ (kèm bản dịch tiếng Việt CHUẨN của công ty — PHẢI dùng đúng bản dịch này):
{items}

QUY TẮC BẮT BUỘC:
- Bản dịch tiếng Việt của thuật ngữ PHẢI đúng như đã cho. KHÔNG tự đặt cách dịch khác,
  KHÔNG phiên âm, KHÔNG bỏ qua.
- Nếu bản dịch chuẩn là tiếng Anh (vd "Picking list") thì giữ nguyên tiếng Anh trong
  câu Việt — đó là cách người trong ngành thật sự viết.
- Câu Nhật 20-70 ký tự, có ngữ cảnh cụ thể (tên hệ thống/bộ phận/thao tác), KHÔNG
  phải câu mẫu trống rỗng.
- Mỗi câu dùng thuật ngữ trong một tình huống KHÁC nhau (đăng ký, tra cứu, lỗi,
  xác nhận, hướng dẫn, báo cáo...).
- Phần còn lại của câu Việt phải trôi chảy như người Việt trong ngành viết.

CHỈ trả về JSON array, không markdown:
[{{"ja":"<câu Nhật>","vi":"<câu Việt>","term":"<thuật ngữ Nhật đã dùng>"}}]"""


class Gen:
    def __init__(self, a, keys, gap):
        self.a, self.keys, self.gap = a, keys, gap
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
                    self.covered.add(o.get("term", ""))
                except Exception:  # noqa: BLE001
                    pass
            self.kept = len(self.seen)
            print(f"[resume] đã có {self.kept:,} câu, phủ {len(self.covered):,} thuật ngữ",
                  flush=True)
        self.fo = OUT.open("a", encoding="utf-8")
        self.fr = OUT.with_name("termgap_rejects.jsonl").open("a", encoding="utf-8")

    def qc(self, ja, vi, term, vi_term):
        why = check(ja, vi)
        if why:
            return why
        if not (12 <= len(ja) <= 150):
            return "do_dai_ja"
        if VI_TONE_STRICT.search(ja) or VI_WORD_IN_JA.search(ja):
            return "ja_lan_tieng_viet"
        if term not in ja:
            return "term_khong_trong_ja"
        # bản dịch chuẩn phải xuất hiện trong câu Việt (khớp lỏng: token đầu tiên)
        head = re.split(r"[,/;(]", vi_term)[0].strip().lower()
        if len(head) >= 3 and head not in vi.lower():
            return "vi_term_khong_dung_ban_chuan"
        return None

    def tasks(self):
        rng = random.Random(99)
        items = list(self.gap.items())
        rng.shuffle(items)
        g = self.a.group
        rnd = 0
        while True:
            for s in range(0, len(items), g):
                chunk = [(t, d["vi"]) for t, d in items[s:s + g]]
                if chunk:
                    yield {"items": chunk, "scene": SCENES[(rnd + s // g) % len(SCENES)]}
            rnd += 1
            rng.shuffle(items)

    def prompt(self, t):
        items = "\n".join(f"- {ja}  →  {vi}" for ja, vi in t["items"])
        return PROMPT.format(k=self.a.per_term, scene=t["scene"], items=items)

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
        want = {ja: vi for ja, vi in t["items"]}
        async with self.lock:
            self.calls += 1
            for o in recs:
                if not isinstance(o, dict):
                    continue
                ja, vi = norm(str(o.get("ja") or "")), norm(str(o.get("vi") or ""))
                term = str(o.get("term") or "").strip()
                if term not in want:            # model tự bịa thuật ngữ khác
                    term = next((k for k in want if k in ja), "")
                if not term:
                    self.rej["term_ngoai_danh_sach"] = self.rej.get("term_ngoai_danh_sach", 0) + 1
                    continue
                why = self.qc(ja, vi, term, want[term])
                if why:
                    self.rej[why] = self.rej.get(why, 0) + 1
                    self.fr.write(json.dumps({"reject": why, "ja": ja, "vi": vi,
                                              "term": term}, ensure_ascii=False) + "\n")
                    continue
                if ja in self.seen:
                    self.rej["trung_lap"] = self.rej.get("trung_lap", 0) + 1
                    continue
                self.seen.add(ja)
                self.covered.add(term)
                self.fo.write(json.dumps(
                    {"ja": ja, "vi": vi, "src": "termgap", "term": term,
                     "vi_term": want[term], "scene": t["scene"]}, ensure_ascii=False) + "\n")
                self.kept += 1
            if self.calls % 20 == 0:
                self.fo.flush()
                self.fr.flush()
            if self.calls % 10 == 0:
                el = (time.time() - self.t0) / 60
                print(f"  {self.calls} req | {self.kept:,} câu | PHỦ "
                      f"{len(self.covered):,}/{len(self.gap):,} thuật ngữ "
                      f"({100*len(self.covered)/len(self.gap):.0f}%) | lỗi {self.errs} "
                      f"| {el:.1f}p", flush=True)

    async def run(self):
        nreq = int(len(self.gap) / max(1, self.a.group) * 2.2 + 50)
        q = asyncio.Queue()
        gen = self.tasks()
        for _ in range(nreq):
            q.put_nowait(next(gen))
        nw = self.a.workers_per_key * len(self.keys)
        gate = asyncio.Semaphore(12)
        print(f"[run] {len(self.gap):,} thuật ngữ thiếu x {self.a.per_term} câu | "
              f"{nw} session | {nreq} request\n", flush=True)
        await asyncio.gather(*[self.worker(i, self.keys[i % len(self.keys)], q, gate)
                               for i in range(nw)])
        self.fo.flush()
        self.fo.close()
        self.fr.flush()
        self.fr.close()
        print(f"\n=== XONG ===")
        print(f"Câu       : {self.kept:,}")
        print(f"PHỦ       : {len(self.covered):,}/{len(self.gap):,} thuật ngữ "
              f"({100*len(self.covered)/len(self.gap):.1f}%)")
        print(f"Request   : {self.calls} | lỗi {self.errs} | {(time.time()-self.t0)/60:.1f} phút")
        if self.rej:
            print("Rớt QC:")
            for k, v in sorted(self.rej.items(), key=lambda x: -x[1])[:8]:
                print(f"  {k:30s} {v:>7,}")
        print(f"Output    : {OUT}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gap", default=str(GAP))
    p.add_argument("--per-term", type=int, default=9, help="số câu mỗi thuật ngữ")
    p.add_argument("--group", type=int, default=5, help="số thuật ngữ mỗi request")
    p.add_argument("--model", default="gemini-3.1-flash-live-preview")
    p.add_argument("--workers-per-key", type=int, default=5)
    a = p.parse_args()
    gap = json.load(open(a.gap, encoding="utf-8"))
    print(f"nạp {len(gap):,} thuật ngữ thiếu từ {a.gap}", flush=True)
    asyncio.run(Gen(a, load_keys(), gap).run())


if __name__ == "__main__":
    main()
