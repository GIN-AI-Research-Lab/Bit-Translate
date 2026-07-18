#!/usr/bin/env python3
"""Vòng 3 — sinh câu từ từ điển idiom/slang (PLAN_RANKUP §2.1 + §8.2).

3 mode (task = "<mode>:<idx>", GEN_TODO = file json list task trong data/synthetic/gen/):
  idc   câu ngữ cảnh cho mục gốc JA: 2 mục/task × 8 câu — như idh vòng 2 nhưng
        thêm yêu cầu ≥2 câu dài 25-40 từ (bệnh câu dài vi→ja, §8.1)
  idcv  câu ngữ cảnh cho mục gốc VIỆT (vong3_glosses_vi): câu Việt chứa mục
        + bản dịch Nhật theo NGHĨA — chiều vi→ja đang 1.22, đòn chính của vòng này
  ct    CẶP TƯƠNG PHẢN nghĩa đen vs nghĩa bóng (§8.2): 1 mục/task, chỉ mục có "lit"
        — dạy model ĐIỀU KIỆN chọn nghĩa, không học vẹt 1-1

Gloss pool đọc từ GLOSS_FILE (mặc định idiom_glosses_reviewed.jsonl — bộ ĐÃ review;
sau khi harvest vòng 3 được Haiku review thì đổi sang file đã lọc).
Ghi data/synthetic/vong3/out_<task>.jsonl, resume theo file có sẵn, gắn "m"=model.
Env: OPENAI_API_KEY/BASE_URL, GEN_MODEL, GEN_RPM, GEN_REASONING, GEN_TODO, GLOSS_FILE.
"""
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
GEN = ROOT / "data" / "synthetic" / "gen"
OUT = ROOT / "data" / "synthetic" / "vong3"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("GEN_MODEL", "glm-4.7-flash")
MIN_GAP = 60.0 / float(os.environ.get("GEN_RPM", "20"))
REASONING = os.environ.get("GEN_REASONING", "")
client = OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])

COMMON = """CHỈ trả JSONL, mỗi dòng {{"ja":"...","vi":"..."}}. Không markdown, không giải thích, không đánh số.
Tiếng Việt TỰ NHIÊN như người Việt nói; tiếng Nhật bản ngữ tự nhiên."""

# Discourse marker / quotative — lỗi đời thật 2026-07-18: って+mệnh lệnh bị dịch trôi
# thành "sẽ...", あくまで dịch nghĩa đen (Opus review G-3).
DM_SEEDS = [
    ("〜しろって／〜するなって (trích mệnh lệnh)", "truyền đạt lại MỆNH LỆNH của người khác — bản dịch phải giữ tính ra lệnh/cấm (bảo là hãy…/đừng…), KHÔNG biến thành 'sẽ làm'"),
    ("〜って言ってた (trích lời gián tiếp)", "kể lại lời người khác — giữ đúng người nói/người nghe, đúng chiều yêu cầu"),
    ("〜だって (nghe nói/trích ngạc nhiên)", "truyền tin nghe được, kèm sắc thái ngạc nhiên"),
    ("あくまで", "chỉ là/đơn thuần là/trên nguyên tắc — KHÔNG dịch 'cho đến cùng' trừ nghĩa kiên trì"),
    ("一応", "tạm coi là/cho chắc/về hình thức thì"),
    ("とりあえず", "trước mắt cứ/tạm thời"),
    ("さすがに", "đến mức này thì/quả là — tuỳ ngữ cảnh khen hoặc chịu hết nổi"),
    ("まさか", "không ngờ/lẽ nào — phủ định kỳ vọng"),
    ("どうせ", "đằng nào cũng/kiểu gì chẳng — buông xuôi hoặc tận dụng"),
    ("せっかく", "mất công/đã trót/hiếm có dịp — tiếc công sức/cơ hội"),
    ("むしろ", "đúng hơn là/thà rằng"),
    ("かえって", "ngược lại càng/hoá ra lại"),
    ("逆に", "ngược lại/mà nói ngược lại thì (khẩu ngữ)"),
    ("要するに／つまり", "tóm lại là/tức là"),
    ("別に〜ない", "chẳng có gì đặc biệt/không hẳn — giọng hờ hững"),
    ("案の定", "quả nhiên/y như rằng"),
    ("どうりで", "thảo nào/hèn gì"),
    ("思ったより", "hơn tưởng tượng/không như nghĩ"),
    ("いまさら", "bây giờ mới… thì muộn rồi/còn nói gì nữa"),
    ("わざわざ", "cất công/đặc biệt bỏ công — kèm sắc thái cảm kích hoặc mỉa"),
    ("せめて", "ít nhất thì/giá mà được"),
    ("ちなみに", "nhân tiện/nói thêm"),
    ("というか", "mà nói đúng ra/hay nói cách khác (khẩu ngữ)"),
    ("それにしても", "dù vậy thì/mà công nhận"),
    ("いずれにせよ／いずれにしても", "dù thế nào đi nữa/đằng nào thì"),
    ("何なら", "nếu cần thì/thậm chí (khẩu ngữ đề nghị)"),
]

P_DM = """Cụm chức năng diễn ngôn tiếng Nhật: {seed} — chức năng/cách dịch: {rule}.
Viết 20 cặp câu Nhật-Việt dùng cụm này ĐÚNG CHỨC NĂNG trong ngữ cảnh đa dạng (hội thoại đời thường, công sở IT, chat, kể chuyện) — xen câu ngắn và câu dài. Bản dịch Việt phải giữ đúng CHỨC NĂNG DIỄN NGÔN và chiều logic (mệnh lệnh giữ tính ra lệnh, phủ định kỳ vọng giữ ngạc nhiên...), TUYỆT ĐỐI không dịch nghĩa đen từng chữ.
""" + COMMON

P_IDC = """Các mục tiếng Nhật (kèm nghĩa Việt chuẩn): {seed}.
Với MỖI mục viết 8 cặp câu Nhật-Việt ở ngữ cảnh KHÁC NHAU (họp, chat đồng nghiệp, gia đình, email, kể chuyện, mạng xã hội) — ngữ cảnh thể hiện NGAY trong câu Nhật, không câu nào giống nhau. Trong 8 câu phải có ≥2 câu DÀI 25-40 từ nhiều mệnh đề. Bản dịch Việt theo NGHĨA đã cho, chỉnh giọng theo ngữ cảnh. TUYỆT ĐỐI không dịch từng chữ.
""" + COMMON

P_IDCV = """Các mục tiếng VIỆT (thành ngữ/slang/khẩu ngữ, kèm cách nói Nhật tương đương): {seed}.
Với MỖI mục viết 8 cặp câu: câu VIỆT tự nhiên CHỨA mục đó (đời thường, công sở, chat, kể chuyện — đa dạng, có ≥2 câu dài 25-40 từ) + bản dịch NHẬT theo NGHĨA/CHỨC NĂNG bằng cách nói đã cho hoặc cách nói Nhật tự nhiên hợp ngữ cảnh hơn. Register Nhật khớp ngữ cảnh (bạn bè → thể thường; công sở → 丁寧語). TUYỆT ĐỐI không dịch nghĩa đen mục ("toang" KHÔNG phải 崩壊, mà 詰んだ/やばい tuỳ câu).
""" + COMMON

P_CT = """Mục: 「{seed}」 — nghĩa dùng/nghĩa bóng: {sense}; nghĩa đen: {lit}.
Viết 10 cặp câu {src_lang} + bản dịch: 5 câu dùng mục theo NGHĨA BÓNG (ngữ cảnh làm rõ), 5 câu ngữ cảnh mà cụm này (hoặc từ trong đó) mang NGHĨA ĐEN thật. Bản dịch phải KHÁC HẲN giữa hai nhóm — mục tiêu: dạy máy dịch phân biệt bằng ngữ cảnh. Ví dụ mẫu: "gấu" = 恋人 trong chuyện hẹn hò / = クマ trong chuyện sở thú.
""" + COMMON


def load_pool():
    f = GEN / os.environ.get("GLOSS_FILE", "idiom_glosses_reviewed.jsonl")
    return [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]


POOL = load_pool()


def build_prompt(mode, idx):
    if mode == "idc":
        pair = POOL[idx * 2:(idx + 1) * 2]
        seed = "; ".join(f"「{g['ja']}」= {g['vi']}" for g in pair)
        return P_IDC.format(seed=seed)
    if mode == "idcv":
        pair = POOL[idx * 2:(idx + 1) * 2]
        seed = "; ".join(f"\"{g['vi']}\" → {g['ja']}" for g in pair)
        return P_IDCV.format(seed=seed)
    if mode == "dm":
        seed, rule = DM_SEEDS[idx]
        return P_DM.format(seed=seed, rule=rule)
    if mode == "ct":
        g = POOL[idx]
        if g.get("side") == "vi":   # mục gốc VIỆT (pool builder gắn side)
            return P_CT.format(seed=g["vi"], sense=g["ja"], lit=g.get("lit", "nghĩa đen từng chữ"),
                               src_lang="tiếng Việt")
        return P_CT.format(seed=g["ja"], sense=g["vi"], lit=g.get("lit", "nghĩa đen từng chữ"),
                           src_lang="tiếng Nhật")
    raise ValueError(mode)


def gen(task):
    mode, idx = task.split(":")[0], int(task.split(":")[1])
    extra = {"reasoning_effort": REASONING} if REASONING else {}
    r = client.chat.completions.create(model=MODEL, temperature=0.85, max_tokens=8000,
                                       messages=[{"role": "user", "content": build_prompt(mode, idx)}],
                                       **extra)
    txt = re.sub(r"^```[a-z]*\n?|```$", "", (r.choices[0].message.content or "").strip(), flags=re.M)
    out = []
    for line in txt.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            p = json.loads(line)
            if p.get("ja") and p.get("vi"):
                out.append({"ja": p["ja"], "vi": p["vi"], "src": f"vong3_{mode}", "m": MODEL})
        except json.JSONDecodeError:
            pass
    return out


def main():
    todo = json.loads((GEN / os.environ["GEN_TODO"]).read_text())
    todo = [t for t in todo if not (OUT / f"out_{t.replace(':', '_')}.jsonl").exists()]
    print(f"vong3 [{MODEL}] pool={len(POOL)} | còn {len(todo)} task", flush=True)
    last, done, total = 0.0, 0, 0
    for t in todo:
        gap = MIN_GAP - (time.time() - last)
        if gap > 0:
            time.sleep(gap)
        last = time.time()
        try:
            pairs = gen(t)
        except Exception as e:  # noqa: BLE001
            print(f"  {t}: LỖI {str(e)[:70]}", flush=True)
            continue
        if pairs:
            (OUT / f"out_{t.replace(':', '_')}.jsonl").write_text(
                "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
            done += 1
            total += len(pairs)
            print(f"  {t}: +{len(pairs)} ({done}/{len(todo)}, tổng {total})", flush=True)
    print(f"XONG {done} task, {total} cặp.", flush=True)


if __name__ == "__main__":
    main()
