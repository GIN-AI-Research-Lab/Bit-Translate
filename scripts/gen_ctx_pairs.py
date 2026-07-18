#!/usr/bin/env python3
"""Sinh cặp ngữ cảnh `ctx ||| src` bằng CODE từ cặp liền kề OpenSubtitles/TED — 0 token LLM.

(PLAN_BUOC5 §8.3 + §9#1, PLAN_RANKUP §2.2, Opus review G-2.)
Trị: zero-pronoun, xưng hô, nhất quán register — nhóm judge 1.9-2.3 mà Google cũng yếu (3.2-3.5).

Cách làm: data/clean/train.{src,ja,vi} GIỮ NGUYÊN thứ tự gốc trong từng corpus
(đã kiểm chứng 2026-07-18: OpenSubtitles đọc liền mạch như kịch bản) → dòng i-1 cùng
nguồn = câu ngữ cảnh của dòng i. Data sạch có dòng bị lọc rớt nên thi thoảng ctx không
thật sự liền kề — chấp nhận được (ctx là gợi ý, không phải target; nhiễu nhẹ vô hại).

Mẫu ja2vi: input "ctx_ja ||| src_ja" → target vi[i]      (chữa zero-pronoun ja→vi)
Mẫu vi2ja: input "ctx_vi ||| src_vi" → target ja[i]      (nhất quán register vi→ja)
LƯU Ý MIX: đây là data MỘT CHIỀU (có field "dir") — binarize KHÔNG được tự sinh chiều
ngược (target "ctx ||| src" là vô nghĩa). Separator ||| đã kiểm tra: 3 token sạch trong SPM.

Ưu tiên mẫu "đáng tiền": target VI có đại từ tường minh / câu JA là câu hỏi-mệnh lệnh
(đúng chỗ ngữ cảnh giải nghĩa); mẫu khác giữ theo xác suất để đa dạng.
Chạy: python scripts/gen_ctx_pairs.py [n_max_mỗi_chiều=25000]
Ghi: data/synthetic/vong3/ctx_pairs.jsonl
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "synthetic" / "vong3" / "ctx_pairs.jsonl"
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 25000   # mỗi chiều
SOURCES = {"opus_OpenSubtitles", "opus_TED2020"}
rng = random.Random(42)

DASH = re.compile(r"^\s*[-–—]\s*")
# đại từ/xưng hô VI tường minh — mẫu mà ngữ cảnh thật sự dạy được cách chọn ngôi
VI_PRON = re.compile(r"\b(anh|em|chị|tôi|tớ|cậu|mình|ông|bà|cô|chú|bác|con|cháu|mày|tao|"
                     r"chúng ta|chúng tôi|bọn mình|các bạn|quý khách|sếp)\b", re.I)
JA_HOOK = re.compile(r"[?？]|てください|なさい|ましょう|んですか|ますか|でしょうか")


def clean(s):
    return DASH.sub("", s.strip())


def ok_len(ctx, src):
    return 2 <= len(ctx) <= 90 and 4 <= len(src) <= 120 and "|||" not in ctx and "|||" not in src


def main():
    srcs = CLEAN.joinpath("train.src").read_text(encoding="utf-8").splitlines()
    jas = CLEAN.joinpath("train.ja").read_text(encoding="utf-8").splitlines()
    vis = CLEAN.joinpath("train.vi").read_text(encoding="utf-8").splitlines()
    assert len(srcs) == len(jas) == len(vis)
    cand_ja2vi, cand_vi2ja = [], []
    seen = set()
    for i in range(1, len(srcs)):
        if srcs[i] not in SOURCES or srcs[i - 1] != srcs[i]:
            continue
        cja, cvi = clean(jas[i - 1]), clean(vis[i - 1])
        sja, svi = clean(jas[i]), clean(vis[i])
        if not (ok_len(cja, sja) and ok_len(cvi, svi)):
            continue
        key = (sja, svi)
        if key in seen:
            continue
        seen.add(key)
        # điểm "đáng tiền": target có đại từ VI tường minh / câu JA hỏi-mệnh lệnh
        hot = bool(VI_PRON.search(svi)) or bool(JA_HOOK.search(sja))
        if hot or rng.random() < 0.15:
            cand_ja2vi.append({"ja": f"{cja} ||| {sja}", "vi": svi,
                               "src": "vong3_ctx", "dir": "ja2vi"})
        # chiều vi→ja: ưu tiên register (câu JA target có keigo/thể thường rõ)
        if hot or rng.random() < 0.10:
            cand_vi2ja.append({"vi": f"{cvi} ||| {svi}", "ja": sja,
                               "src": "vong3_ctx", "dir": "vi2ja"})
    rng.shuffle(cand_ja2vi)
    rng.shuffle(cand_vi2ja)
    out = cand_ja2vi[:N_MAX] + cand_vi2ja[:N_MAX]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out),
                   encoding="utf-8")
    print(f"ứng viên: ja2vi {len(cand_ja2vi):,} | vi2ja {len(cand_vi2ja):,}")
    print(f"GHI {len(out):,} mẫu ctx ||| src -> {OUT}")


if __name__ == "__main__":
    main()
