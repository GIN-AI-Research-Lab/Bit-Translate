# -*- coding: utf-8 -*-
"""
Exp AB — SUB-BIT mới: ROLLOUT-KD (on-policy/DAgger) × BIT-CURRICULUM lồng nhau (0.33→1.56bpw)
trên Qwen3-0.6B. Thiết kế từ tổng hợp toàn bộ bằng chứng lab (README Bài 0–19):

  1) Lỗ hổng loss: mọi KD trước giờ là TEACHER-FORCED — student chưa bao giờ được train trên
     trạng thái DO CHÍNH NÓ SINH RA, nơi nhiễu lượng tử đẩy nó vào attractor lặp ("và có thể,
     và có thể…" @1.56bpw; 30B S1 "đúng ngữ pháp cục bộ, vô nghĩa toàn cục"). PPL teacher-forced
     mù với hiện tượng này (Bài 12/15: "sinh-mở đòi hỏi cao hơn PPL nhiều"). Fix: mỗi K bước,
     student TỰ SINH rollout; teacher chấm full-vocab (top-K + lseT như exp_w v2) NGAY TRÊN
     ngữ cảnh đó; loss = (1−λ)·KL_TF + λ·KL_rollout. Toy gate G3 (exp_aa) xác nhận cơ chế:
     exposure gap 0.535 → 0.377, self-acc +15.8đ trên manifold có cấu trúc.
  2) Đường đi tối ưu: curriculum bit-budget qua họ mask N:M LỒNG NHAU
     1:32 ⊂ 1:16 ⊂ 1:8 ⊂ 1:4 ⊂ 2:4 (0.331→1.566bpw, đúng dải user yêu cầu; đích cuối
     = đúng khung TQ33/F0: 2:4 + scale f8-g64). Mask cố định TOÀN BỘ từ đầu (tournament trên
     Wanda importance) — không bao giờ refresh (luật "mask refresh = ×2.2"); chuyển stage chỉ
     THÊM ô. MASKED-STE bắt buộc: toy gate G2 đo trên tensor thật cho thấy full-STE làm ô
     pruned random-walk 0.22 (hơn cả ô kept!) → grow shock ×10.8 và 85% ô mới bật nhiễu ±1;
     masked-STE: drift 0, shock ×4.7, 0.8% ô mới bật (soft-start tự nhiên).
  3) Highway columns (tùy chọn): kênh massive-activation của hidden (đo được, không hardcode
     — 30B là kênh 0, 0.6B báo cáo ~48/52) giữ FP ở PHÍA CỘT TRỌNG SỐ từ đầu train
     (bake-in, khác vá hậu kỳ outlier-fix đã âm tính), giá ~0.05bpw.

Arms (so sánh CÙNG tổng bước, cùng seed):
  base    --ladder 2:4              --rollout 0     (GEN4-class, mốc)
  roll    --ladder 2:4              --rollout 1     (đo riêng tác dụng rollout-KD)
  curr    --ladder 1:32,...,2:4     --rollout 0     (đo riêng curriculum)
  cr      --ladder 1:32,...,2:4     --rollout 1     (gộp)

Thước BẮT BUỘC ngoài PPL: behavior battery (sinh THẬT greedy + sampling t=0.7, validator
tự động: số học/fact/code-exec/format) + loop-rate + divergence curve (exp_ac).

Chạy smoke máy B (CPU):  python exp_ab_subbit_curriculum.py --smoke
Modal: cloud/modal_qat_subbit.py (KHÔNG tự chạy — ví đang cạn, cần user xác nhận).
"""
import argparse
import glob
import io
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root")
from exp_r_qat_lite import (CAL_CHAT, CAL_CODE, CAL_EN, CAL_MATH, CAL_ZH,   # noqa: E402
                            CODE_EVAL, EN_EVAL, HARD_BLOCKS, EASY_RANGE, LIN_PATHS,
                            MATH_EVAL, NORM_PATHS, ZH_EVAL, apply_gauge,
                            apply_perm_gauge, eval_ppl, inv_softplus, read_lines,
                            to_f16, to_f8)

OUT_DIR = os.environ.get("EXPR_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
DEF_MODEL = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
DEF_VI = r"E:\Bit-Translate\cloud\qat_data\train_slice.vi"
DEF_JA = r"E:\Bit-Translate\cloud\qat_data\train_slice.ja"
DEF_DVI = r"E:\Bit-Translate\cloud\qat_data\dev.vi"
DEF_DJA = r"E:\Bit-Translate\cloud\qat_data\dev.ja"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ============================ họ mask lồng nhau + quantizer ============================
def nm_bpw(n, m, sgroup=64, sbits=8.0):
    return n / m * math.log2(3) + math.log2(math.comb(m, n)) / m + sbits / sgroup


CANON = [(1, 32), (1, 16), (1, 8), (1, 4), (2, 4)]   # chuỗi chia-đôi mà tournament hỗ trợ


def parse_ladder(s):
    lad = []
    for part in s.split(","):
        n, m = part.strip().split(":")
        lad.append((int(n), int(m)))
    assert len(lad) == 1 or lad == CANON[len(CANON) - len(lad):], \
        f"ladder phải là 1 stage hoặc suffix của {CANON} (họ lồng nhau tournament)"
    return lad


def nested_rank(imp, ladder):
    """rank[i,j] = stage sớm nhất ô (i,j) active (tournament từ mịn -> thô; xem exp_aa G1).
    Yêu cầu: các stage là dãy chia đôi nhóm chuẩn (…,1:4M,…,1:8,1:4) kết thúc (2:4) hoặc
    1 stage duy nhất."""
    R, C = imp.shape
    L = len(ladder)
    rank = torch.full((R, C), 127, dtype=torch.long, device=imp.device)
    if L == 1:
        n, m = ladder[0]
        g = imp.view(R, -1, m)
        topn = g.topk(n, dim=2).indices
        base = torch.arange(0, C, m, device=imp.device).view(1, -1, 1)
        flat = (topn + base).view(R, -1)
        rank.scatter_(1, flat, torch.zeros_like(flat))
        return rank
    assert ladder[-1] == (2, 4) and ladder[-2] == (1, 4), \
        "curriculum nhiều stage phải kết thúc ...,1:4,2:4"
    g4 = imp.view(R, -1, 4)
    top2 = g4.topk(2, dim=2).indices
    base = torch.arange(0, C, 4, device=imp.device).view(1, -1, 1)
    flat2 = (top2 + base).view(R, -1)
    rank.scatter_(1, flat2[:, 1::2], torch.full_like(flat2[:, 1::2], L - 1))
    rank.scatter_(1, flat2[:, 0::2], torch.full_like(flat2[:, 0::2], L - 2))
    win = flat2[:, 0::2].view(R, -1)
    for k in range(L - 3, -1, -1):
        pair = win.view(R, -1, 2)
        w_imp = imp.gather(1, pair.reshape(R, -1)).view(R, -1, 2)
        sel = w_imp.argmax(dim=2, keepdim=True)
        win = pair.gather(2, sel).squeeze(2)
        rank.scatter_(1, win, torch.full_like(win, k))
    return rank


class NestedQLinear(nn.Module):
    """Ternary N:M theo stage lồng nhau + scale học được trên lưới f8-g64 (khung TQ33/F0)
    + masked-STE (grad=0 trên ô pruned — vệ sinh bắt buộc cho curriculum, xem exp_aa G2)
    + highway columns (cột kênh massive-activation giữ FP, ~+0.05bpw)."""

    def __init__(self, lin, imp, ladder, stage=0, G=64, sdtype="f8", use_bias=True,
                 hw_cols=None):
        super().__init__()
        W0 = lin.weight.data
        R, C = W0.shape
        self.R, self.C, self.G = R, C, G
        assert C % G == 0 and C % ladder[0][1] == 0, f"C={C} phải chia hết G/M"
        self.ladder, self.sdtype = ladder, sdtype
        self.Wfp = nn.Parameter(W0.clone())
        self.bias = nn.Parameter(torch.zeros(R, dtype=W0.dtype, device=W0.device)) \
            if use_bias else None
        self.register_buffer("rank", nested_rank(imp, ladder))
        hw = torch.zeros(C, dtype=torch.bool, device=W0.device)
        if hw_cols is not None and len(hw_cols):
            hw[torch.as_tensor(hw_cols, device=W0.device)] = True
        self.register_buffer("hw", hw)
        self.register_buffer("act", torch.zeros_like(W0))
        self.register_buffer("gmask", torch.zeros_like(W0))
        self.raw_s = nn.Parameter(torch.zeros(R, C // G, 1, dtype=W0.dtype,
                                              device=W0.device))
        self._hook = self.Wfp.register_hook(lambda g: g * self.gmask)
        self.set_stage(stage)

    @torch.no_grad()
    def set_stage(self, k):
        self.stage = int(k)
        act = (self.rank <= k).to(self.Wfp.dtype)
        act[:, self.hw] = 0.0
        gm = act.clone()
        gm[:, self.hw] = 1.0
        self.act.copy_(act)
        self.gmask.copy_(gm)
        self.refit_scale()

    @torch.no_grad()
    def refit_scale(self):
        """Lloyd 3 vòng trên ô ACTIVE — gọi mỗi lần đổi stage (scale đóng băng trong stage,
        kiểu EfficientQAT/GEN4)."""
        Wv = (self.Wfp.data * self.act).view(self.R, -1, self.G)
        Av = self.act.view(self.R, -1, self.G)
        cnt = Av.sum(2, keepdim=True).clamp(min=1)
        s = (Wv.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
        for _ in range(3):
            t = torch.round(Wv / s).clamp(-1, 1) * Av
            num = (Wv * t).sum(2, keepdim=True)
            den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
            s = (num / den).abs().clamp(min=1e-6)
        self.raw_s.data = inv_softplus(s.float().clamp(min=4.1e-3)).to(self.Wfp.dtype)

    def _splane(self):
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        qs = to_f16(s) if self.sdtype == "f16" else to_f8(s)
        return (s + (qs - s).detach()).clamp(min=1e-6)

    def quant(self):
        s = self._splane()
        Wv = (self.Wfp * self.act).view(self.R, -1, self.G)
        t = (torch.round(Wv / s.detach()).clamp(-1, 1)
             * self.act.view(self.R, -1, self.G)).detach()
        return (t * s).reshape(self.R, self.C)

    def forward(self, x):
        q = self.quant()
        Wq = q + (self.Wfp - self.Wfp.detach())          # STE; hook lọc grad ô pruned
        Wq = torch.where(self.hw.view(1, -1), self.Wfp, Wq)   # highway: FP thật, grad thật
        return F.linear(x, Wq, self.bias)

    def bpw(self):
        n, m = self.ladder[self.stage]
        return nm_bpw(n, m, self.G, 16.0 if self.sdtype == "f16" else 8.0) \
            + 16.0 * self.hw.float().mean().item()


# ============================ highway detection (đo, không đoán) ============================
@torch.no_grad()
def detect_highway(model, tok, texts, dev, k=6, thresh=32.0, min_layers=0.6, seq=64):
    """Kênh massive-activation của KHÔNG GIAN HIDDEN sau norm (input q/k/v/gate/up):
    absmax per-channel per-layer; kênh là 'massive' nếu absmax > thresh × median(layer)
    ở ≥ min_layers tỉ lệ layer. Trả về ≤k kênh (có thể rỗng — 0.6B phải đo, không suy
    từ 30B; bài học 'phải đo lại' trong RESEARCH_TQ33_OUTLIER_FIX)."""
    H = model.config.hidden_size
    NL = len(model.model.layers)
    mx = torch.zeros(NL, H, device=dev)
    hooks = []

    def mk(li):
        def h(mod, inp):
            a = inp[0].detach().reshape(-1, inp[0].shape[-1]).abs().amax(0)
            mx[li] = torch.maximum(mx[li], a.float())
        return h
    for li, blk in enumerate(model.model.layers):
        hooks.append(blk.self_attn.q_proj.register_forward_pre_hook(mk(li)))
    for s in texts:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=seq).input_ids.to(dev)
        if ids.shape[1] >= 4:
            model(ids)
    for h in hooks:
        h.remove()
    med = mx.median(dim=1, keepdim=True).values.clamp(min=1e-6)
    frac = (mx > thresh * med).float().mean(0)
    cand = torch.nonzero(frac >= min_layers).flatten()
    if len(cand) > k:
        cand = cand[frac[cand].argsort(descending=True)[:k]]
    cols = sorted(cand.tolist())
    log(f"highway: {len(cols)} kênh {cols} (thresh {thresh}x, ≥{min_layers:.0%} layer)"
        if cols else f"highway: KHÔNG có kênh vượt {thresh}× ở ≥{min_layers:.0%} layer")
    return cols


# ============================ behavior battery (sinh THẬT + validator) ============================
BATTERY = [
    ("math", "Q: 7 + 15 = ?\nA:", "number", 22),
    ("math", "Q: 12 * 3 = ?\nA:", "number", 36),
    ("math", "Q: 100 - 37 = ?\nA:", "number", 63),
    ("math", "Q: A box has 12 apples. Tom takes 5. How many apples are left?\nA:", "number", 7),
    ("math", "Q: What is half of 26?\nA:", "number", 13),
    ("math", "Q: 3 * 4 + 2 = ?\nA:", "number", 14),
    ("fact", "Q: What is the capital of France?\nA:", "contains", ["paris"]),
    ("fact", "Hỏi: Thủ đô của Việt Nam là thành phố nào?\nĐáp:",
     "contains", ["hà nội", "ha noi", "hanoi"]),
    ("fact", "Q: How many days are in one week?\nA:", "contains", ["7", "seven"]),
    ("fact", "Q: 日本の首都はどこですか。\n答え:", "contains", ["東京", "tokyo"]),
    ("fact", "Q: What color is the sky on a clear day?\nA:", "contains", ["blue"]),
    ("fact", "Q: Con mèo kêu tiếng gì?\nĐáp:", "contains", ["meo", "meow"]),
    ("code", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return",
     "code_eval", ("add", (2, 3), 5)),
    ("code", "def square(x):\n    \"\"\"Return x squared.\"\"\"\n    return",
     "code_eval", ("square", (4,), 16)),
    ("code", "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n    return",
     "code_eval", ("is_even", (4,), True)),
    ("code", "def first_char(s):\n    \"\"\"Return the first character of s.\"\"\"\n    return",
     "code_eval", ("first_char", ("hello",), "h")),
    ("code", "def double(x):\n    \"\"\"Return 2 times x.\"\"\"\n    return",
     "code_eval", ("double", (5,), 10)),
    ("inst", "List exactly three fruits, one per line:\n1.", "lines3", None),
    ("inst", "Reply with one word only. The opposite of hot is:", "contains", ["cold"]),
    ("inst", "Complete the sequence: one, two, three, four,", "contains", ["five"]),
    ("inst", "続きを一つだけ: 一、二、三、", "contains", ["四"]),
    ("inst", "Điền tiếp: thứ hai, thứ ba, thứ", "contains", ["tư", "4"]),
]


def _first_number(txt):
    num = ""
    for ch in txt:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return int(num) if num else None


def _validate(kind, expect, prompt, out):
    low = out.lower()
    try:
        if kind == "number":
            return _first_number(out) == expect
        if kind == "contains":
            return any(e.lower() in low for e in expect)
        if kind == "lines3":
            body = out.split("\n\n")[0]
            items = [ln for ln in ("1." + body).split("\n") if ln.strip()]
            return 2 <= len([x for x in items if any(c.isalpha() for c in x)]) <= 4
        if kind == "code_eval":
            fname, fargs, fret = expect
            line = out.split("\n")[0].strip()
            src = prompt + " " + line if not prompt.rstrip().endswith(":") else prompt + line
            ns = {}
            exec(compile(src, "<bat>", "exec"), {"__builtins__": {}}, ns)   # noqa: S102
            return ns[fname](*fargs) == fret
    except Exception:
        return False
    return False


def _loopy(ids):
    seq = ids.tolist()
    if len(seq) < 12:
        return False
    bg = list(zip(seq, seq[1:]))
    if len(set(bg)) / max(len(bg), 1) < 0.35:
        return True
    for k in range(len(seq) - 12):
        g = tuple(seq[k:k + 4])
        if tuple(seq[k + 4:k + 8]) == g and tuple(seq[k + 8:k + 12]) == g:
            return True
    return False


@torch.no_grad()
def behavior_battery(model, tok, dev, temps=(0.0, 0.7), max_new=48, seed=0,
                     log_all=False, items=None):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bat = BATTERY[:items] if items else BATTERY
    res = {}
    for temp in temps:
        torch.manual_seed(seed)
        n_pass, n_loop, per_dom = 0, 0, {}
        transcripts = []
        for dom, prompt, kind, expect in bat:
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            kw = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            if temp > 0:
                kw.update(do_sample=True, temperature=temp, top_p=0.9)
            else:
                kw.update(do_sample=False)
            o = model.generate(ids, **kw)[0][ids.shape[1]:]
            txt = tok.decode(o, skip_special_tokens=True)
            ok = _validate(kind, expect, prompt, txt)
            lp = _loopy(o.cpu())
            n_pass += ok
            n_loop += lp
            d = per_dom.setdefault(dom, [0, 0])
            d[0] += ok
            d[1] += 1
            transcripts.append({"dom": dom, "ok": bool(ok), "loop": bool(lp),
                                "prompt": prompt[:40], "out": txt[:120]})
        key = f"t{temp}"
        res[key] = {"pass_rate": round(n_pass / len(bat), 3),
                    "loop_rate": round(n_loop / len(bat), 3),
                    "per_dom": {k: f"{v[0]}/{v[1]}" for k, v in per_dom.items()}}
        if log_all:
            res[key]["transcripts"] = transcripts
        log(f"  battery t={temp}: pass {n_pass}/{len(bat)}"
            f" loop {n_loop}/{len(bat)} | {res[key]['per_dom']}")
    return res


# ============================ rollout-KD ============================
def loop_entry(row, plen):
    """Vị trí ĐẦU TIÊN chuỗi rơi vào lặp 4-gram (hoặc None). Phát hiện từ exp_ac probe:
    một khi ngữ cảnh đã trong loop, CHÍNH TEACHER cũng dự đoán tiếp loop (induction) —
    self-KL tụt về ~0, gradient vô dụng + teacher 'đồng lõa'. Tín hiệu học nằm ở CỬA VÀO
    loop → giữ entry + vài token, cắt loss phần sau."""
    seq = row.tolist()
    for p in range(plen + 8, len(seq) - 4):
        if seq[p - 8:p - 4] == seq[p - 4:p] == seq[p:p + 4][:4]:
            return p
    return None


class RolloutBuffer:
    """Mỗi refresh: student TỰ SINH từ prompt pool; teacher chấm top-K + lseT (exp_w v2 —
    KL đúng cả khối đuôi). Buffer dùng lại refresh_every bước (semi-on-policy, kiểu GKD).
    Loss chỉ tính vùng tự sinh, CẮT sau điểm-vào-loop (xem loop_entry)."""

    def __init__(self, prompts_ids, plen, T_roll, topk, kd_temp, loopcut=True):
        self.pool = prompts_ids
        self.plen, self.T_roll, self.topk, self.T = plen, T_roll, topk, kd_temp
        self.loopcut = loopcut
        self.data = None
        self.stat_cut = 0.0

    @torch.no_grad()
    def refresh(self, model, teacher, batch, dev, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randint(0, self.pool.shape[0], (batch,), generator=g)
        pids = self.pool[idx].to(dev)
        gen = model.generate(pids, max_new_tokens=self.T_roll, do_sample=True,
                             temperature=1.0, top_p=0.95,
                             pad_token_id=model.config.eos_token_id)
        tl = teacher(gen).logits.float()
        v, ix = tl.topk(self.topk, dim=-1)
        lseT = (tl / self.T).logsumexp(-1)
        m = torch.zeros(gen.shape, device=dev)
        m[:, self.plen - 1:-1] = 1.0               # logits tại p chấm token p+1 tự sinh
        if self.loopcut:
            ncut = 0
            for b in range(gen.shape[0]):
                p = loop_entry(gen[b].cpu(), self.plen)
                if p is not None:
                    m[b, p + 2:] = 0.0             # giữ cửa-vào-loop + 2 token
                    ncut += 1
            self.stat_cut = ncut / gen.shape[0]
        self.data = {"ids": gen, "v": v / self.T, "ix": ix, "lseT": lseT, "m": m}

    def loss(self, model):
        d = self.data
        sl = model(d["ids"]).logits.float() / self.T
        s_lse = sl.logsumexp(-1, keepdim=True)
        s_at = sl.gather(-1, d["ix"]) - s_lse
        t_lp = d["v"] - d["lseT"].unsqueeze(-1)
        t_p = t_lp.exp()
        t_tail = (1.0 - t_p.sum(-1)).clamp(min=1e-5)
        s_tail = (1.0 - s_at.exp().sum(-1)).clamp(min=1e-5)
        kl = (t_p * (t_lp - s_at)).sum(-1) + t_tail * (t_tail.log() - s_tail.log())
        m = d["m"]
        return (kl * m).sum() / m.sum().clamp(min=1) * (self.T * self.T)


def build_rollout_prompts(tok, texts, plen, n, dev):
    """Prompt đúng plen token, KHÔNG pad (lọc đoạn đủ dài) — generate không cần mask."""
    out, i = [], 0
    while len(out) < n and i + 3 <= len(texts):
        ids = tok(" ".join(texts[i:i + 3]), return_tensors="pt").input_ids[0]
        if ids.shape[0] >= plen:
            out.append(ids[:plen])
        i += 3
    assert out, "không đủ đoạn dài làm rollout prompt"
    return torch.stack(out)


# ============================ S1 compact (sequential 2-pass, port exp_r) ============================
def s1_sequential(model, tok, calib_ids, wrapped, dev, passes=(70, 30), smoke=False):
    layers = model.model.layers
    NB = len(layers)
    ropes, H_fp = [], [[] for _ in range(NB + 1)]
    with torch.no_grad():
        for ids in calib_ids:
            h = model.model.embed_tokens(ids)
            pos = torch.arange(ids.shape[1], device=dev)[None]
            cos, sin = model.model.rotary_emb(h, pos)
            ropes.append((cos, sin))
            for b, blk in enumerate(layers):
                H_fp[b].append(h.clone())
                h = blk(h, position_embeddings=(cos, sin))
            H_fp[NB].append(h.clone())
    NC = len(calib_ids)
    eval_sub = list(range(0, NC, max(1, NC // 12)))[:12]
    t0 = time.time()
    for p_idx, base_steps in enumerate((4, 2) if smoke else passes):
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            f = HARD_BLOCKS.get(b, 0.75 if b in EASY_RANGE else 1.0)
            steps = max(2, int(round(base_steps * f)))
            mods = [w for (bb, _, _, w) in wrapped if bb == b]
            norm_ws = []
            for sub, name in NORM_PATHS:
                parent = getattr(blk, sub) if sub else blk
                norm_ws.append(getattr(parent, name).weight)
            for nw in norm_ws:
                nw.requires_grad_(True)
            opt = torch.optim.Adam([
                {"params": [w.Wfp for w in mods], "lr": 1e-3},
                {"params": [w.raw_s for w in mods], "lr": 5e-3},
                {"params": [w.bias for w in mods if w.bias is not None], "lr": 5e-4},
                {"params": norm_ws, "lr": 5e-4}])

            def eval_block():
                with torch.no_grad():
                    return sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]),
                                          H_fp[b + 1][s]).item() for s in eval_sub)
            best = eval_block()
            best_state = ([w.Wfp.detach().clone() for w in mods],
                          [w.raw_s.detach().clone() for w in mods],
                          [w.bias.detach().clone() for w in mods if w.bias is not None],
                          [nw.detach().clone() for nw in norm_ws])
            for step in range(steps):
                idx = random.sample(range(NC), min(6, NC))
                opt.zero_grad()
                loss = sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]),
                                      H_fp[b + 1][s]) for s in idx) / len(idx)
                loss.backward()
                opt.step()
                if step % 10 == 9 or step == steps - 1:
                    v = eval_block()
                    if v < best:
                        best = v
                        best_state = ([w.Wfp.detach().clone() for w in mods],
                                      [w.raw_s.detach().clone() for w in mods],
                                      [w.bias.detach().clone() for w in mods
                                       if w.bias is not None],
                                      [nw.detach().clone() for nw in norm_ws])
            with torch.no_grad():
                bi = 0
                for w, Wb, Sb in zip(mods, best_state[0], best_state[1]):
                    w.Wfp.data, w.raw_s.data = Wb, Sb
                    if w.bias is not None:
                        w.bias.data = best_state[2][bi]
                        bi += 1
                for nw, nb in zip(norm_ws, best_state[3]):
                    nw.data = nb
            for nw in norm_ws:
                nw.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"S1 pass {p_idx + 1} xong ({time.time() - t0:.0f}s)")
    del H_fp, H_q, ropes


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-glob", default=DEF_MODEL)
    ap.add_argument("--train-vi", default=DEF_VI)
    ap.add_argument("--train-ja", default=DEF_JA)
    ap.add_argument("--dev-vi", default=DEF_DVI)
    ap.add_argument("--dev-ja", default=DEF_DJA)
    ap.add_argument("--kd-en", default="")
    ap.add_argument("--kd-code", default="")
    ap.add_argument("--kd-zh", default="")
    ap.add_argument("--kd-math", default="")
    ap.add_argument("--train-sents", type=int, default=60_000)
    # curriculum
    ap.add_argument("--ladder", default="1:32,1:16,1:8,1:4,2:4")
    ap.add_argument("--stage-steps", default="400,400,600,800,1800",
                    help="số bước KD mỗi stage (tổng = ngân sách arm; arm 1 stage: 1 số)")
    # rollout-KD
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--roll-lambda", type=float, default=0.5)
    ap.add_argument("--roll-refresh", type=int, default=4)
    ap.add_argument("--roll-tokens", type=int, default=64)
    ap.add_argument("--roll-plen", type=int, default=24)
    ap.add_argument("--roll-topk", type=int, default=64)
    ap.add_argument("--roll-min-bpw", type=float, default=0.6,
                    help="chỉ bật rollout khi bpw stage ≥ mức này (student quá nát thì "
                         "rollout toàn rác, ưu tiên TF trước)")
    ap.add_argument("--roll-loopcut", type=int, default=1,
                    help="1 = cắt loss rollout sau điểm-vào-loop (teacher đồng lõa với "
                         "loop trong-ngữ-cảnh — đo thật ở exp_ac)")
    # highway
    ap.add_argument("--highway", type=int, default=0)
    ap.add_argument("--hw-k", type=int, default=6)
    ap.add_argument("--hw-thresh", type=float, default=32.0)
    # train recipe (GEN4-class)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--ce-w", type=float, default=0.1)
    ap.add_argument("--calib", type=int, default=60)
    ap.add_argument("--calib-seq", type=int, default=96)
    ap.add_argument("--perm-gauge", type=int, default=1)
    ap.add_argument("--s1-passes", type=int, default=2)
    ap.add_argument("--best-metric", default="geo6")
    ap.add_argument("--save-stage-ckpt", type=int, default=0)
    ap.add_argument("--save-ckpt", type=int, default=1)
    ap.add_argument("--battery-transcripts", type=int, default=1)
    ap.add_argument("--tag", default="exp_ab")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "exp_ab.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ladder = parse_ladder(args.ladder)
    stage_steps = [int(x) for x in args.stage_steps.split(",")]
    assert len(stage_steps) == len(ladder), "--stage-steps phải khớp --ladder"
    if args.smoke:
        stage_steps = [2] * len(ladder)
        args.calib, args.train_sents, args.batch, args.seq = 10, 64, 2, 48
        args.roll_tokens, args.roll_refresh = 12, 2
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    torch.set_num_threads(5)
    log(f"exp_ab: ladder={ladder} steps={stage_steps} rollout={args.rollout}"
        f" highway={args.highway} dev={dev}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = glob.glob(args.model_glob)[0]
    tok = AutoTokenizer.from_pretrained(mdir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()
    dev_vi = read_lines(args.dev_vi, 400)[-16:]
    dev_ja = read_lines(args.dev_ja, 400)[-16:]
    fp_vi = eval_ppl(model, tok, dev_vi, dev)
    log(f"FP32 vi {fp_vi:.1f}")

    # ---- gauge + calib mixwj + perm (đúng gia phả S1-v2/GEN4) ----
    apply_gauge(model)
    pg = eval_ppl(model, tok, dev_vi, dev)
    assert abs(pg - fp_vi) / fp_vi < 0.01, "gauge sai"
    vi_c = read_lines(args.dev_vi, 20)
    ja_c = read_lines(args.dev_ja, 32)
    k = 4 if args.smoke else None
    calib = (vi_c[:(k or 20)] + ja_c[:(k or 32)] + CAL_EN[:k] + CAL_CODE[:k]
             + CAL_ZH[:k] + CAL_MATH[:k] + CAL_CHAT[:k])
    random.Random(0).shuffle(calib)
    linears = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and "layers." in n]
    xn_acc, handles = {n: None for n, _ in linears}, []

    def mk(nm):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            xn_acc[nm] = v if xn_acc[nm] is None else xn_acc[nm] + v
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True,
                      max_length=args.calib_seq).input_ids.to(dev)
            if ids.shape[1] >= 4:
                calib_ids.append(ids)
                model(ids)
    for h in handles:
        h.remove()
    xnorm = {n: xn_acc[n].sqrt() for n in xn_acc}
    if args.perm_gauge:
        apply_perm_gauge(model, xnorm, 4)        # M=4 = nhóm cấu trúc của khung đích TQ33
        pp = eval_ppl(model, tok, dev_vi, dev)
        assert abs(pp - fp_vi) / fp_vi < 0.01, "perm-gauge sai"
        log(f"perm-gauge OK (FP {pp:.1f})")
    hw_cols = detect_highway(model, tok, calib[:24], dev, k=args.hw_k,
                             thresh=args.hw_thresh) if args.highway else []

    # ---- wrap NestedQLinear @stage0 + S1 compact ----
    orig = {n: m.weight.data.clone() for n, m in linears}
    wrapped = []
    for b, blk in enumerate(model.model.layers):
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            imp = orig[key].abs() * xnorm[key][None, :].clamp(min=1e-8)
            hw = hw_cols if name not in ("o_proj", "down_proj") else []
            w = NestedQLinear(lin, imp, ladder, stage=0, G=64, sdtype="f8",
                              use_bias=True, hw_cols=hw)
            setattr(parent, name, w)
            wrapped.append((b, parent, name, w))
    bpw0 = wrapped[0][3].bpw()
    log(f"wrap {len(wrapped)} linear @stage0 {ladder[0]} ({bpw0:.3f} bpw)")
    s1_sequential(model, tok, calib_ids, wrapped, dev,
                  passes=(100, 50, 30) if args.s1_passes >= 3 else (70, 30),
                  smoke=args.smoke)
    v_s1 = eval_ppl(model, tok, dev_vi, dev)
    log(f"S1 @stage0: PPL vi {v_s1:.1f}")

    # ---- teacher + data KD ----
    teacher = AutoModelForCausalLM.from_pretrained(
        mdir, dtype=(torch.bfloat16 if dev == "cuda" else torch.float32)).to(dev).eval()
    apply_gauge(teacher)     # cùng gauge; KHÔNG cần cùng perm (KD so logit, bất biến hoán vị)
    for p in teacher.parameters():
        p.requires_grad_(False)
    T = args.train_sents
    pool, miss = [], 0
    for kk, pth, n_ in (("en", args.kd_en, int(T * 0.15)), ("code", args.kd_code, int(T * 0.10)),
                        ("zh", args.kd_zh, int(T * 0.10)), ("math", args.kd_math, int(T * 0.05))):
        ls = read_lines(pth, n_) if (pth and os.path.exists(pth)) else []
        if kk == "code":
            ls = [s.replace("\\n", "\n").replace("\\\\", "\\") for s in ls]
        if not ls:
            miss += n_
        pool += ls
    tr_vi = read_lines(args.train_vi, int(T * 0.24) + miss - int(miss * 0.6))
    tr_ja = read_lines(args.train_ja, int(T * 0.36) + int(miss * 0.6))
    train_txt = tr_vi + tr_ja + pool
    random.Random(1).shuffle(train_txt)
    log(f"KD pool: {len(train_txt)} câu (vi {len(tr_vi)} ja {len(tr_ja)} extra {len(pool)})")
    roll_buf = None
    if args.rollout:
        rp = build_rollout_prompts(tok, train_txt[:3 * 512], args.roll_plen,
                                   256 if not args.smoke else 8, dev)
        roll_buf = RolloutBuffer(rp, args.roll_plen, args.roll_tokens,
                                 args.roll_topk, args.kd_temp,
                                 loopcut=bool(args.roll_loopcut))
        log(f"rollout prompts: {rp.shape[0]} × {args.roll_plen} tok"
            f" | loopcut={args.roll_loopcut}")

    # ---- trainable: W + bias (freeze scale+norm trong stage — GEN4/EfficientQAT) ----
    params_w = [w.Wfp for (_, _, _, w) in wrapped]
    params_b = [w.bias for (_, _, _, w) in wrapped if w.bias is not None]
    for p in params_w + params_b:
        p.requires_grad_(True)
    opt = torch.optim.Adam([{"params": params_w, "lr": args.lr},
                            {"params": params_b, "lr": args.lr}])
    probes = {"vi": dev_vi, "ja": dev_ja, "en": EN_EVAL, "code": CODE_EVAL,
              "zh": ZH_EVAL, "math": MATH_EVAL}

    def eval6():
        return {k: eval_ppl(model, tok, v, dev, max_tok=(160 if k == "code" else 96))
                for k, v in probes.items()}

    def geo(d):
        return math.exp(sum(math.log(max(x, 1e-9)) for x in d.values()) / len(d))

    p0 = eval6()
    best_score = geo(p0)
    best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
    log("step-0: " + " / ".join(f"{k} {v:.0f}" for k, v in p0.items())
        + f" | geo6 {best_score:.1f}")

    use_ac = dev == "cuda"
    gstep, nan_cnt = 0, 0
    stage_log = []
    for si, (n_, m_) in enumerate(ladder):
        if si > 0:
            for (_, _, _, w) in wrapped:
                w.set_stage(si)                      # grow mask lồng nhau + refit scale
            v_tr = eval_ppl(model, tok, dev_vi, dev)
            log(f"== stage {si} {n_}:{m_} ({wrapped[0][3].bpw():.3f} bpw)"
                f" | PPL vi sau grow: {v_tr:.1f}")
        steps = stage_steps[si]
        bpw_now = wrapped[0][3].bpw()
        roll_on = args.rollout and bpw_now >= args.roll_min_bpw
        WARM = max(1, min(100, steps // 10))
        eval_every = max(2, steps // 6)
        t1 = time.time()
        for st in range(steps):
            fac = min(1.0, (st + 1) / WARM)
            if st >= WARM:
                fac = 0.5 * (1 + math.cos(math.pi * (st - WARM) / max(1, steps - WARM)))
            for gr in opt.param_groups:
                gr["lr"] = args.lr * fac
            lam = 0.0
            if roll_on and st >= WARM:
                lam = args.roll_lambda * min(1.0, (st - WARM) / max(1, int(0.3 * steps)))
                if st % args.roll_refresh == 0 or roll_buf.data is None:
                    roll_buf.refresh(model, teacher, args.batch, dev, seed=gstep)
            batch = random.sample(train_txt, min(args.batch, len(train_txt)))
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.seq)
            ids = enc.input_ids.to(dev)
            am = enc.attention_mask.to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                with torch.no_grad():
                    tl = teacher(ids, attention_mask=am).logits
                sl = model(ids, attention_mask=am).logits
            Tt = args.kd_temp
            lt = F.log_softmax(tl.float() / Tt, -1)
            ls = F.log_softmax(sl.float() / Tt, -1)
            kl_tok = (lt.exp() * (lt - ls)).sum(-1)
            mm = am.float()
            loss_tf = (kl_tok * mm).sum() / mm.sum().clamp(min=1) * (Tt * Tt)
            if args.ce_w > 0:
                m2 = (am[:, 1:] * am[:, :-1]).float()
                ce = F.cross_entropy(sl.float()[:, :-1].transpose(1, 2), ids[:, 1:],
                                     reduction="none")
                loss_tf = loss_tf + args.ce_w * (ce * m2).sum() / m2.sum().clamp(min=1)
            loss = loss_tf
            if lam > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                    lr_roll = roll_buf.loss(model)
                loss = (1 - lam) * loss_tf + lam * lr_roll
            if not torch.isfinite(loss):
                nan_cnt += 1
                opt.zero_grad()
                if nan_cnt >= 3:
                    for gr in opt.param_groups:
                        gr["lr"] *= 0.5
                    nan_cnt = 0
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_w + params_b, 1.0)
            opt.step()
            gstep += 1
            if st % eval_every == eval_every - 1 or st == steps - 1:
                pe = eval6()
                sc = geo(pe)
                lam_s = f" λ{lam:.2f} cut{roll_buf.stat_cut:.0%}" if roll_on else ""
                log(f"  s{si} {st + 1}/{steps}: loss {loss.item():.3f}{lam_s} | "
                    + " / ".join(f"{k} {v:.0f}" for k, v in pe.items())
                    + f" | geo6 {sc:.1f} ({time.time() - t1:.0f}s)")
                if sc < best_score:
                    best_score = sc
                    best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
        # ---- hết stage: đo đầy đủ ----
        pe = eval6()
        nval = 8 if args.smoke else 100
        big_vi = eval_ppl(model, tok, read_lines(args.dev_vi, 200)[-nval:], dev)
        bat = behavior_battery(model, tok, dev,
                               temps=((0.0,) if args.smoke else (0.0, 0.7)),
                               max_new=(12 if args.smoke else 48),
                               items=(4 if args.smoke else None))
        stage_log.append({"stage": f"{n_}:{m_}", "bpw": round(bpw_now, 3),
                          "ppl": {k: round(v, 1) for k, v in pe.items()},
                          "geo6": round(geo(pe), 1), "val100_vi": round(big_vi, 1),
                          "battery": {k: {kk: vv for kk, vv in v.items()
                                          if kk != "transcripts"}
                                      for k, v in bat.items()},
                          "rollout_on": bool(roll_on)})
        log(f"== stage {si} XONG: geo6 {geo(pe):.1f} | val100 {big_vi:.1f}")
        if args.save_stage_ckpt:
            torch.save({"state_dict": {k: v.half().cpu()
                                       for k, v in model.state_dict().items()},
                        "meta": {"tag": args.tag, "stage": f"{n_}:{m_}",
                                 "bpw": bpw_now}},
                       args.out.replace(".pt", f"_s{si}.pt"))

    # ---- kết: nạp best, bake, battery đầy đủ + divergence ----
    model.load_state_dict(best_sd)
    pe = eval6()
    nval = 8 if args.smoke else 100
    big_vi = eval_ppl(model, tok, read_lines(args.dev_vi, 200)[-nval:], dev)
    bat_best = behavior_battery(model, tok, dev,
                                temps=((0.0,) if args.smoke else (0.0, 0.7)),
                                max_new=(12 if args.smoke else 48),
                                log_all=bool(args.battery_transcripts),
                                items=(4 if args.smoke else None))
    div = None
    try:
        from exp_ac_divergence_probe import BUCKETS, kl_bucketed, build_prompt_texts
        texts = build_prompt_texts(args.dev_vi, n_vi=(2 if args.smoke else 8), n_en=1)
        enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=24)
        pids = enc.input_ids.to(dev)
        torch.manual_seed(0)
        gen = model.generate(pids, attention_mask=enc.attention_mask.to(dev),
                             max_new_tokens=(12 if args.smoke else 48), do_sample=True,
                             temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
        am_g = torch.ones_like(gen)
        am_g[:, :24] = enc.attention_mask
        kl_self = kl_bucketed(teacher, model, gen, am_g, 24, dev)
        encf = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
                   max_length=24 + (12 if args.smoke else 48))
        kl_tf = kl_bucketed(teacher, model, encf.input_ids.to(dev),
                            encf.attention_mask.to(dev), 24, dev)
        div = {"kl_self": kl_self, "kl_tf": kl_tf, "buckets": list(map(list, BUCKETS))}
        log(f"divergence: self {kl_self} | tf {kl_tf}")
    except Exception as e:
        log(f"divergence probe lỗi (không chặn): {type(e).__name__}: {e}")

    with torch.no_grad():
        for b, parent, name, w in wrapped:
            nl = nn.Linear(w.C, w.R, bias=True).to(dev)
            Wq = w.quant()
            Wq[:, w.hw] = w.Wfp.data[:, w.hw]
            nl.weight.data = Wq.detach()
            nl.bias.data = w.bias.detach()
            setattr(parent, name, nl)
    log("=> EXP AB best: " + " / ".join(f"{k} {v:.0f}" for k, v in pe.items())
        + f" | geo6 {geo(pe):.1f} | val100 {big_vi:.1f}")
    if args.save_ckpt:
        torch.save({"state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
                    "meta": {"tag": args.tag, "ladder": args.ladder,
                             "rollout": args.rollout, "highway": hw_cols,
                             "bpw_final": wrapped[0][3].bpw(), "geo6": geo(pe),
                             "val100_vi": big_vi}}, args.out)
        log(f"đã lưu {args.out}")
    rj = os.path.join(OUT_DIR, "exp_ab_results.json")
    old = {}
    if os.path.exists(rj):
        try:
            with io.open(rj, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = {}
    old[args.tag] = {"ladder": args.ladder, "stage_steps": stage_steps,
                     "rollout": args.rollout, "roll_lambda": args.roll_lambda,
                     "highway_cols": hw_cols, "s1_vi": v_s1, "fp_vi": fp_vi,
                     "stages": stage_log,
                     "final": {"ppl": {k: round(v, 1) for k, v in pe.items()},
                               "geo6": round(geo(pe), 1), "val100_vi": round(big_vi, 1),
                               "battery": bat_best, "divergence": div}}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    log(f"đã ghi {rj} (key={args.tag})")


if __name__ == "__main__":
    main()
