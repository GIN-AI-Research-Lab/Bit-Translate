# -*- coding: utf-8 -*-
"""
Exp R — QAT-LITE END-TO-END (nấc cuối của cái núm PTQ->QAT). Thiết kế cho MÁY A (3060 Ti 8GB),
chạy được cả CPU (chậm) và có --smoke để kiểm end-to-end trước khi bàn giao.

2 GIAI ĐOẠN:
  S1  Dựng lại trạng thái kỷ lục (công thức exp_q trên N3: gauge + Wanda + sequential 2-pass
      + scale/bias học được + norm co-tune + budget). GPU: ~10-15 phút.
  S2  E2E KD-STE: mở khóa TOÀN BỘ 196 wrapper (Wfp+scale+bias) + mọi norm, tối ưu
      KL(teacher_FP || student_ternary) trên text train, STE xuyên suốt. Đây là "train" thật
      nhưng nhẹ: ~1500 step x batch 8 x 128 tok ~ 1.5M token, ~30-60 phút trên 3060 Ti.
      VRAM (bf16 master + Adam bf16 + teacher bf16): ~5.5GB — vừa 8GB.

Kỳ vọng ghi trước: 400 (vi, mốc exp_q N3) -> vùng 150-300. KHÔNG hứa chạm FP 69.
Persist: lưu state_dict đã bake + kết quả (vá lỗ hổng "chưa từng lưu model").

Chạy máy A (WSL2):  python exp_r_qat_lite.py --device cuda --steps 1500 \
    --train-vi <path>/train.vi --train-ja <path>/train.ja --dev-vi <path>/dev.vi --dev-ja <path>/dev.ja
Smoke máy B:        python exp_r_qat_lite.py --smoke
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

DEF_MODEL = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
DEF_VI = r"D:\Bit-Translate-data\clean_v7g\train.vi"
DEF_JA = r"D:\Bit-Translate-data\clean_v7g\train.ja"
DEF_DVI = r"D:\Bit-Translate-data\clean_v7g\dev.vi"
DEF_DJA = r"D:\Bit-Translate-data\clean_v7g\dev.ja"
OUT_DIR = os.environ.get("EXPR_OUT_DIR", r"e:\Bit-Translate\eval\lowbit_ptq")

LIN_PATHS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
             ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]

# Cổng eval đa miền (quan sát catastrophic forgetting — KHÔNG dùng để chọn best):
# bộ câu cố định để so TƯƠNG ĐỐI trước/sau train, không phải benchmark tuyệt đối.
EN_EVAL = [
    "The industrial revolution transformed the way people lived and worked across Europe.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The committee announced that the annual budget would be reviewed next quarter.",
    "Despite heavy rain, the marathon continued as scheduled through the city center.",
    "Quantum computers exploit superposition to perform certain calculations faster.",
    "The novel explores themes of memory, loss, and the passage of time.",
    "Scientists discovered a new species of deep-sea fish near the volcanic vents.",
    "The central bank raised interest rates to curb rising inflation.",
    "Ancient trade routes connected distant civilizations across mountains and deserts.",
    "The software update includes security patches and performance improvements.",
    "Renewable energy sources now account for a growing share of electricity production.",
    "The museum's new exhibition features artifacts from the Bronze Age.",
]
CODE_EVAL = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "for i in range(10):\n    if i % 2 == 0:\n        print(f'even: {i}')",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop()",
    "import json\nwith open('config.json') as f:\n    config = json.load(f)\nprint(config.get('name', 'default'))",
    "const result = numbers.filter(x => x > 0).map(x => x * 2).reduce((a, b) => a + b, 0);",
    "SELECT name, COUNT(*) as total FROM orders GROUP BY name HAVING COUNT(*) > 5 ORDER BY total DESC;",
    "try:\n    value = int(user_input)\nexcept ValueError:\n    value = 0\nfinally:\n    print(value)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])",
]
NORM_PATHS = [("", "input_layernorm"), ("", "post_attention_layernorm"),
              ("self_attn", "q_norm"), ("self_attn", "k_norm")]
HARD_BLOCKS = {2: 2.0, 26: 2.0, 27: 2.0, 0: 1.5, 1: 1.5, 3: 1.25}
EASY_RANGE = set(range(8, 21))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_lines(path, n, skip=0):
    out = []
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f):
            if i < skip:
                continue
            ln = ln.strip()
            if ln:
                out.append(ln)
            if len(out) >= n:
                break
    return out


def to_f8(s):
    try:
        return s.to(torch.float8_e4m3fn).to(s.dtype)
    except Exception:
        sign = torch.sign(s)
        a = s.abs().clamp(min=2 ** -9)
        e = torch.floor(torch.log2(a))
        return sign * (2 ** e) * (torch.round(a / (2 ** e) * 8) / 8)


def wanda_nm_mask(W, xnorm, N, M):
    R, C = W.shape
    imp = W.abs() * xnorm[None, :].clamp(min=1e-8)
    pad = (M - C % M) % M
    A = F.pad(imp, (0, pad)) if pad else imp
    g = A.view(R, -1, M)
    kth = g.kthvalue(M - N + 1, dim=2, keepdim=True).values
    return (g >= kth).view(R, -1)[:, :C]


def inv_softplus(y):
    return y + torch.log(-torch.expm1(-y))


class LearnQLinear(nn.Module):
    def __init__(self, lin, mask, G=64):
        super().__init__()
        W0 = lin.weight.data
        self.Wfp = nn.Parameter(W0.clone())
        self.bias = nn.Parameter(torch.zeros(W0.shape[0], dtype=W0.dtype, device=W0.device))
        R, C = W0.shape
        self.R, self.C, self.G = R, C, G
        self.pad = (G - C % G) % G
        self.register_buffer("maskf", mask.to(W0.dtype))
        with torch.no_grad():
            Wv, Mv = self._views(W0)
            Wm = Wv * Mv
            cnt = Mv.sum(2, keepdim=True).clamp(min=1)
            s = (Wm.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
            for _ in range(3):
                t = torch.round(Wm / s).clamp(-1, 1) * Mv
                num = (Wm * t).sum(2, keepdim=True)
                den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
                s = (num / den).abs().clamp(min=1e-6)
        self.raw_s = nn.Parameter(inv_softplus(s.float()).to(W0.dtype))

    def _views(self, W):
        Wp = F.pad(W, (0, self.pad)) if self.pad else W
        Mp = F.pad(self.maskf, (0, self.pad)) if self.pad else self.maskf
        return Wp.view(self.R, -1, self.G), Mp.view(self.R, -1, self.G)

    def quant(self):
        s = F.softplus(self.raw_s).clamp(min=4e-3)
        s_q = s + (to_f8(s) - s).detach()
        s_q = s_q.clamp(min=1e-6)
        Wv, Mv = self._views(self.Wfp)
        t = (torch.round((Wv * Mv) / s_q.detach()).clamp(-1, 1) * Mv).detach()
        return (t * s_q).reshape(self.R, -1)[:, :self.C]

    def forward(self, x):
        q = self.quant()
        Wq = q + self.Wfp - self.Wfp.detach()
        return F.linear(x, Wq, self.bias)


@torch.no_grad()
def apply_gauge(model):
    cfg = model.config
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    rep = n_q // n_kv
    for blk in model.model.layers:
        Wd = blk.mlp.down_proj.weight.data
        Wu = blk.mlp.up_proj.weight.data
        c = Wd.abs().mean(dim=0).clamp(min=1e-8)
        d = c / torch.exp(torch.log(c).mean())
        Wu.mul_(d[:, None])
        Wd.div_(d[None, :])
        Wv = blk.self_attn.v_proj.weight.data
        Wo = blk.self_attn.o_proj.weight.data
        co = Wo.abs().mean(dim=0).clamp(min=1e-8).view(n_q, hd)
        m = torch.ones(n_kv * hd, device=Wv.device, dtype=Wv.dtype)
        for kv in range(n_kv):
            qs = [kv * rep + r for r in range(rep)]
            m[kv * hd:(kv + 1) * hd] = torch.exp(sum(torch.log(co[q]) for q in qs) / rep)
        m = m / torch.exp(torch.log(m).mean())
        Wv.mul_(m[:, None])
        for kv in range(n_kv):
            for r in range(rep):
                q = kv * rep + r
                Wo[:, q * hd:(q + 1) * hd] /= m[kv * hd:(kv + 1) * hd][None, :]


@torch.no_grad()
def eval_ppl(model, tok, lines, dev, max_tok=96):
    nll, ntok = 0.0, 0
    for s in lines:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=max_tok).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        nll += model(ids, labels=ids).loss.float().item() * (ids.shape[1] - 1)
        ntok += ids.shape[1] - 1
    return float(np.exp(nll / max(ntok, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--model-glob", default=DEF_MODEL)
    ap.add_argument("--train-vi", default=DEF_VI)
    ap.add_argument("--train-ja", default=DEF_JA)
    ap.add_argument("--dev-vi", default=DEF_DVI)
    ap.add_argument("--dev-ja", default=DEF_DJA)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--calib", type=int, default=60, help="số câu cho S1 (sequential)")
    ap.add_argument("--train-sents", type=int, default=8000, help="số câu cho S2 (KD)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-skip", type=int, default=200_000, help="số dòng bỏ qua đầu file train")
    ap.add_argument("--freeze-scales", type=int, default=0, help="1 = đóng băng raw_s ở S2 (kiểu EfficientQAT)")
    ap.add_argument("--freeze-norms", type=int, default=0, help="1 = đóng băng norm ở S2")
    ap.add_argument("--nm-n", type=int, default=2, help="N của mask N:M (2:4=1.56bpw, 1:4=1.02, 1:8=0.70, 1:10=0.62)")
    ap.add_argument("--nm-m", type=int, default=4)
    ap.add_argument("--fast", type=int, default=0, help="1 = S2 batch thật + bf16 autocast (v3)")
    # v4 — gói train-polish (mặc định tắt để giữ so sánh được với v3)
    ap.add_argument("--cosine", type=int, default=0, help="1 = cosine decay sau warmup")
    ap.add_argument("--kd-temp", type=float, default=1.0, help="nhiệt độ KD (2.0 = dark knowledge)")
    ap.add_argument("--ce-w", type=float, default=0.0, help="trọng số CE trộn thêm (vd 0.1)")
    ap.add_argument("--ema", type=float, default=0.0, help="decay EMA (vd 0.999); 0 = tắt")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "qat_lite_n3.pt"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.calib, args.train_sents, args.batch = 6, 12, 64, 2
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(5)
    log(f"device={dev} steps={args.steps} smoke={args.smoke}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdir = glob.glob(args.model_glob)[0]
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float32).to(dev).eval()

    dev_vi = read_lines(args.dev_vi, 400)[-16:]
    dev_ja = read_lines(args.dev_ja, 400)[-16:]
    ppl_fp = (eval_ppl(model, tok, dev_vi, dev), eval_ppl(model, tok, dev_ja, dev))
    log(f"FP32: vi {ppl_fp[0]:.1f} / ja {ppl_fp[1]:.1f}")

    # ===== S1: gauge + sanity =====
    apply_gauge(model)
    ppl_g = eval_ppl(model, tok, dev_vi, dev)
    log(f"FP sau gauge: vi {ppl_g:.1f}")
    if abs(ppl_g - ppl_fp[0]) / ppl_fp[0] > 0.01:
        log("!!! gauge sai — abort")
        sys.exit(1)

    vi_c = read_lines(args.dev_vi, args.calib // 2)
    ja_c = read_lines(args.dev_ja, args.calib // 2)
    calib = [x for pr in zip(vi_c, ja_c) for x in pr]
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "layers." in n]
    xn_acc, handles = {n: None for n, _ in linears}, []
    def mk(nm_):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            v = x.pow(2).sum(0)
            xn_acc[nm_] = v if xn_acc[nm_] is None else xn_acc[nm_] + v
        return h
    for n, m in linears:
        handles.append(m.register_forward_pre_hook(mk(n)))
    calib_ids = []
    with torch.no_grad():
        for s in calib:
            ids = tok(s, return_tensors="pt", truncation=True, max_length=96).input_ids.to(dev)
            if ids.shape[1] >= 4:
                calib_ids.append(ids)
                model(ids)
    for h in handles:
        h.remove()
    xnorm = {n: xn_acc[n].sqrt() for n in xn_acc}
    orig = {n: m.weight.data.clone() for n, m in linears}

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

    wrapped = []
    for b, blk in enumerate(layers):
        for sub, name in LIN_PATHS:
            parent = getattr(blk, sub)
            lin = getattr(parent, name)
            key = f"model.layers.{b}.{sub}.{name}"
            w = LearnQLinear(lin, wanda_nm_mask(orig[key], xnorm[key], args.nm_n, args.nm_m))
            setattr(parent, name, w)
            wrapped.append((b, parent, name, lin, w))

    t0 = time.time()
    pass_steps = (6, 4) if args.smoke else (70, 30)
    for p_idx, base_steps in enumerate(pass_steps):
        H_q = [H_fp[0][s].clone() for s in range(NC)]
        for b, blk in enumerate(layers):
            f = HARD_BLOCKS.get(b, 0.75 if b in EASY_RANGE else 1.0)
            steps = max(4, int(round(base_steps * f)))
            mods = [w for (bb, _, _, _, w) in wrapped if bb == b]
            norm_ws = []
            for sub, name in NORM_PATHS:
                parent = getattr(blk, sub) if sub else blk
                norm_ws.append(getattr(parent, name).weight)
            for nw_ in norm_ws:
                nw_.requires_grad_(True)
            opt = torch.optim.Adam([
                {"params": [w.Wfp for w in mods], "lr": 1e-3},
                {"params": [w.raw_s for w in mods], "lr": 5e-3},
                {"params": [w.bias for w in mods], "lr": 5e-4},
                {"params": norm_ws, "lr": 5e-4},
            ])

            def eval_block():
                with torch.no_grad():
                    return sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]),
                                          H_fp[b + 1][s]).item() for s in eval_sub)

            best = eval_block()
            best_state = ([w.Wfp.detach().clone() for w in mods],
                          [w.raw_s.detach().clone() for w in mods],
                          [w.bias.detach().clone() for w in mods],
                          [nw_.detach().clone() for nw_ in norm_ws])
            for step in range(steps):
                idx = random.sample(range(NC), min(6, NC))
                opt.zero_grad()
                loss = sum(F.mse_loss(blk(H_q[s], position_embeddings=ropes[s]), H_fp[b + 1][s])
                           for s in idx) / len(idx)
                loss.backward()
                opt.step()
                if step % 10 == 9 or step == steps - 1:
                    v = eval_block()
                    if v < best:
                        best = v
                        best_state = ([w.Wfp.detach().clone() for w in mods],
                                      [w.raw_s.detach().clone() for w in mods],
                                      [w.bias.detach().clone() for w in mods],
                                      [nw_.detach().clone() for nw_ in norm_ws])
            with torch.no_grad():
                for w, Wb, Sb, Bb in zip(mods, best_state[0], best_state[1], best_state[2]):
                    w.Wfp.data, w.raw_s.data, w.bias.data = Wb, Sb, Bb
                for nw_, nb_ in zip(norm_ws, best_state[3]):
                    nw_.data = nb_
            for nw_ in norm_ws:
                nw_.requires_grad_(False)
            with torch.no_grad():
                for s in range(NC):
                    H_q[s] = blk(H_q[s], position_embeddings=ropes[s]).detach()
        log(f"S1 pass {p_idx+1} xong ({time.time()-t0:.0f}s)")
    del H_fp, H_q, ropes
    pv = eval_ppl(model, tok, dev_vi, dev)
    log(f"S1 (dựng lại ~exp_q): PPL vi {pv:.1f}  (mốc lab: 400.1)")

    # ===== S2: E2E KD-STE =====
    log("S2: nạp teacher FP...")
    teacher = AutoModelForCausalLM.from_pretrained(
        mdir, dtype=(torch.bfloat16 if dev == "cuda" else torch.float32)).to(dev).eval()
    apply_gauge(teacher)  # teacher cùng gauge (tương đương chính xác, giữ hidden khớp)
    for p in teacher.parameters():
        p.requires_grad_(False)

    tr_vi = read_lines(args.train_vi, args.train_sents // 2, skip=args.train_skip)
    tr_ja = read_lines(args.train_ja, args.train_sents // 2, skip=args.train_skip)
    train_txt = [x for pr in zip(tr_vi, tr_ja) for x in pr]
    log(f"S2 data: {len(train_txt)} câu")

    params_w = [w.Wfp for (_, _, _, _, w) in wrapped]
    params_s = [w.raw_s for (_, _, _, _, w) in wrapped]
    params_b = [w.bias for (_, _, _, _, w) in wrapped]
    norm_all = []
    for b, blk in enumerate(layers):
        for sub, name in NORM_PATHS:
            parent = getattr(blk, sub) if sub else blk
            norm_all.append(getattr(parent, name).weight)
    norm_all.append(model.model.norm.weight)
    groups = [{"params": params_w, "lr": args.lr}, {"params": params_b, "lr": args.lr}]
    trainable = params_w + params_b
    if not args.freeze_scales:
        groups.append({"params": params_s, "lr": args.lr * 2})
        trainable += params_s
    if not args.freeze_norms:
        groups.append({"params": norm_all, "lr": args.lr * 0.5})
        trainable += norm_all
    for p in trainable:
        p.requires_grad_(True)
    log(f"S2 trainable: W+bias{' +scales' if not args.freeze_scales else ''}"
        f"{' +norms' if not args.freeze_norms else ''} | lr {args.lr}")
    opt = torch.optim.Adam(groups)
    # regression guard step-0: S2 không bao giờ được phép bàn giao trạng thái tệ hơn S1.
    # Best theo GEO-MEAN(vi, ja) — tránh hy sinh ja để đẹp vi (bài học run 1: ja nổ 145k mà vi hồi).
    v0 = eval_ppl(model, tok, dev_vi, dev)
    j0 = eval_ppl(model, tok, dev_ja, dev)
    e0 = eval_ppl(model, tok, EN_EVAL, dev)
    c0 = eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)
    best_score = math.sqrt(v0 * j0)
    best_vi = v0
    best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
    log(f"S2 step-0 (mốc S1): PPL vi {v0:.1f} / ja {j0:.1f} | en {e0:.1f} / code {c0:.1f}")
    nan_cnt = 0
    eval_every = max(2, args.steps // 12)
    base_lrs = [g["lr"] for g in opt.param_groups]
    WARMUP = max(1, min(100, args.steps // 10))
    ema = [p.detach().clone() for p in trainable] if args.ema > 0 else None

    def ema_swap():
        for p, e in zip(trainable, ema):
            tmp = p.data.clone()
            p.data.copy_(e)
            e.copy_(tmp)
    t1 = time.time()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_ac = args.fast and dev == "cuda"
    for step in range(args.steps):
        fac = min(1.0, (step + 1) / WARMUP)
        if args.cosine and step >= WARMUP:
            prog = (step - WARMUP) / max(1, args.steps - WARMUP)
            fac = 0.5 * (1 + math.cos(math.pi * prog))
        for g, bl in zip(opt.param_groups, base_lrs):
            g["lr"] = bl * fac
        batch = random.sample(train_txt, min(args.batch, len(train_txt)))
        opt.zero_grad()
        loss_acc = 0.0
        if args.fast:
            # v3: batch THẬT (pad + attention_mask), KL chỉ tính trên token thật, bf16 autocast
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.seq)
            ids = enc.input_ids.to(dev)
            am = enc.attention_mask.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                with torch.no_grad():
                    tl = teacher(ids, attention_mask=am).logits
                sl = model(ids, attention_mask=am).logits
            T = args.kd_temp
            lt = F.log_softmax(tl.float() / T, -1)
            ls = F.log_softmax(sl.float() / T, -1)
            kl_tok = (lt.exp() * (lt - ls)).sum(-1)          # [B, T]
            m = am.float()
            loss = (kl_tok * m).sum() / m.sum().clamp(min=1) * (T * T)
            if args.ce_w > 0:                                 # v4: neo đuôi phân bố bằng CE thật
                m2 = (am[:, 1:] * am[:, :-1]).float()
                ce_tok = F.cross_entropy(sl.float()[:, :-1].transpose(1, 2), ids[:, 1:],
                                         reduction="none")
                loss = loss + args.ce_w * (ce_tok * m2).sum() / m2.sum().clamp(min=1)
            loss.backward()
            loss_acc = loss.item()
        else:
            for s in batch:
                ids = tok(s, return_tensors="pt", truncation=True, max_length=args.seq).input_ids.to(dev)
                if ids.shape[1] < 4:
                    continue
                with torch.no_grad():
                    tl = teacher(ids).logits.float()
                sl = model(ids).logits.float()
                # FIX 02/08: chia theo SỐ TOKEN (batchmean chia theo batch=1 -> loss to gấp ~seq_len)
                loss = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1),
                                log_target=True, reduction="sum") / ids.shape[1]
                (loss / len(batch)).backward()
                loss_acc += loss.item() / len(batch)
        if not math.isfinite(loss_acc):
            nan_cnt += 1
            log(f"  step {step}: loss NaN/inf — bỏ step ({nan_cnt})")
            opt.zero_grad()
            if nan_cnt >= 3:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                nan_cnt = 0
            continue
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for p, e in zip(trainable, ema):
                    e.mul_(args.ema).add_(p.data, alpha=1 - args.ema)
        if step % eval_every == eval_every - 1 or step == args.steps - 1:
            if ema is not None:
                ema_swap()                      # eval + chọn best trên trọng số EMA (mượt STE)
            v = eval_ppl(model, tok, dev_vi, dev)
            j = eval_ppl(model, tok, dev_ja, dev)
            en = eval_ppl(model, tok, EN_EVAL, dev)
            cd = eval_ppl(model, tok, CODE_EVAL, dev, max_tok=160)
            sc = math.sqrt(v * j)
            log(f"  step {step+1}/{args.steps}: KL/token {loss_acc:.4f} | vi {v:.1f} / ja {j:.1f} | geo {sc:.1f} | en {en:.1f} / code {cd:.1f} ({time.time()-t1:.0f}s)")
            if sc < best_score:
                best_score = sc
                best_vi = v
                best_sd = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if ema is not None:
                ema_swap()
    if best_sd is not None:
        model.load_state_dict(best_sd)

    # bake + persist + eval cuối
    with torch.no_grad():
        for b, parent, name, lin, w in wrapped:
            new_lin = nn.Linear(w.C, w.R, bias=True).to(dev)
            new_lin.weight.data = w.quant().detach()
            new_lin.bias.data = w.bias.detach()
            setattr(parent, name, new_lin)
    pv, pj = eval_ppl(model, tok, dev_vi, dev), eval_ppl(model, tok, dev_ja, dev)
    big_vi = eval_ppl(model, tok, read_lines(args.dev_vi, 200)[-100:], dev)
    log(f"=> EXP R QAT-lite: PPL vi {pv:.1f} / ja {pj:.1f} | validation 100 câu vi: {big_vi:.1f}")
    torch.save({"state_dict": {k: v.half().cpu() for k, v in model.state_dict().items()},
                "meta": {"config": "N3+gauge+QATlite", "bpw": 1.566, "ppl_vi": pv, "ppl_ja": pj,
                         "ppl_vi_100c": big_vi, "steps": args.steps}}, args.out)
    log(f"Đã lưu model bake: {args.out}")
    rj = os.path.join(OUT_DIR, "exp_r_results.json")
    out = {}
    if os.path.exists(rj):
        try:
            with io.open(rj, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception:
            out = {}
    out[f"exp_r[steps={args.steps},smoke={args.smoke}]"] = {
        "bpw": 1.566, "ppl_vi": pv, "ppl_ja": pj, "ppl_vi_100c": big_vi,
        "s1_vi": None, "fp_vi": ppl_fp[0], "fp_ja": ppl_fp[1]}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=" * 72)
    print(f"EXP R — QAT-lite e2e (N3 1.56bpw + gauge)  [mốc exp_q: vi 400.1]")
    print(f"  sau QAT-lite: vi {pv:10.1f}  ja {pj:10.1f}  | 100 câu vi: {big_vi:.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
