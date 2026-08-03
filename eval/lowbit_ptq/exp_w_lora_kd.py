# -*- coding: utf-8 -*-
"""
Exp W — LoRA-KD cho model LỚN đã nén ternary (bước (b) sau exp_v S1).

Nguyên lý: student = ckpt bake ternary (exp_v) ĐÓNG BĂNG + LoRA r nhỏ bf16 trên mọi linear
(≈ absorber activation-fit — đường exp_p, nay ở dạng đúng); teacher KHÔNG cần trong RAM:
pha T chạy teacher gốc 1 lượt, cache top-K logits (~1GB) lên đĩa; pha S train LoRA bằng
KL(topk) + CE, gradient checkpointing → vừa 1×A100-80 cho 30B.

Bit accounting: LoRA r=8 trên 30B ≈ +0.016 bpw (không tính teacher — vứt sau pha T).

Chạy (Modal, một container 2 pha):  python exp_w_lora_kd.py --ckpt /vol/out/expv_...pt ...
Smoke: --model-id Qwen/Qwen3-0.6B --smoke (ckpt bỏ trống -> tự bake nhanh? KHÔNG — smoke
dùng chính model gốc làm "student" để test đường ống, chấp nhận KL~0).
"""
import argparse
import io
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
random.seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root")
from exp_r_qat_lite import (CODE_EVAL, EN_EVAL, MATH_EVAL, ZH_EVAL,  # noqa: E402
                            read_lines)

OUT_DIR = os.environ.get("EXPR_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class LoRALinear(nn.Module):
    """Linear đóng băng + nhánh LoRA bf16 (B @ A, init A~N(0,σ) B=0)."""

    def __init__(self, lin, r):
        super().__init__()
        self.base = lin
        for p in self.base.parameters():
            p.requires_grad_(False)
        dt = lin.weight.dtype
        dev = lin.weight.device
        self.A = nn.Parameter(torch.randn(r, lin.in_features, dtype=dt, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, dtype=dt, device=dev))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


def to_bias_linears(model):
    """Đổi mọi linear lượng-tử-được sang bias=True để nạp ckpt bake của exp_v/exp_r."""
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        subs = [blk.self_attn]
        subs += list(blk.mlp.experts) if moe else [blk.mlp]
        for sub in subs:
            for name, lin in list(sub.named_children()):
                if isinstance(lin, nn.Linear) and "norm" not in name and name != "gate":
                    if lin.bias is None:
                        nl = nn.Linear(lin.in_features, lin.out_features, bias=True,
                                       dtype=lin.weight.dtype)
                        nl.weight.data = lin.weight.data
                        nl.bias.data.zero_()
                        setattr(sub, name, nl)


def attach_lora(model, r):
    n = 0
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        subs = [blk.self_attn]
        subs += list(blk.mlp.experts) if moe else [blk.mlp]
        for sub in subs:
            for name, lin in list(sub.named_children()):
                if isinstance(lin, nn.Linear) and "norm" not in name and name != "gate":
                    setattr(sub, name, LoRALinear(lin, r))
                    n += 1
    return n


def build_kd_batches(tok, args, n_steps):
    """Danh sách batch CỐ ĐỊNH (seed 0) — pha T và S đi qua CÙNG thứ tự."""
    pool = []

    def add(path, n, skip=0, unesc=False, w=1):
        if not path or not os.path.exists(path):
            return 0
        ls = read_lines(path, n, skip=skip)
        if unesc:
            ls = [s.replace("\\n", "\n").replace("\\\\", "\\") for s in ls]
        pool.extend(ls * w)
        return len(ls)

    nvi = add(args.train_vi, args.kd_vi, skip=0)
    nja = add(args.train_ja, args.kd_ja, skip=0)
    nen = add(args.kd_en, args.kd_dom)
    nco = add(args.kd_code, args.kd_dom, unesc=True)
    nzh = add(args.kd_zh, args.kd_dom)
    nma = add(args.kd_math, args.kd_dom // 2)
    log(f"KD pool: vi {nvi} + ja {nja} + en {nen} + code {nco} + zh {nzh} + math {nma}"
        f" = {len(pool)} câu")
    rng = random.Random(0)
    batches = []
    for _ in range(n_steps):
        batches.append(rng.sample(pool, args.batch))
    return batches


@torch.no_grad()
def eval_probes(model, tok, probes, dev):
    out = {}
    for name, lines in probes.items():
        nll, ntok = 0.0, 0
        for s in lines:
            ids = tok(s, return_tensors="pt", truncation=True,
                      max_length=(160 if name == "code" else 96)).input_ids.to(dev)
            if ids.shape[1] < 2:
                continue
            lg = model(ids).logits.float()
            nll += F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum").item()
            ntok += ids.shape[1] - 1
        out[name] = math.exp(nll / max(ntok, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--ckpt", default="", help="ckpt bake ternary exp_v (trống = smoke)")
    ap.add_argument("--train-vi", default="/root/qat_data/train_slice.vi")
    ap.add_argument("--train-ja", default="/root/qat_data/train_slice.ja")
    ap.add_argument("--dev-vi", default="/root/qat_data/dev.vi")
    ap.add_argument("--dev-ja", default="/root/qat_data/dev.ja")
    ap.add_argument("--kd-en", default="")
    ap.add_argument("--kd-code", default="")
    ap.add_argument("--kd-zh", default="")
    ap.add_argument("--kd-math", default="")
    ap.add_argument("--kd-vi", type=int, default=18_000)
    ap.add_argument("--kd-ja", type=int, default=27_000, help="ja 60/40 vs vi")
    ap.add_argument("--kd-dom", type=int, default=8_000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--ce-w", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--tlogits", default="", help="file cache top-k logits (pha T ghi, S đọc)")
    ap.add_argument("--val100", type=int, default=1)
    ap.add_argument("--tag", default="exp_w")
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.kd_vi, args.kd_ja, args.kd_dom = 4, 2, 40, 40, 20
        args.topk = 16
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"exp_w LoRA-KD: {args.model_id} r={args.rank} steps={args.steps} dev={dev}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batches = build_kd_batches(tok, args, args.steps)
    tl_path = args.tlogits or os.path.join(OUT_DIR, f"tlogits_{args.tag}.pt")

    # ===== PHA T — teacher top-K logits (bỏ qua nếu cache đã có) =====
    if not os.path.exists(tl_path):
        log("PHA T: nạp teacher gốc lên GPU, cache top-K logits...")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev).eval()
        cache = []
        t0 = time.time()
        with torch.no_grad():
            for i, bt in enumerate(batches):
                enc = tok(bt, return_tensors="pt", padding=True, truncation=True,
                          max_length=args.seq)
                ids = enc.input_ids.to(dev)
                am = enc.attention_mask.to(dev)
                lg = teacher(ids, attention_mask=am).logits.float()
                v, ix = lg.topk(args.topk, dim=-1)
                cache.append({"ids": ids.cpu().to(torch.int32),
                              "am": am.cpu().to(torch.int8),
                              "v": v.cpu().to(torch.float16),
                              "ix": ix.cpu().to(torch.int32)})
                if i % 200 == 199:
                    log(f"  T {i+1}/{len(batches)} ({time.time()-t0:.0f}s)")
        torch.save(cache, tl_path)
        log(f"PHA T xong: {tl_path} ({os.path.getsize(tl_path)/1e9:.2f} GB)")
        del teacher, cache
        torch.cuda.empty_cache()
    else:
        log(f"PHA T: cache có sẵn {tl_path}")

    # ===== PHA S — student ternary + LoRA =====
    log("PHA S: nạp student...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    if args.ckpt:
        to_bias_linears(model)
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        state = sd["state_dict"] if "state_dict" in sd else sd
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected: {unexpected[:5]}"
        log(f"nạp ckpt ternary: {args.ckpt}")
    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = attach_lora(model, args.rank)
    model.to(dev)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"LoRA: {n_lora} linear, {sum(p.numel() for p in params)/1e6:.0f}M trainable")
    opt = torch.optim.Adam(params, lr=args.lr)
    cache = torch.load(tl_path, map_location="cpu", weights_only=False)

    dev_vi = read_lines(args.dev_vi, 400)[-16:]
    dev_ja = read_lines(args.dev_ja, 400)[-16:]
    probes = {"vi": dev_vi, "ja": dev_ja, "en": EN_EVAL, "code": CODE_EVAL,
              "zh": ZH_EVAL, "math": MATH_EVAL}
    p0 = eval_probes(model, tok, probes, dev)
    geo = lambda d: math.exp(sum(math.log(max(v, 1e-9)) for v in d.values()) / len(d))
    best_score = geo(p0)
    best_lora = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if ".A" in k or ".B" in k}
    log("step-0 (mốc S1): " + " / ".join(f"{k} {v:.1f}" for k, v in p0.items())
        + f" | geo6 {best_score:.1f}")

    T = args.kd_temp
    eval_every = max(2, args.steps // 12)
    WARMUP = max(1, args.steps // 20)
    t1 = time.time()
    for step in range(args.steps):
        fac = min(1.0, (step + 1) / WARMUP)
        prog = max(0.0, (step - WARMUP) / max(1, args.steps - WARMUP))
        for g in opt.param_groups:
            g["lr"] = args.lr * fac * 0.5 * (1 + math.cos(math.pi * prog))
        cb = cache[step]
        ids = cb["ids"].to(dev).long()
        am = cb["am"].to(dev).float()
        tv = cb["v"].to(dev).float() / T
        tix = cb["ix"].to(dev).long()
        opt.zero_grad()
        sl = model(ids, attention_mask=am.long()).logits.float()
        slT = sl / T
        lse = slT.logsumexp(-1, keepdim=True)
        s_at = slT.gather(-1, tix) - lse          # log p_s tại top-K
        t_p = F.softmax(tv, -1)                   # phân bố teacher trên top-K (renorm)
        t_lp = F.log_softmax(tv, -1)
        kl_tok = (t_p * (t_lp - s_at)).sum(-1)    # [B,Tk]
        loss = (kl_tok * am).sum() / am.sum().clamp(min=1) * (T * T)
        if args.ce_w > 0:
            m2 = (am[:, 1:] * am[:, :-1])
            ce = F.cross_entropy(sl[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
            loss = loss + args.ce_w * (ce * m2).sum() / m2.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % eval_every == eval_every - 1 or step == args.steps - 1:
            pe = eval_probes(model, tok, probes, dev)
            sc = geo(pe)
            log(f"  step {step+1}/{args.steps}: KL {loss.item():.3f} | "
                + " / ".join(f"{k} {v:.1f}" for k, v in pe.items())
                + f" | geo6 {sc:.1f} ({time.time()-t1:.0f}s)")
            if sc < best_score:
                best_score = sc
                best_lora = {k: v.detach().clone() for k, v in model.state_dict().items()
                             if ".A" in k or ".B" in k}
    model.load_state_dict(best_lora, strict=False)

    pf = eval_probes(model, tok, probes, dev)
    log("=> EXP W: " + " / ".join(f"{k} {v:.1f}" for k, v in pf.items())
        + f" | geo6 {geo(pf):.1f}")
    big_vi = big_ja = None
    if args.val100 and not args.smoke:
        v100 = {"vi100": read_lines(args.dev_vi, 200)[-100:],
                "ja100": read_lines(args.dev_ja, 200)[-100:]}
        pv = eval_probes(model, tok, v100, dev)
        big_vi, big_ja = pv["vi100"], pv["ja100"]
        log(f"   val-100 vi: {big_vi:.1f} | ja: {big_ja:.1f}")
    for pr in ("Hôm nay trời mưa nên tôi quyết định", "The most important thing about"):
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            o = model.generate(ids, max_new_tokens=40, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        log(f"  sinh: {tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)!r}")

    rj = os.path.join(OUT_DIR, "exp_w_results.json")
    outj = {}
    if os.path.exists(rj):
        try:
            with io.open(rj, "r", encoding="utf-8") as f:
                outj = json.load(f)
        except Exception:
            outj = {}
    outj[args.tag] = {"model": args.model_id, "ckpt": args.ckpt, "rank": args.rank,
                      "steps": args.steps, "s1_probe": p0, "kd": pf, "geo6": geo(pf),
                      "val100_vi": big_vi, "val100_ja": big_ja}
    with io.open(rj, "w", encoding="utf-8") as f:
        json.dump(outj, f, ensure_ascii=False, indent=2)
    if args.out and not args.smoke:
        torch.save({"lora": best_lora,
                    "meta": {"tag": args.tag, "rank": args.rank, "kd": pf,
                             "val100_vi": big_vi, "val100_ja": big_ja}}, args.out)
        log(f"đã lưu LoRA: {args.out}")
    log("EXP W XONG")


if __name__ == "__main__":
    main()
