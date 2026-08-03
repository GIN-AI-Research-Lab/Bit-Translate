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


def attach_lora(model, r, scope="all"):
    """scope='attn+down': attention đủ 4 + CHỈ down_proj mỗi expert (6.3k module thay 18.6k
    — giảm ×3 số lần phóng kernel, giữ absorber ở tầng tổn thương chính)."""
    n = 0
    for blk in model.model.layers:
        moe = hasattr(blk.mlp, "experts")
        subs = [(blk.self_attn, True)]
        subs += [(ex, False) for ex in (blk.mlp.experts if moe else [blk.mlp])]
        for sub, is_attn in subs:
            for name, lin in list(sub.named_children()):
                if not isinstance(lin, nn.Linear) or "norm" in name or name == "gate":
                    continue
                if scope == "attn+down" and not is_attn and name != "down_proj":
                    continue
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
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lora-scope", default="attn+down", choices=["all", "attn+down"])
    ap.add_argument("--lr", type=float, default=3e-5)
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
                # lseT: normalizer FULL-vocab ở nhiệt độ T -> pha S khớp được XÁC SUẤT TUYỆT ĐỐI
                # (v2 fix: renorm top-K từng chặt đuôi -> thiên kiến entropy giết ja/zh)
                lseT = (lg / args.kd_temp).logsumexp(-1)
                cache.append({"ids": ids.cpu().to(torch.int32),
                              "am": am.cpu().to(torch.int8),
                              "v": v.cpu().to(torch.float16),
                              "ix": ix.cpu().to(torch.int32),
                              "lseT": lseT.cpu().to(torch.float32)})
                if i % 200 == 199:
                    log(f"  T {i+1}/{len(batches)} ({time.time()-t0:.0f}s)")
        torch.save({"T": args.kd_temp, "batches": cache}, tl_path)
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
    n_lora = attach_lora(model, args.rank, args.lora_scope)
    model.to(dev)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"LoRA: {n_lora} linear, {sum(p.numel() for p in params)/1e6:.0f}M trainable")
    opt = torch.optim.Adam(params, lr=args.lr)
    obj = torch.load(tl_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "batches" in obj:
        assert abs(obj["T"] - args.kd_temp) < 1e-6, "kd-temp KHÁC lúc build cache (lseT sai)"
        cache = obj["batches"]
    else:
        cache = obj   # cache v1 (không lseT — renorm top-K, thiên kiến entropy đã biết)

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
        ids_f = cb["ids"].to(dev).long()
        am_f = cb["am"].to(dev).float()
        tv_f = cb["v"].to(dev).float() / T
        tix_f = cb["ix"].to(dev).long()
        lse_f = cb["lseT"].to(dev).float() if "lseT" in cb else None
        opt.zero_grad()
        Bf = ids_f.shape[0]
        mb = max(1, Bf // 2)                       # micro-batch ×2 (OOM fix, giữ loss chuẩn)
        am_tot = am_f.sum().clamp(min=1)
        m2_f = (am_f[:, 1:] * am_f[:, :-1])
        m2_tot = m2_f.sum().clamp(min=1)
        loss_acc = 0.0
        for s0 in range(0, Bf, mb):
            ids, am = ids_f[s0:s0 + mb], am_f[s0:s0 + mb]
            tv, tix = tv_f[s0:s0 + mb], tix_f[s0:s0 + mb]
            sl = model(ids, attention_mask=am.long()).logits.float()
            slT = sl / T
            s_lse = slT.logsumexp(-1, keepdim=True)
            s_at = slT.gather(-1, tix) - s_lse     # log p_s TUYỆT ĐỐI tại top-K (temp T)
            if lse_f is not None:
                # v2: KL đúng cả KHỐI ĐUÔI — teacher probs tuyệt đối nhờ lseT full-vocab.
                # Fix thiên kiến entropy: renorm top-K từng ép student bóp đuôi -> ja/zh nổ.
                t_lp = tv - lse_f[s0:s0 + mb].unsqueeze(-1)
                t_p = t_lp.exp()
                t_tail = (1.0 - t_p.sum(-1)).clamp(min=1e-5)
                s_tail = (1.0 - s_at.exp().sum(-1)).clamp(min=1e-5)
                kl_tok = (t_p * (t_lp - s_at)).sum(-1) \
                    + t_tail * (t_tail.log() - s_tail.log())
            else:                                   # v1 renorm (giữ để so sánh)
                t_p = F.softmax(tv, -1)
                t_lp = F.log_softmax(tv, -1)
                kl_tok = (t_p * (t_lp - s_at)).sum(-1)
            loss = (kl_tok * am).sum() / am_tot * (T * T)
            if args.ce_w > 0:
                m2 = m2_f[s0:s0 + mb]
                ce = F.cross_entropy(sl[:, :-1].transpose(1, 2), ids[:, 1:],
                                     reduction="none")
                loss = loss + args.ce_w * (ce * m2).sum() / m2_tot
            loss.backward()
            loss_acc += loss.item()
            del sl, slT, lse, s_at
        loss = torch.tensor(loss_acc)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 49:
            log(f"  ... {step+1}/{args.steps} | KL {loss_acc:.3f}"
                f" | {(time.time()-t1)/(step+1):.1f}s/bước")
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
                if args.out and not args.smoke:   # lưu TĂNG DẦN — app chết không mất trắng
                    torch.save({"lora": {k: v.cpu() for k, v in best_lora.items()},
                                "meta": {"tag": args.tag, "rank": args.rank,
                                         "step": step + 1, "probe": pe, "geo6": sc}},
                               args.out)
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
