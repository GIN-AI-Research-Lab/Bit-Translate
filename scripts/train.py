#!/usr/bin/env python3
"""Bước 4c — train the BitNet b1.58 decoder-only translator.

Follows CLAUDE.md §7: data preloaded to RAM, bf16 AMP (Ampere), 8-bit Adam
(bitsandbytes), length-bucketed token-budget batching, gradient accumulation,
periodic checkpoint + resume, file logging. Designed to run unattended for days
under nohup; kill/relaunch resumes from checkpoints/last.pt.
"""
import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Graceful pause: on SIGTERM/SIGINT set a flag; the loop saves last.pt and
# exits cleanly at the next optimizer-step boundary, so pausing to play a game
# loses no progress. pause_training.sh sends SIGTERM.
_STOP = {"flag": False}
def _on_signal(signum, frame):
    _STOP["flag"] = True
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bitnet import BitNetLM, BitNetConfig

ROOT = Path(__file__).parent.parent
BIN = ROOT / "data" / "bin"
CKPT = ROOT / "checkpoints"
PAD_ID = 0


def load_split(split):
    idx = np.load(BIN / f"{split}.index.npy")          # [N,2] length, tgt_start
    toks = np.fromfile(BIN / f"{split}.tokens.u16", dtype=np.uint16)
    offsets = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(idx[:, 0], out=offsets[1:])
    return toks, idx, offsets


def make_batches(idx, max_tokens, rng, drop_last_partial=False, pad_multiple=1):
    """Length-bucketed token-budget micro-batches. Shuffles megabatches, sorts
    each by length so padding waste is small, packs under max_tokens.
    pad_multiple>1: pack theo độ dài ĐÃ LÀM TRÒN lên bội số -> số shape (B,T) khác
    nhau ít đi hẳn, torch.compile 292M hội tụ (không thì compile graph mới mãi ->
    treo) + allocator ít phân mảnh."""
    n = len(idx)
    order = rng.permutation(n)
    mega = 50 * max(1, max_tokens // int(idx[:, 0].mean() + 1))
    batches = []
    for s in range(0, n, mega):
        chunk = order[s:s + mega]
        chunk = chunk[np.argsort(idx[chunk, 0])]  # by length asc
        cur, cur_max = [], 0
        for i in chunk:
            L = int(idx[i, 0])
            new_max = max(cur_max, L)
            new_max_p = -(-new_max // pad_multiple) * pad_multiple if pad_multiple > 1 else new_max
            if cur and new_max_p * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(int(i)); cur_max = new_max
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate(batch, toks, idx, offsets, device, pad_multiple=1):
    lens = idx[batch, 0]
    tstart = idx[batch, 1]
    maxlen = int(lens.max())
    if pad_multiple > 1:
        maxlen = -(-maxlen // pad_multiple) * pad_multiple   # làm tròn lên bội số
    B = len(batch)
    padded = np.full((B, maxlen), PAD_ID, dtype=np.int64)
    for r, i in enumerate(batch):
        o = offsets[i]; L = int(lens[r])
        padded[r, :L] = toks[o:o + L]
    # NB: no non_blocking=True — async H2D from pageable memory triggers
    # "CUDA driver error: device not ready" under this WSL2 setup.
    padded = torch.from_numpy(padded).to(device)
    inp = padded[:, :-1]
    tgt = padded[:, 1:]
    # mask: label at pos j (=token j+1) counts iff it's in target segment & not pad
    jp1 = np.arange(1, maxlen)[None, :]
    mask = (jp1 >= tstart[:, None]) & (jp1 < lens[:, None])
    mask = torch.from_numpy(mask).to(device)
    return inp, tgt, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--grad-accum", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr-anchor", type=int, default=0,
                    help="LR schedule coi step này là mốc bắt đầu (restart LR khi resume "
                         "vòng data mới; vd resume ở 14000 -> --lr-anchor 14000 để LR chạy lại từ đỉnh)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=100)      # overwrite last.pt (~440MB)
    ap.add_argument("--milestone-every", type=int, default=2000)  # keep stepN.pt
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ckpt-every-k", type=int, default=1,
                    help="selective gradient checkpointing: chỉ checkpoint 1/k block "
                         "(k=2/4 nhẹ VRAM hơn full GC nhưng nhanh hơn; cần --grad-ckpt)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--fused-adamw", action="store_true",
                    help="AdamW(fused=True): diệt spike ~2.33GB của foreach lúc opt.step "
                         "(điều kiện tiên quyết để 292M fit 8GB khi bớt gradient-checkpoint)")
    ap.add_argument("--sdpa", default="auto", choices=["auto", "flash", "mem"],
                    help="mem = mem-efficient attention; né bug flash-attn backward "
                         "'CUDA driver error: device not ready' của WSL2")
    ap.add_argument("--pad-multiple", type=int, default=1,
                    help="gom độ dài batch về bội số này (vd 32) -> ít shape (B,T) hơn "
                         "-> torch.compile 292M hội tụ nhanh (không thì compile graph mới "
                         "mãi -> treo) + allocator ít phân mảnh. đổi lấy ~ít token thừa.")
    # Game mode: cap VRAM so a game has room, and throttle GPU duty cycle.
    ap.add_argument("--mem-frac", type=float, default=0.0,
                    help=">0 caps this process to that fraction of VRAM (e.g. 0.45)")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="sleep this * step_time after each step (1.0 ~= 50%% GPU duty)")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run this many steps on a tiny slice")
    # Kích thước model (mặc định = 110M gốc). Scale 300M: --d-model 1152 --n-layers 16
    # --n-heads 18 --d-ff 3072 (giữ head_dim 64 để khớp graph build_bitnet_158).
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--vocab-size", type=int, default=32000,
                    help="đổi khi tokenizer có thêm token mới (vd >>fix<<, "
                         "scripts/add_fix_token.py) — phải khớp checkpoint đã "
                         "mở rộng bằng scripts/expand_checkpoint_vocab.py")
    ap.add_argument("--max-seq", type=int, default=256,
                    help="độ dài tối đa cache RoPE (BitNetConfig.max_seq) — PHẢI "
                         ">= chiều dài sequence dài nhất trong data (vd task "
                         ">>fix<< đóng gói tới 320 token, xem "
                         "scripts/pack_fix_into_bin.py MAX_SEQ). Sequence dài hơn "
                         "giá trị này crash RuntimeError trong apply_rope (đã xảy "
                         "ra thật ở v3 step~10500+, tensor 270 vs cache 256).")
    args = ap.parse_args()

    CKPT.mkdir(exist_ok=True)
    device = "cuda"
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
        print(f"VRAM capped to {args.mem_frac:.0%} (game mode)", flush=True)

    toks, idx, offsets = load_split("train")
    if args.smoke:
        idx = idx[:20000]
    print(f"train sequences: {len(idx):,} | tokens: {len(toks):,}", flush=True)

    cfg = BitNetConfig(vocab_size=args.vocab_size, d_model=args.d_model, d_ff=args.d_ff,
                       n_layers=args.n_layers, n_heads=args.n_heads, max_seq=args.max_seq)
    model = BitNetLM(cfg).to(device)
    model.gradient_checkpointing = args.grad_ckpt
    model.ckpt_every_k = args.ckpt_every_k
    print(f"params: {model.num_params()/1e6:.1f}M "
          f"(d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} ff={cfg.d_ff})", flush=True)

    # NB: plain AdamW, not bitsandbytes Adam8bit — bnb's 8-bit optimizer kernels
    # trigger "CUDA driver error: device not ready" under async execution in
    # this WSL2 setup (blocking mode works but is ~5-10x slower). For a 110M
    # model the fp32 optimizer states (~1.8GB) fit the 8GB card fine, so 8-bit
    # Adam isn't needed here despite CLAUDE.md §7.3.
    decay, nodecay = [], []
    for n_, p in model.named_parameters():
        (nodecay if p.ndim < 2 else decay).append(p)
    adam_kw = dict(lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    if args.fused_adamw:
        adam_kw["fused"] = True   # kernel CUDA chuẩn của torch (KHÔNG phải bnb 8-bit)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.wd},
        {"params": nodecay, "weight_decay": 0.0},
    ], **adam_kw)

    step = 0
    last = CKPT / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        step = ck["step"]
        # checkpoint mở rộng vocab (scripts/expand_checkpoint_vocab.py) KHÔNG có
        # "opt" (shape optimizer cũ không khớp embedding mới) -> optimizer mới
        # tinh, coi như khởi động lại pha train (multi-task >>fix<<, PLAN_KD_JA2VI).
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
            except (ValueError, RuntimeError) as e:
                print(f"CẢNH BÁO: opt state không khớp ({e}) -> dùng optimizer mới", flush=True)
        else:
            print("checkpoint không có optimizer state -> dùng optimizer mới", flush=True)
        print(f"resumed at step {step}", flush=True)

    # Cache ternary weight qua các microbatch (BitNet b1.58): tính weight_quant 1
    # lần/optimizer-step thay vì mỗi microbatch. Gọi SAU load checkpoint (cache từ
    # weights đã resume) và TRƯỚC compile (guard nhánh ổn định). No-op nếu
    # BITNET_OPT tắt wqcache. autocast bf16 -> cache bf16.
    model.attach_wq_autorefresh(opt, torch.bfloat16)

    run_model = model
    if args.compile:
        try:
            # dynamic=True: our batches vary in shape (length bucketing); without
            # it, inductor recompiles per shape and thrashes.
            run_model = torch.compile(model, dynamic=True)
            print("torch.compile enabled (dynamic=True)", flush=True)
        except Exception as e:
            print(f"compile failed, continuing eager: {e}", flush=True)

    # Chọn backend attention: mem-efficient né bug flash-attn backward
    # ("CUDA driver error: device not ready") của WSL2. auto = để torch tự chọn.
    import contextlib
    sdpa_ctx = contextlib.nullcontext()
    if args.sdpa != "auto":
        from torch.nn.attention import sdpa_kernel, SDPBackend
        _b = {"flash": SDPBackend.FLASH_ATTENTION,
              "mem": SDPBackend.EFFICIENT_ATTENTION}[args.sdpa]
        sdpa_ctx = sdpa_kernel([_b])
        print(f"SDPA backend forced: {args.sdpa}", flush=True)

    def lr_at(s):
        s = s - args.lr_anchor                       # restart: schedule tính từ mốc anchor
        total = max(1, args.max_steps - args.lr_anchor)
        if s < args.warmup:
            return args.lr * s / max(1, args.warmup)
        t = (s - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))

    logf = open(CKPT / "train.log", "a", buffering=1)
    def log(m):
        print(m, flush=True); logf.write(m + "\n")

    log(f"=== start/resume step={step} max_steps={args.max_steps} "
        f"max_tokens={args.max_tokens} grad_accum={args.grad_accum} lr={args.lr} ===")
    rng = np.random.default_rng(1234 + step)
    model.train()
    t0 = time.time(); tok_seen = 0; loss_acc = 0.0; micro = 0
    accum_target = 1 if args.smoke else args.grad_accum

    # Vào sdpa backend context MỘT LẦN cho toàn bộ vòng train (sdpa_kernel trả
    # _GeneratorContextManager dùng-một-lần — KHÔNG được `with` lặp mỗi microbatch).
    # nullcontext (khi --sdpa auto) enter một lần cũng vô hại. Process chỉ để train
    # nên không cần exit; nó tự dọn khi process kết thúc.
    sdpa_ctx.__enter__()

    step_t0 = time.time()
    while step < args.max_steps:
        for batch in make_batches(idx, args.max_tokens, rng, pad_multiple=args.pad_multiple):
            inp, tgt, mask = collate(batch, toks, idx, offsets, device,
                                     pad_multiple=args.pad_multiple)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = run_model(inp, targets=tgt, loss_mask=mask)
            (loss / accum_target).backward()
            loss_acc += loss.item(); tok_seen += int(mask.sum().item()); micro += 1

            if micro % accum_target == 0:
                lr = lr_at(step)
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1

                if args.throttle > 0:
                    step_dt = time.time() - step_t0
                    time.sleep(args.throttle * step_dt)   # yield GPU to the game
                step_t0 = time.time()

                if step % args.log_every == 0:
                    dt = time.time() - t0
                    log(f"step {step} | loss {loss_acc/ (args.log_every*accum_target):.4f} "
                        f"| lr {lr:.2e} | {tok_seen/dt:,.0f} tok/s | {dt/args.log_every:.2f}s/step")
                    loss_acc = 0.0; tok_seen = 0; t0 = time.time()

                if step % args.save_every == 0 or step >= args.max_steps:
                    tmp = CKPT / "last.pt.tmp"
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "cfg": vars(cfg)}, tmp)
                    os.replace(tmp, last)   # atomic; safe to kill mid-train
                    if step % args.milestone_every == 0 or step >= args.max_steps:
                        torch.save({"model": model.state_dict(), "step": step, "cfg": vars(cfg)},
                                   CKPT / f"step{step}.pt")
                    log(f"  saved checkpoint @ step {step}")

                if _STOP["flag"]:
                    tmp = CKPT / "last.pt.tmp"
                    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                                "step": step, "cfg": vars(cfg)}, tmp)
                    os.replace(tmp, last)
                    log(f"=== PAUSED: saved at step {step}, exiting cleanly ===")
                    return

                if step >= args.max_steps:
                    break
        if args.smoke and step >= args.smoke:
            break
    log("=== training loop exited ===")


if __name__ == "__main__":
    main()
