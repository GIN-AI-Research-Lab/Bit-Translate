# -*- coding: utf-8 -*-
"""
Exp AA — TOY GATES cho hướng nghiên cứu sub-bit mới (rollout-KD + bit-curriculum).
100% CPU máy B, $0, chạy TRƯỚC khi tốn tiền Modal — bắt lỗi thiết kế bằng tensor nhỏ.

5 cổng (mỗi cổng PASS/FAIL + số đo, ghi exp_aa_results.json):
  G1  Họ mask N:M LỒNG NHAU (1:32⊂1:16⊂1:8⊂1:4⊂2:4) + kế toán bpw + masked-STE đúng.
  G2  Transition-shock: full-STE để ô pruned random-walk (bài học B1a2/exp_r) → khi GROW
      mask, ô mới thừa kế NHIỄU; masked-STE giữ ô pruned đóng băng → grow êm. Đo trên
      tensor THẬT Qwen3-0.6B.
  G3  Toy rollout-KD (ngôn ngữ ngoặc + oracle): student ternary bị exposure bias khi
      free-running; DAgger-KD (teacher chấm trên ngữ cảnh CỦA STUDENT) giảm vi phạm
      so teacher-forced-KD cùng ngân sách. Đây là cơ chế lõi của exp_ab.
  G4  (chuyển sang exp_ac_divergence_probe.py — cần model 0.6B thật, file riêng)
  G5  Router margin loss (hạt giống A, cho pha 30B): phạt "gần-hòa" làm giảm flip-rate
      của top-k dưới nhiễu lượng tử mà không phá MSE sạch — toy MoE 16 expert.

Chạy: set OMP_NUM_THREADS=5 && python eval/lowbit_ptq/exp_aa_subbit_toys.py
"""
import glob
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0)
torch.set_num_threads(5)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_GLOB = r"D:\Bit-Translate-data\hf_cache\hub\models--Qwen--Qwen3-0.6B\snapshots\*"
LADDER = ((1, 32), (1, 16), (1, 8), (1, 4), (2, 4))   # 0.33 → 0.47 → 0.70 → 1.02 → 1.57 bpw
RESULTS = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gate(name, ok, detail):
    RESULTS[name] = {"pass": bool(ok), **detail}
    log(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ============================== bộ lượng tử dùng chung ==============================
def nm_bpw(n, m, sgroup=64, sbits=8.0):
    """Kế toán bpw của exp_r: payload ternary + entropy mask + scale."""
    return n / m * math.log2(3) + math.log2(math.comb(m, n)) / m + sbits / sgroup


def nested_rank(imp, ladder=LADDER):
    """Rank kích hoạt LỒNG NHAU từ importance [R,C] (C chia hết cho M lớn nhất).
    rank[i,j] = chỉ số stage sớm nhất mà ô (i,j) active (0..len(ladder)-1), 127 = không bao giờ.
    Xây bằng 'tournament' từ mịn → thô: top-2/4 (stage cuối), top-1/4, rồi winner của từng
    cặp nhóm khi M nhân đôi. Bảo đảm mask(stage k) ⊆ mask(stage k+1) THEO CẤU TRÚC —
    không bao giờ refresh mask giữa chừng (tôn trọng luật 'mask refresh = ×2.2' của lab)."""
    R, C = imp.shape
    Mmax = ladder[0][1]
    assert C % Mmax == 0, "C phải chia hết cho M lớn nhất"
    rank = torch.full((R, C), 127, dtype=torch.long)
    # stage cuối (2:4): top2 mỗi nhóm 4; stage áp cuối (1:4): top1 mỗi nhóm 4
    g4 = imp.view(R, -1, 4)
    top2 = g4.topk(2, dim=2).indices                       # [R, C/4, 2]
    base = torch.arange(0, C, 4).view(1, -1, 1)
    flat2 = (top2 + base).view(R, -1)
    rank.scatter_(1, flat2[:, 1::2], torch.full_like(flat2[:, 1::2], len(ladder) - 1))
    rank.scatter_(1, flat2[:, 0::2], torch.full_like(flat2[:, 0::2], len(ladder) - 2))
    # các stage thô hơn: winner giữa 2 nhóm con (theo imp của cell thắng nhóm con)
    win = flat2[:, 0::2].view(R, -1)                       # winner mỗi nhóm-4 [R, C/4]
    for k in range(len(ladder) - 3, -1, -1):               # 1:8 -> 1:16 -> 1:32
        pair = win.view(R, -1, 2)                          # 2 nhóm con cạnh nhau
        w_imp = imp.gather(1, pair.reshape(R, -1)).view(R, -1, 2)
        sel = w_imp.argmax(dim=2, keepdim=True)
        win = pair.gather(2, sel).squeeze(2)               # [R, C/(4*2^i)]
        rank.scatter_(1, win, torch.full_like(win, k))
    return rank


def fit_scale(Wm, act, G=64, iters=3):
    """Scale per-(row, nhóm G) kiểu Lloyd trên ô ACTIVE (act = mask float)."""
    R, C = Wm.shape
    Wv = (Wm * act).view(R, -1, G)
    Av = act.view(R, -1, G)
    cnt = Av.sum(2, keepdim=True).clamp(min=1)
    s = (Wv.abs().sum(2, keepdim=True) / cnt).clamp(min=1e-6)
    for _ in range(iters):
        t = torch.round(Wv / s).clamp(-1, 1) * Av
        num = (Wv * t).sum(2, keepdim=True)
        den = (t * t).sum(2, keepdim=True).clamp(min=1e-8)
        s = (num / den).abs().clamp(min=1e-6)
    return s


def to_f8(s):
    try:
        return s.to(torch.float8_e4m3fn).to(s.dtype)
    except Exception:
        return s


class NestedQLinear(nn.Module):
    """Bản toy của quantizer exp_ab: mask lồng nhau theo rank + masked-STE tùy chọn
    + highway columns (cột input giữ FP, loại khỏi mask/quant)."""

    def __init__(self, W0, imp, stage=0, G=64, masked_ste=True, hw_cols=None, bias=True):
        super().__init__()
        R, C = W0.shape
        self.R, self.C, self.G = R, C, G
        self.Wfp = nn.Parameter(W0.clone())
        self.bias = nn.Parameter(torch.zeros(R)) if bias else None
        self.register_buffer("rank", nested_rank(imp))
        hw = torch.zeros(C, dtype=torch.bool)
        if hw_cols is not None:
            hw[hw_cols] = True
        self.register_buffer("hw", hw)
        self.masked_ste = masked_ste
        self._h = self.Wfp.register_hook(self._grad_hook)
        self.set_stage(stage)

    def _grad_hook(self, g):
        if self.masked_ste:
            return g * self.gmask
        return g

    @torch.no_grad()
    def set_stage(self, k):
        self.stage = k
        act = (self.rank <= k).float()
        act[:, self.hw] = 0.0                      # highway KHÔNG đi qua quant
        gm = act.clone()
        gm[:, self.hw] = 1.0                       # nhưng vẫn được train (grad thật)
        if hasattr(self, "act"):
            self.act.copy_(act)
            self.gmask.copy_(gm)
            self.s.copy_(fit_scale(self.Wfp.data, act, self.G))
        else:
            self.register_buffer("act", act)
            self.register_buffer("gmask", gm)
            self.register_buffer("s", fit_scale(self.Wfp.data, act, self.G))

    def quant(self):
        s = to_f8(self.s)
        Wv = (self.Wfp * self.act).view(self.R, -1, self.G)
        t = (torch.round(Wv / s).clamp(-1, 1) * self.act.view(self.R, -1, self.G)).detach()
        return (t * s).reshape(self.R, self.C)

    def forward(self, x):
        q = self.quant()
        Wq = q + (self.Wfp - self.Wfp.detach())    # STE (grad = 1 mọi ô; hook lọc nếu masked)
        Wq = torch.where(self.hw.view(1, -1), self.Wfp, Wq)   # highway: FP thật
        return F.linear(x, Wq, self.bias)

    def bpw(self, ladder=LADDER):
        n, m = ladder[self.stage]
        hw_extra = 16.0 * self.hw.float().mean().item()
        return nm_bpw(n, m, self.G) + hw_extra


# ============================== G1 — họ mask + STE ==============================
def gate1():
    torch.manual_seed(1)
    R, C = 64, 256
    imp = torch.rand(R, C)
    rank = nested_rank(imp)
    # (a) lồng nhau + đúng đếm N:M từng stage
    ok, counts = True, []
    prev = None
    for k, (n, m) in enumerate(LADDER):
        act = (rank <= k)
        per = act.view(R, -1, m).sum(2)
        cnt_ok = bool((per <= n).all()) and bool((per == n).float().mean() > 0.999)
        nest_ok = True if prev is None else bool((act | prev).eq(act).all())
        ok &= cnt_ok and nest_ok
        counts.append({"nm": f"{n}:{m}", "max_per_group": int(per.max()),
                       "bpw": round(nm_bpw(n, m), 3), "nested": nest_ok})
        prev = act
    # (b) bpw khớp mốc lab (exp_l/exp_r từng đo cùng công thức)
    expect = {(1, 32): 0.331, (1, 16): 0.474, (1, 8): 0.699, (1, 4): 1.023, (2, 4): 1.566}
    bpw_ok = all(abs(nm_bpw(n, m) - v) < 5e-3 for (n, m), v in expect.items())
    # (c) masked-STE: grad = 0 đúng trên ô pruned, ≠0 trên ô active + highway
    lin = NestedQLinear(torch.randn(R, C) * 0.02, imp, stage=1, masked_ste=True,
                        hw_cols=[0, 7])
    x = torch.randn(8, C)
    lin(x).pow(2).mean().backward()
    g = lin.Wfp.grad
    pruned = (lin.act == 0) & (~lin.hw.view(1, -1).expand(R, C))
    ste_ok = bool((g[pruned].abs().max() == 0)) and bool(g[:, 0].abs().max() > 0) \
        and bool((g * lin.act).abs().max() > 0)
    # (d) highway: cột hw đi FP chính xác (so forward thủ công)
    with torch.no_grad():
        y = lin(x)
        Wq = lin.quant()
        Wq[:, lin.hw] = lin.Wfp[:, lin.hw]
        y_ref = F.linear(x, Wq, lin.bias)
    hw_ok = bool((y - y_ref).abs().max() < 1e-5)
    gate("G1_nested_family", ok and bpw_ok and ste_ok and hw_ok,
         {"counts": counts, "bpw_match_lab": bpw_ok, "masked_ste": ste_ok, "highway": hw_ok})


# ============================== G2 — transition shock ==============================
def load_real_tensor():
    try:
        from safetensors import safe_open
        mdir = glob.glob(MODEL_GLOB)[0]
        st = glob.glob(os.path.join(mdir, "*.safetensors"))[0]
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                if "layers.2.mlp.down_proj.weight" in k:
                    return f.get_tensor(k).float()[:256, :512].contiguous(), True
    except Exception as e:
        log(f"  (không nạp được tensor thật: {type(e).__name__} — dùng synthetic heavy-tail)")
    W = torch.randn(256, 512) * 0.02
    W[torch.rand_like(W) < 0.01] *= 8
    return W, False


def run_drift(W0, imp, masked, steps=300, lr=1e-3, seed=3):
    torch.manual_seed(seed)
    lin = NestedQLinear(W0, imp, stage=2, masked_ste=masked, bias=False)  # 1:8
    X = torch.randn(64, W0.shape[1])
    with torch.no_grad():
        Y = X @ W0.t()
    opt = torch.optim.Adam([lin.Wfp], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(lin(X), Y).backward()
        opt.step()
    pruned = (lin.act == 0)
    drift = (lin.Wfp.data - W0).abs()
    with torch.no_grad():
        y_before = lin(X)
        lin.set_stage(3)                                    # grow 1:8 -> 1:4
        y_after = lin(X)
        new = (lin.rank == 3)
        newly_on = (torch.round(lin.Wfp.data.view(lin.R, -1, lin.G) / to_f8(lin.s))
                    .clamp(-1, 1).reshape(lin.R, lin.C)[new] != 0).float().mean().item()
    shock = ((y_after - y_before).norm() / y_before.norm()).item()
    err = (F.mse_loss(y_after, Y) / Y.pow(2).mean()).item()
    return {"drift_pruned": drift[pruned].mean().item(),
            "drift_kept": drift[~pruned].mean().item(),
            "grow_shock_rel": shock, "err_after_rel": err,
            "new_cells_on_ratio": newly_on}


def gate2():
    W0, real = load_real_tensor()
    imp = W0.abs() * (0.5 + torch.rand(W0.shape[1]))[None, :]
    a = run_drift(W0.clone(), imp, masked=False)
    b = run_drift(W0.clone(), imp, masked=True)
    # kỳ vọng ghi trước: full-STE pruned drift ~ kept drift (random walk);
    # masked-STE pruned drift = 0; shock khi grow của masked NHỎ HƠN full.
    ok = (b["drift_pruned"] < 1e-9 and a["drift_pruned"] > 3 * b["drift_pruned"]
          and b["grow_shock_rel"] <= a["grow_shock_rel"] * 1.05)
    gate("G2_transition_shock", ok,
         {"real_tensor": real, "full_ste": {k: round(v, 5) for k, v in a.items()},
          "masked_ste": {k: round(v, 5) for k, v in b.items()}})


# ============================== G3 — toy rollout-KD ==============================
# Task "copy-chain trên manifold có cấu trúc": data vàng chỉ gồm 8 PATTERN cố định
# (từ điển) lặp chu kỳ 8: s_t = s_{t-8}. Oracle exact cho MỌI prefix — kể cả prefix
# student tự sinh có lỗi: "copy token (t-8) CỦA CHÍNH MÀY" (0.9 one-hot + 0.1 uniform).
# Điểm mấu chốt (học từ 2 bản toy hỏng trước): off-manifold phải KHÁC PHÂN BỐ data vàng.
# TF-KD chỉ thấy 8 pattern nguyên vẹn → được phép "ghi nhớ pattern" thay vì học luật
# copy tổng quát; khi tự sinh lệch 1 token, cửa sổ đột biến nằm ngoài từ điển → model
# ghi nhớ sẽ "bẻ về pattern đã thuộc" (analog của rơi-vào-loop-đã-thuộc ở LLM 1.56bpw).
# Rollout-KD (DAgger) train đúng trên trạng thái đột biến đó với target oracle.
NV, PLEN, TLEN, NDICT = 16, 8, 40, 8      # vocab (BOS = NV), prompt 8, tổng 40, 8 pattern
DICT = torch.randint(0, NV, (NDICT, PLEN), generator=torch.Generator().manual_seed(123))


def oracle_batch(seqs):
    """[B,T] -> dist [B,T,NV] cho token t | prefix<t. t<PLEN: uniform (không tính loss)."""
    B, T = seqs.shape
    tgt = torch.full((B, T, NV), 1.0 / NV)
    if T > PLEN:
        src = seqs[:, :T - PLEN]                            # token (t-8) của CHÍNH chuỗi
        oh = F.one_hot(src.clamp(max=NV - 1), NV).float()
        tgt[:, PLEN:] = 0.9 * oh + 0.1 / NV
    return tgt


def dict_prompts(B, gen, mutate=0):
    """Prompt từ từ điển; mutate>0 = đổi ngẫu nhiên k vị trí (vào off-manifold có kiểm soát)."""
    p = DICT[torch.randint(0, NDICT, (B,), generator=gen)].clone()
    for _ in range(mutate):
        pos = torch.randint(0, PLEN, (B,), generator=gen)
        val = torch.randint(0, NV, (B,), generator=gen)
        p[torch.arange(B), pos] = val
    return p


def sample_gold(B, gen):
    """Chuỗi vàng: pattern từ điển + lặp chu kỳ 8 hoàn hảo (manifold hẹp có cấu trúc)."""
    p = dict_prompts(B, gen)
    reps = [p]
    while sum(r.shape[1] for r in reps) < TLEN:
        reps.append(p)
    return torch.cat(reps, 1)[:, :TLEN]


class TinyLM(nn.Module):
    def __init__(self, d=64, nh=4, nl=2, quant_stage=None, imp_seed=7):
        super().__init__()
        self.emb = nn.Embedding(NV + 1, d)                  # + BOS = NV
        self.pos = nn.Embedding(TLEN + 1, d)
        self.blocks = nn.ModuleList()
        self.d = d
        g = torch.Generator().manual_seed(imp_seed)
        for _ in range(nl):
            blk = nn.ModuleDict({
                "ln1": nn.LayerNorm(d), "ln2": nn.LayerNorm(d),
                "qkv": self._lin(3 * d, d, quant_stage, g),
                "o": self._lin(d, d, quant_stage, g),
                "up": self._lin(4 * d, d, quant_stage, g),
                "down": self._lin(d, 4 * d, quant_stage, g)})
            self.blocks.append(blk)
        self.nh = nh
        self.head = nn.Linear(d, NV)

    @staticmethod
    def _lin(R, C, quant_stage, g):
        W0 = torch.randn(R, C, generator=g) * (1.0 / math.sqrt(C))
        if quant_stage is None:
            lin = nn.Linear(C, R)
            lin.weight.data = W0
            return lin
        imp = W0.abs() + 1e-3 * torch.rand(R, C, generator=g)
        return NestedQLinear(W0, imp, stage=quant_stage, G=32, masked_ste=True)

    def forward(self, ids):
        B, T = ids.shape
        h = self.emb(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for blk in self.blocks:
            x = blk["ln1"](h)
            qkv = blk["qkv"](x).view(B, T, 3, self.nh, self.d // self.nh)
            q, k, v = qkv.unbind(2)
            att = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.d // self.nh)
            att = att.masked_fill(mask[None, None], -1e9).softmax(-1)
            h = h + blk["o"](torch.einsum("bhts,bshd->bthd", att, v).reshape(B, T, self.d))
            h = h + blk["down"](F.gelu(blk["up"](blk["ln2"](h))))
        return self.head(h)


def self_consistency(seqs):
    """Acc chu kỳ trên vùng TỰ SINH: mean(s_t == s_{t-8}), t ≥ PLEN — đo được không cần
    gold (mạch nội tại của CHÍNH chuỗi). Kèm curve theo vị trí để soi compounding."""
    ok = (seqs[:, PLEN:] == seqs[:, :-PLEN]).float()
    curve = ok.mean(0)
    return ok.mean().item(), [round(v, 3) for v in curve[::8].tolist()]


@torch.no_grad()
def rollout(model, B, gen, temp=1.0, mutate=0, greedy=False):
    """Prompt từ điển (±đột biến) + tự sinh 32 token."""
    prm = dict_prompts(B, gen, mutate=mutate)
    seqs = torch.cat([torch.full((B, 1), NV, dtype=torch.long), prm], 1)
    for _ in range(TLEN - PLEN):
        logits = model(seqs)[:, -1] / temp
        if greedy:
            tok = logits.argmax(-1, keepdim=True)
        else:
            tok = torch.multinomial(logits.softmax(-1), 1, generator=gen)
        seqs = torch.cat([seqs, tok], 1)
    return seqs[:, 1:]


def kd_step(model, opt, ctx, tgt):
    logits = model(F.pad(ctx, (1, 0), value=NV))[:, :-1]   # dự đoán token t từ prefix<t
    kl = -(tgt * F.log_softmax(logits, -1)).sum(-1)
    loss = kl[:, PLEN:].mean()                             # chỉ tính vùng tiếp diễn
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train_arm(use_rollout, steps=500, B=48, seed=11, stage=2):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = TinyLM(quant_stage=stage)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for st in range(steps):
        if use_rollout and st >= steps // 5 and st % 2 == 1:
            ctx = rollout(model, B, gen)          # DAgger: trạng thái CỦA student (tự trôi)
        else:
            ctx = sample_gold(B, gen)             # teacher-forced: manifold từ điển
        tgt = oracle_batch(ctx)                   # oracle chấm MỌI prefix
        kd_step(model, opt, ctx, tgt)
    g2 = torch.Generator().manual_seed(999)
    # 3 thước: TF-acc (thước cũ, trên manifold vàng) / tự sinh sampled từ prompt sạch /
    # GREEDY từ prompt ĐỘT BIẾN 2 vị trí (recovery — vào off-manifold có kiểm soát)
    acc_s, curve = self_consistency(rollout(model, 192, g2, temp=1.0))
    acc_m, curve_m = self_consistency(rollout(model, 192, g2, mutate=2, greedy=True))
    gold = sample_gold(96, torch.Generator().manual_seed(777))
    with torch.no_grad():
        logits = model(F.pad(gold, (1, 0), value=NV))[:, :-1]
        tf_acc = (logits.argmax(-1)[:, PLEN:] == gold[:, PLEN:]).float().mean().item()
    return {"tf": tf_acc, "roll_sampled": acc_s, "roll_mutated_greedy": acc_m,
            "curve_sampled": curve, "curve_mutated": curve_m}


def gate3():
    t0 = time.time()
    a = train_arm(False)
    b = train_arm(True)
    gap_a = max(a["tf"] - a["roll_sampled"], 1e-9)          # exposure gap dưới SAMPLING
    gap_b = max(b["tf"] - b["roll_sampled"], 1e-9)          # (chế độ lỗi thật của lab)
    late_a, late_b = a["curve_sampled"][-1], b["curve_sampled"][-1]
    ok = (b["roll_sampled"] - a["roll_sampled"] > 0.08 and gap_b < gap_a * 0.85
          and late_b > late_a * 1.3 and b["tf"] > a["tf"] - 0.05
          and b["roll_mutated_greedy"] >= a["roll_mutated_greedy"] - 0.02)
    gate("G3_toy_rollout_kd", ok,
         {"tf_kd": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()},
          "rollout_kd": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()},
          "exposure_gap_sampled": {"tf_kd": round(gap_a, 3), "rollout_kd": round(gap_b, 3)},
          "note": "data vàng = 8 pattern cố định (manifold hẹp có cấu trúc). Thước CHÍNH "
                  "= self-acc tự sinh SAMPLING (đúng chế độ lỗi thật: sinh sampling mất "
                  "mạch); phụ = greedy từ prompt đột biến (recovery). Kỳ vọng ghi trước: "
                  "TF-KD sụp curve theo vị trí (compounding); rollout-KD +≥8đ self-acc, "
                  "bucket cuối +≥30%, TF-acc không tụt, recovery không xấu đi.",
          "sec": round(time.time() - t0, 1)})


# ============================== G5 — router margin ==============================
class ToyMoE(nn.Module):
    def __init__(self, d=32, E=16, seed=5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.router = nn.Parameter(torch.randn(E, d, generator=g) * 0.3)
        self.experts = nn.Parameter(torch.randn(E, d, d, generator=g) * 0.15)

    def forward(self, x, ret_logits=False):
        z = x @ self.router.t()
        w, ix = z.softmax(-1).topk(2, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        y = torch.zeros_like(x)
        for j in range(2):
            Wsel = self.experts[ix[:, j]]                    # [B,d,d]
            y = y + w[:, j:j + 1] * torch.bmm(Wsel, x.unsqueeze(2)).squeeze(2)
        return (y, z, ix) if ret_logits else y


def flip_rate(model, X, sigma, trials=8, seed=6):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        _, z, ix = model(X, ret_logits=True)
        base = ix.sort(-1).values
        flips = 0
        for _ in range(trials):
            Xn = X + torch.randn(X.shape, generator=g) * sigma
            _, _, ixn = model(Xn, ret_logits=True)
            flips += (ixn.sort(-1).values != base).any(-1).float().mean().item()
    return flips / trials


def gate5():
    torch.manual_seed(4)
    teacher = ToyMoE(seed=21)
    X = torch.randn(4096, 32)
    with torch.no_grad():
        Y = teacher(X)
    sigma = 0.05 * X.std().item()

    def fit(margin_w, steps=500, seed=13):
        torch.manual_seed(seed)
        m = ToyMoE(seed=99)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for st in range(steps):
            i = torch.randint(0, X.shape[0], (256,))
            y, z, ix = m(X[i], ret_logits=True)
            loss = F.mse_loss(y, Y[i])
            if margin_w > 0:
                zs = z.sort(-1, descending=True).values
                gap = zs[:, 1] - zs[:, 2]                    # k-th selected vs best unselected
                loss = loss + margin_w * F.relu(0.30 * z.std().detach() - gap).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            mse_clean = F.mse_loss(m(X), Y).item()
            g = torch.Generator().manual_seed(42)
            y0 = m(X)
            inst = 0.0
            for _ in range(4):    # output instability: ||m(x+ε)−m(x)|| / ||m(x)||
                yn = m(X + torch.randn(X.shape, generator=g) * sigma)
                inst += ((yn - y0).norm() / y0.norm()).item()
        return m, mse_clean, inst / 4

    m0, c0, i0 = fit(0.0)
    m1, c1, i1 = fit(0.03)
    f0, f1 = flip_rate(m0, X[:1024], sigma), flip_rate(m1, X[:1024], sigma)
    ok = f1 < f0 * 0.75 and c1 < c0 * 1.10 and i1 < i0
    gate("G5_router_margin", ok,
         {"flip_rate": {"no_margin": round(f0, 4), "margin": round(f1, 4)},
          "mse_clean": {"no_margin": round(c0, 5), "margin": round(c1, 5)},
          "output_instability": {"no_margin": round(i0, 5), "margin": round(i1, 5)},
          "note": "margin phạt gap (logit chọn thứ k) − (logit bị loại tốt nhất). "
                  "Tiêu chí chính: flip −25%+ (đại lượng đích của finding routing); "
                  "guard: MSE sạch +≤10%, instability ||m(x+ε)−m(x)|| phải GIẢM (ở toy "
                  "thành phần nhiễu-mượt chiếm ưu thế nên chỉ đòi chiều đúng; ở 30B flip "
                  "= đổi NGUYÊN expert nên kỳ vọng tác động lớn hơn nhiều)."})


def main():
    t0 = time.time()
    log("EXP AA — toy gates cho sub-bit (rollout-KD + curriculum + router margin)")
    gate1()
    gate2()
    gate3()
    gate5()
    RESULTS["_meta"] = {"sec_total": round(time.time() - t0, 1),
                        "torch": torch.__version__}
    p = os.path.join(OUT_DIR, "exp_aa_results.json")
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    n_pass = sum(1 for k, v in RESULTS.items() if isinstance(v, dict) and v.get("pass"))
    n_all = sum(1 for k, v in RESULTS.items() if isinstance(v, dict) and "pass" in v)
    log(f"XONG {n_pass}/{n_all} gate PASS — {p}")


if __name__ == "__main__":
    main()
