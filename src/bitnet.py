"""BitNet b1.58 decoder-only Transformer — bitnet.cpp-COMPATIBLE architecture.

Matches the "bitnet-b1.58" arch that bitnet.cpp/llama.cpp implements in
build_bitnet_158() (verified against Microsoft's BitNet-b1.58-2B-4T, which runs
correctly on this build; the older "bitnet"/build_bitnet SiLU graph is BROKEN
here). A trained checkpoint converts to GGUF (arch "bitnet-b1.58") and runs on
bitnet.cpp's fast ternary CPU kernels. Per block:
    attn_norm  -> q,k,v  (shared RMSNorm before q/k/v; the projections are
                          plain BitLinear with NO internal norm)
    attn_sub_norm -> o   (RMSNorm on attention output before o_proj)
    ffn_norm   -> gate,up
    ffn_sub_norm -> down (RMSNorm on the squared-ReLU hidden before down_proj)
FFN is gated squared-ReLU: down( relu(gate(x))^2 * up(x) )  (NOT SwiGLU).
Final output_norm -> lm_head (tied to token embedding).

Quantization-aware training: master weights stay bf16/fp32; forward fake-quantizes
with a straight-through estimator (ternary weights, 8-bit per-token activations).
The 1.58-bit benefit is realized at inference/packing.

Tối ưu tốc độ train (BẬT mặc định, mọi thứ tương đương toán học CHÍNH XÁC với bản
gốc — parity Δloss=0, gradient khớp; đã guard để INFERENCE/GGUF bit-identical):
  BITNET_OPT=off (hoặc rỗng) -> tắt hết, quay về hành vi gốc từng-bit.
  - ste       : STE bằng autograd.Function (bỏ tensor fp32 trung gian -> thủ phạm
                "eager OOM 292M"); CHỈ khi grad bật, eval -> quant thẳng (giá trị y hệt).
  - wqcache   : cache ternary weight 1 lần/optimizer-step (thay 32-64 lần/step);
                train loop gọi model.attach_wq_autorefresh(opt, bf16). Eval không đụng.
  - fusedproj : gộp GEMM q/k/v & gate/up + quantize activation 1 lần; CHỈ khi training
                (generate/generate_cached/GGUF đi đường 3-GEMM gốc -> bit-identical).
  - maskce    : lm_head+CE chỉ trên vị trí loss_mask=True; CHỈ path targets+mask (train).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_OPT = os.environ.get("BITNET_OPT", "ste,wqcache,fusedproj,maskce")
_OPT = set() if _OPT.strip() in ("", "off") else set(f.strip() for f in _OPT.split(","))
USE_STE_FN = "ste" in _OPT
USE_WQ_CACHE = "wqcache" in _OPT
USE_FUSED_PROJ = "fusedproj" in _OPT
USE_MASK_CE = "maskce" in _OPT


def activation_quant(x):
    """8-bit, per-token (last-dim) absmax. Computed in fp32 for stability."""
    xf = x.float()
    scale = 127.0 / xf.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    q = (xf * scale).round().clamp_(-128, 127) / scale
    return q.to(x.dtype)


def weight_quant(w):
    """Ternary {-1,0,1} via absmean scaling. Computed in fp32."""
    wf = w.float()
    scale = 1.0 / wf.abs().mean().clamp_(min=1e-5)
    q = (wf * scale).round().clamp_(-1, 1) / scale
    return q.to(w.dtype)


class _ActQuantSTE(torch.autograd.Function):
    """y = activation_quant(x); backward: dL/dx = dL/dy (identity STE).
    forward chạy grad-disabled -> không giữ tensor fp32 trung gian như trick
    'x + (q(x)-x).detach()'. dynamo trace qua HOP autograd_function_apply (torch>=2.1)."""

    @staticmethod
    def forward(ctx, x):
        return activation_quant(x)

    @staticmethod
    def backward(ctx, g):
        return g


class _WeightSTE(torch.autograd.Function):
    """forward trả ternary weight ĐÃ TÍNH SẴN q (cache dùng chung mọi microbatch
    của 1 optimizer step; có thể bf16); backward đẩy grad nguyên vẹn về master
    weight fp32 (STE identity), cast về dtype master."""

    @staticmethod
    def forward(ctx, w, q):
        ctx.w_dtype = w.dtype
        return q

    @staticmethod
    def backward(ctx, g):
        return g.to(ctx.w_dtype), None


def _ste_act(x):
    # eval/no-grad: giá trị == activation_quant(x) y hệt, khỏi dựng Function.
    if USE_STE_FN and torch.is_grad_enabled():
        return _ActQuantSTE.apply(x)
    return x + (activation_quant(x) - x).detach()


def _ste_weight(w, cache):
    if cache is not None:
        if USE_STE_FN and torch.is_grad_enabled():
            return _WeightSTE.apply(w, cache)
        return w + (cache - w).detach()          # trick gốc, khỏi tính lại quant
    if USE_STE_FN and torch.is_grad_enabled():
        return _WeightSTE.apply(w, weight_quant(w.detach()))
    return w + (weight_quant(w) - w).detach()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class BitLinear(nn.Module):
    """Ternary-weight, 8-bit-activation linear. NO internal norm (the block
    applies the shared/sub RMSNorm before this), no bias — matches bitnet.cpp."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self._wq_frozen = None  # inference: precomputed ternary weight
        self._wq_cache = None   # train: ternary weight cache trong 1 optimizer step

    def quantized_weight(self):
        if self._wq_frozen is not None:
            return self._wq_frozen
        return _ste_weight(self.weight, self._wq_cache if USE_WQ_CACHE else None)

    def forward(self, x, x_prequant=False):
        # x_prequant=True: input đã activation_quant sẵn (fused proj dùng chung).
        if not x_prequant:
            x = _ste_act(x)
        return F.linear(x, self.quantized_weight())


def precompute_rope(head_dim, max_seq, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.attn_norm = RMSNorm(cfg.d_model)          # shared, before q/k/v
        self.wq = BitLinear(cfg.d_model, cfg.d_model)
        self.wk = BitLinear(cfg.d_model, cfg.d_model)
        self.wv = BitLinear(cfg.d_model, cfg.d_model)
        self.attn_sub_norm = RMSNorm(cfg.d_model)      # before o_proj
        self.wo = BitLinear(cfg.d_model, cfg.d_model)

    def _qkv(self, h):
        B, T, _ = h.shape
        if USE_FUSED_PROJ and self.training:
            # quantize activation 1 lần + gộp wq/wk/wv thành 1 GEMM [d->3d].
            # Eval/inference KHÔNG vào đây -> 3-GEMM gốc, GGUF bit-identical.
            hq = _ste_act(h)
            w_all = torch.cat([self.wq.quantized_weight(),
                               self.wk.quantized_weight(),
                               self.wv.quantized_weight()], dim=0)
            q, k, v = F.linear(hq, w_all).chunk(3, dim=-1)
        else:
            q, k, v = self.wq(h), self.wk(h), self.wv(h)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x, cos, sin):
        h = self.attn_norm(x)
        q, k, v = self._qkv(h)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.wo(self.attn_sub_norm(out))


class FFN(nn.Module):
    """Gated squared-ReLU FFN = the OFFICIAL BitNet b1.58 recipe (arch
    "bitnet-b1.58" / build_bitnet_158): down( relu(gate(x))^2 * up(x) ).
    NOT SwiGLU/SiLU — bitnet.cpp's working graph hardcodes LLM_FFN_RELU_SQR,
    and the SiLU "bitnet" graph (build_bitnet) is broken on this build."""

    def __init__(self, cfg):
        super().__init__()
        self.ffn_norm = RMSNorm(cfg.d_model)           # shared, before gate/up
        self.gate = BitLinear(cfg.d_model, cfg.d_ff)
        self.up = BitLinear(cfg.d_model, cfg.d_ff)
        self.ffn_sub_norm = RMSNorm(cfg.d_ff)          # before down
        self.down = BitLinear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        h = self.ffn_norm(x)
        if USE_FUSED_PROJ and self.training:
            hq = _ste_act(h)
            w_all = torch.cat([self.gate.quantized_weight(),
                               self.up.quantized_weight()], dim=0)
            g_pre, u = F.linear(hq, w_all).chunk(2, dim=-1)
        else:
            g_pre, u = self.gate(h), self.up(h)
        g = F.relu(g_pre).square()                     # relu(gate)^2
        h = g * u                                      # * up
        return self.down(self.ffn_sub_norm(h))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(x, cos, sin)
        x = x + self.ffn(x)
        return x


class BitNetConfig:
    def __init__(self, vocab_size=32000, d_model=768, n_layers=12, n_heads=12,
                 d_ff=2048, max_seq=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq = max_seq


class BitNetLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)  # tied lm_head; keep logits sane
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.output_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        cos, sin = precompute_rope(cfg.d_model // cfg.n_heads, cfg.max_seq)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = False
        self.ckpt_every_k = 1   # >1: chỉ checkpoint block i%k==0 (selective GC)

    @torch.no_grad()
    def refresh_wq_cache(self, dtype=None):
        """Tính lại ternary cache cho mọi BitLinear; gọi 1 lần trước train và
        sau mỗi opt.step(). dtype=bf16 khi autocast bf16 (cache 2B/param, khỏi
        cast lại mỗi microbatch). copy_ vào CÙNG buffer -> torch.compile không
        recompile (metadata cache không đổi)."""
        for m in self.modules():
            if isinstance(m, BitLinear):
                wq = weight_quant(m.weight)
                if dtype is not None:
                    wq = wq.to(dtype)
                if m._wq_cache is None or m._wq_cache.dtype != wq.dtype \
                        or m._wq_cache.shape != wq.shape:
                    m._wq_cache = wq
                else:
                    m._wq_cache.copy_(wq)

    def attach_wq_autorefresh(self, opt, dtype=None):
        """Opt-in từ train loop: refresh ngay (TRƯỚC compile để guard nhánh ổn
        định) + hook tự refresh sau mỗi opt.step()."""
        if not USE_WQ_CACHE:
            return None
        self.refresh_wq_cache(dtype)
        return opt.register_step_post_hook(
            lambda optimizer, args, kwargs: self.refresh_wq_cache(dtype))

    def clear_wq_cache(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m._wq_cache = None

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for i, blk in enumerate(self.blocks):
            ck = (self.gradient_checkpointing and self.training
                  and (self.ckpt_every_k <= 1 or i % self.ckpt_every_k == 0))
            if ck:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
        # masked-CE: chỉ lm_head+CE trên vị trí loss_mask=True (train). Giá trị
        # loss/grad y hệt (sum hàng chọn ÷ n == sum(l*m)÷m.sum()); tiết kiệm
        # FLOPs lm_head + ~0.6-0.8GB VRAM logits fp32. Path targets=None (infer)
        # và path không-mask giữ nguyên -> logits đầy đủ.
        if targets is not None and loss_mask is not None and USE_MASK_CE and self.training:
            sel = loss_mask.reshape(-1).nonzero(as_tuple=True)[0]
            n = sel.numel()
            if n == 0:
                return None, (x.sum() * 0.0).float()
            xs = x.reshape(B * T, -1).index_select(0, sel)
            logits_s = self.lm_head(self.output_norm(xs))
            ts = targets.reshape(-1).index_select(0, sel)
            loss = F.cross_entropy(logits_s.float(), ts, reduction="sum") / n
            return None, loss
        logits = self.lm_head(self.output_norm(x))
        loss = None
        if targets is not None:
            l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                targets.reshape(-1), reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (l * m).sum() / m.sum().clamp_(min=1.0)
            else:
                loss = l.mean()
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def freeze_for_inference(self):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m._wq_frozen = weight_quant(m.weight).detach()
        return self

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, eos_id, temperature=0.0, top_k=0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq:])
            logits = logits[:, -1, :]
            if temperature <= 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return idx

    # ---- Inference-only KV-cache decoding (does NOT affect the training path) ----
    def _attn_step(self, attn, x, cos, sin, past_kv):
        B, T, _ = x.shape
        H, Dh = attn.n_heads, attn.head_dim
        h = attn.attn_norm(x)
        q = attn.wq(h).view(B, T, H, Dh).transpose(1, 2)
        k = attn.wk(h).view(B, T, H, Dh).transpose(1, 2)
        v = attn.wv(h).view(B, T, H, Dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        is_causal = past_kv is None and T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return attn.wo(attn.attn_sub_norm(out)), (k, v)

    @torch.no_grad()
    def generate_cached(self, idx, max_new_tokens, eos_id, rep_penalty=1.0):
        caches = [None] * len(self.blocks)
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            T = cur.shape[1]
            x = self.embed(cur)
            cos, sin = self.rope_cos[pos:pos + T], self.rope_sin[pos:pos + T]
            for i, blk in enumerate(self.blocks):
                a, kv = self._attn_step(blk.attn, x, cos, sin, caches[i])
                x = x + a
                x = x + blk.ffn(x)
                caches[i] = kv
            logits = self.lm_head(self.output_norm(x)[:, -1, :])
            if rep_penalty != 1.0:
                for t in set(idx[0].tolist()):
                    logits[0, t] /= rep_penalty
            nxt = logits.argmax(-1, keepdim=True)
            pos += T
            cur = nxt
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt) == eos_id:
                break
        return idx
