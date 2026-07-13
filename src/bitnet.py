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
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        x = x + (activation_quant(x) - x).detach()
        if self._wq_frozen is not None:
            w = self._wq_frozen
        else:
            w = self.weight
            w = w + (weight_quant(w) - w).detach()
        return F.linear(x, w)


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
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
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
        g = F.relu(self.gate(h)).square()              # relu(gate)^2
        h = g * self.up(h)                             # * up
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

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        x = self.embed(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin)
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
