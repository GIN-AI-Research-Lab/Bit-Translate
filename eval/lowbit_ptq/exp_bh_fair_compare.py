# -*- coding: utf-8 -*-
"""
So SANH CONG BANG: config sensitivity cua ta vs Q2_K off-the-shelf, tren CUNG text ppl_big.txt,
lay ti le suy giam vs Q4_K_M TRONG CUNG framework (PyTorch cho ta, llama.cpp cho Q2_K).
Tra loi cau then chot: config tu che (2.71bpw, can kernel) co THANG Q2_K (2.6bpw, kernel san,
llama.cpp do x1.12) khong -> co dang tu viet kernel khong.

Chay: python eval/lowbit_ptq/exp_bh_fair_compare.py
"""
import gc, io, json, math, os, sys, time
import gguf, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_bg_real_measure import (GGUF_PATH, N_LAYERS, DTYPE, dq, ternary_2d, intN_2d, q3,
                                 build_and_compress)   # tai dung build+nen y het exp_bg

SC = "C:/Users/ADMINI~1/AppData/Local/Temp/claude/F--Project-Ai-Bit-Translate/0e490a69-c8cc-420a-b588-0830d420f6f6/scratchpad/ppl_big.txt"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bh_results.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def ppl_on_file(model, tok, path, ctx=512):
    text = io.open(path, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0]
    # chunk theo ctx nhu llama-perplexity (non-overlapping), trung binh NLL
    nlls, ntok = [], 0
    for i in range(0, ids.shape[0] - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if chunk.shape[0] < 2:
            break
        inp = chunk[:-1].unsqueeze(0)
        tgt = chunk[1:]
        lg = model(inp).logits[0].float()
        nll = F.cross_entropy(lg, tgt, reduction="sum")
        nlls.append(nll.item())
        ntok += tgt.shape[0]
    return math.exp(sum(nlls) / ntok)


@torch.no_grad()
def build_baseline(byname):
    """Y het build_and_compress nhung KHONG nen (Q4_K_M-dequant fp16 thuan)."""
    cfg = OlmoeConfig(vocab_size=50304, hidden_size=2048, intermediate_size=1024,
                      num_hidden_layers=N_LAYERS, num_attention_heads=16, num_key_value_heads=16,
                      max_position_embeddings=4096, rms_norm_eps=1e-5, num_experts_per_tok=8,
                      num_experts=64, norm_topk_prob=False, tie_word_embeddings=False, attention_bias=False)
    with torch.device("meta"):
        model = OlmoeForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(DTYPE)
    hd = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
    model.model.rotary_emb.inv_freq.copy_(inv); model.model.rotary_emb.original_inv_freq.copy_(inv)
    sd = model.state_dict()
    def put(h, g): sd[h].copy_(dq(byname, g, DTYPE))
    put("model.embed_tokens.weight", "token_embd.weight"); put("model.norm.weight", "output_norm.weight")
    put("lm_head.weight", "output.weight")
    for L in range(N_LAYERS):
        for hn, gn in [("self_attn.q_proj","attn_q"),("self_attn.k_proj","attn_k"),("self_attn.v_proj","attn_v"),
                       ("self_attn.o_proj","attn_output"),("self_attn.q_norm","attn_q_norm"),("self_attn.k_norm","attn_k_norm"),
                       ("input_layernorm","attn_norm"),("post_attention_layernorm","ffn_norm"),("mlp.gate","ffn_gate_inp")]:
            put(f"model.layers.{L}.{hn}.weight", f"blk.{L}.{gn}.weight")
        g_=dq(byname,f"blk.{L}.ffn_gate_exps.weight",DTYPE); u_=dq(byname,f"blk.{L}.ffn_up_exps.weight",DTYPE)
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"].copy_(torch.cat([g_,u_],dim=1)); del g_,u_
        sd[f"model.layers.{L}.mlp.experts.down_proj"].copy_(dq(byname,f"blk.{L}.ffn_down_exps.weight",DTYPE))
    gc.collect(); model.eval(); return model


def main():
    r = gguf.GGUFReader(GGUF_PATH); byname = {t.name: t for t in r.tensors}
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

    log("build BASELINE (Q4_K_M-dequant fp16 thuan)...")
    m = build_baseline(byname)
    ppl_base = ppl_on_file(m, tok, SC)
    log(f"  baseline PPL (cung text ppl_big) = {ppl_base:.4f}")
    del m; gc.collect()

    log("build config SENSITIVITY cua ta (down=tern+gate_up=int3+attn=int4+emb/head=int4)...")
    m = build_and_compress(byname)
    ppl_ours = ppl_on_file(m, tok, SC)
    log(f"  our-config PPL (cung text) = {ppl_ours:.4f}  -> x{ppl_ours/ppl_base:.3f} vs baseline")
    del m; gc.collect()

    # llama.cpp so lieu (do o buoc truoc, cung text)
    llama = {"Q4_K_M": 2.386, "Q3_K_M": 2.423, "Q2_K": 2.679}
    q2_ratio = llama["Q2_K"] / llama["Q4_K_M"]
    our_ratio = ppl_ours / ppl_base

    out = {"text": "ppl_big.txt (cung file voi llama-perplexity)",
           "pytorch": {"baseline_q4dequant": round(ppl_base, 4), "our_config": round(ppl_ours, 4),
                       "our_ratio_vs_q4": round(our_ratio, 3), "our_bpw": 2.71},
           "llamacpp": {"Q4_K_M": llama["Q4_K_M"], "Q3_K_M": llama["Q3_K_M"], "Q2_K": llama["Q2_K"],
                        "Q2K_ratio_vs_q4": round(q2_ratio, 3), "Q2K_bpw": 2.6}}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== SO SANH CONG BANG (cung text, ti le vs Q4 trong cung framework) ===")
    log(f"  Config CUA TA (PyTorch): x{our_ratio:.3f} @ 2.71bpw (CAN tu viet kernel)")
    log(f"  Q2_K off-the-shelf (llama.cpp): x{q2_ratio:.3f} @ 2.6bpw (kernel SAN, 56 tok/s)")
    if our_ratio <= q2_ratio + 0.02:
        log(f"  => Ta NGANG/THANG Q2_K -> co the dang tu viet kernel (nhung Q2_K it bit hon + co san).")
    else:
        log(f"  => Q2_K TOT HON (it bit hon, chat luong tot hon, co kernel san). Config tu che KHONG "
            f"dang tu viet kernel cho OLMoE - dung thang Q2_K la thuc dung nhat.")


if __name__ == "__main__":
    main()
