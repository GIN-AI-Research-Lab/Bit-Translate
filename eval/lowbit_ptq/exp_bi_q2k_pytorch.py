# -*- coding: utf-8 -*-
"""
So CHINH XAC config ta vs Q2_K CUNG framework: dequant file Q2_K GGUF ra fp16, chay PPL PyTorch
tren cung text ppl_big -> so truc tiep voi config ta (7.89) va baseline Q4-dequant (5.01) da do
o exp_bh. Bo nhieu framework (truoc: ta PyTorch x1.57 vs Q2_K llama.cpp x1.12 - lech framework).

Chay: python eval/lowbit_ptq/exp_bi_q2k_pytorch.py
"""
import gc, io, json, math, os, sys, time
import gguf, torch, torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_bh_fair_compare import ppl_on_file, SC
from exp_bg_real_measure import N_LAYERS, DTYPE, dq

Q2K_GGUF = "E:/hf_gguf_cache/olmoe-1b-7b-0924-q2_k.gguf"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_bi_results.json")
# so lieu tu exp_bh (CUNG text, CUNG framework PyTorch)
BASE_Q4DEQUANT = 5.0108
OUR_CONFIG = 7.8888


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def build_from_gguf(byname):
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
        if L % 8 == 0: log(f"  layer {L} dequant xong")
    gc.collect(); model.eval(); return model


def main():
    r = gguf.GGUFReader(Q2K_GGUF); byname = {t.name: t for t in r.tensors}
    tok = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    log("dequant Q2_K -> fp16, build model...")
    m = build_from_gguf(byname)
    ppl_q2k = ppl_on_file(m, tok, SC)
    log(f"  Q2_K-dequant PPL (cung text, PyTorch) = {ppl_q2k:.4f}")

    out = {"framework": "PyTorch, cung text ppl_big, cung baseline Q4-dequant",
           "baseline_q4dequant": BASE_Q4DEQUANT,
           "q2k_dequant": round(ppl_q2k, 4), "q2k_ratio": round(ppl_q2k / BASE_Q4DEQUANT, 3), "q2k_bpw": 2.6,
           "our_config": OUR_CONFIG, "our_ratio": round(OUR_CONFIG / BASE_Q4DEQUANT, 3), "our_bpw": 2.71}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")

    log("\n=== SO CHINH XAC (cung PyTorch, cung text, cung baseline) ===")
    log(f"  baseline Q4-dequant: {BASE_Q4DEQUANT:.2f}")
    log(f"  Q2_K (2.6bpw):       {ppl_q2k:.2f}  (x{ppl_q2k/BASE_Q4DEQUANT:.2f})")
    log(f"  Config ta (2.71bpw): {OUR_CONFIG:.2f}  (x{OUR_CONFIG/BASE_Q4DEQUANT:.2f})")
    if ppl_q2k < OUR_CONFIG:
        log(f"  => Q2_K TOT HON ta {OUR_CONFIG/ppl_q2k:.2f}x o cung framework (khang dinh: khong lech framework)")
    else:
        log(f"  => Ta tot hon Q2_K {ppl_q2k/OUR_CONFIG:.2f}x (dao nguoc ket luan cross-framework truoc!)")


if __name__ == "__main__":
    main()
