# -*- coding: utf-8 -*-
"""
Probe giả thuyết 2B (contextual sparsity) tren Qwen3-0.6B DENSE (khong phai MoE - 2A adaptive
top-k khong ap dung duoc vi dense khong co expert/router). CHUA xay bo du doan - chi DO tien
de: cau truc contextual-sparsity co TON TAI du de dang theo duoi khong?

Do gi (input cua down_proj = SiLU(gate(x))*up(x), dai 3072/layer):
1. TAP TRUNG NANG LUONG: moi token, sort |neuron| giam dan, tim % neuron can giu de dat
   90/95/99% nang luong L2 (nang luong = cai quyet dinh output y = down @ intermediate).
   Giu it neuron = bo qua nhieu cot down_proj = tiet kiem byte. Day la TRAN ly thuyet.
2. TINH CONTEXTUAL: tap neuron "quan trong" co DOI theo token khong?
   - static ceiling: dung importance TRUNG BINH toan bo token lam mask CO DINH, giu cung so
     neuron, do nang luong/token giu lai. Neu ~ oracle per-token -> sparsity TINH (chan,
     prune tinh la du). Neu kem han nhieu -> CONTEXTUAL (dang xay du doan per-token).
   - Jaccard giua tap top-k cua cac cap token ngau nhien (thap = contextual).

Luu y trung thuc: SiLU KHONG cho 0 cung nhu ReLU -> "sparsity" o day la MEM (nang luong tap
trung), khong phai 0 tuyet doi. Do dung bang nang luong, khong dem so 0.

Chay: python eval/lowbit_ptq/exp_ap_activation_sparsity.py
"""
import io
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = ("F:/Project Ai/Quantazation/hf_cache/hub/models--Qwen--Qwen3-0.6B/"
             "snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_aw_results.json")

CALIB_TEXTS = [
    "The city council approved funding for a new transit line connecting downtown to the airport.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2",
    "The stock market fell sharply after the central bank signaled higher interest rates.",
    "Hôm nay trời mưa to, tôi phải mang ô đi làm và đường phố rất đông người qua lại.",
    "今日は天気がいいので、公園へ散歩に行きましょう。桜がとても綺麗に咲いています。",
    "Researchers found that regular exercise reduces the risk of cardiovascular disease.",
    "Quantum computers use qubits that can exist in superposition of multiple states at once.",
    "Việc học một ngôn ngữ mới đòi hỏi sự kiên nhẫn và luyện tập đều đặn mỗi ngày.",
    "The ancient library held thousands of scrolls documenting the history of the empire.",
    "彼は毎朝six時に起きて、コーヒーを飲みながら新聞を読むのが日課です。",
    "Climate models predict rising sea levels will affect coastal cities within decades.",
    "The chef prepared a delicate sauce by slowly reducing the wine over low heat.",
    "Máy tính lượng tử có thể giải một số bài toán nhanh hơn máy tính cổ điển rất nhiều.",
    "A gentle breeze carried the scent of pine through the quiet mountain valley at dawn.",
    "The negotiation concluded with both parties agreeing to a phased reduction of tariffs.",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("tai Qwen3-0.6B dense (fp32 CPU, model nho ~1.2GB)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    NL = model.config.num_hidden_layers
    I = model.config.intermediate_size

    # hook input cua down_proj moi layer (= SiLU intermediate)
    caps = {L: [] for L in range(NL)}

    def mk(L):
        def h(mod, inp):
            caps[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
        return h

    handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk(L))
               for L in range(NL)]
    log(f"chay forward {len(CALIB_TEXTS)} doan calib (en/vi/ja/code) qua {NL} layer...")
    for t in CALIB_TEXTS:
        ids = tok(t, return_tensors="pt").input_ids
        model(ids)
    for h in handles:
        h.remove()

    # gop tat ca token cua tat ca layer? Khong - do RIENG tung layer roi tong hop, va
    # gop token de do contextual. Lam per-layer.
    frac90, frac95, frac99 = [], [], []
    static_ret_at30, jaccard30 = [], []
    per_layer = []
    g = torch.Generator().manual_seed(0)
    for L in range(NL):
        A = torch.cat(caps[L], 0)            # [T, I]
        T = A.shape[0]
        energy = A.pow(2)                     # [T, I]
        tot = energy.sum(1, keepdim=True).clamp(min=1e-12)
        sorted_e, sorted_idx = energy.sort(1, descending=True)
        cum = sorted_e.cumsum(1) / tot        # [T, I] cumulative energy fraction
        # so neuron can de dat nguong (trung binh tren token)
        k90 = (cum < 0.90).sum(1).float().mean().item() + 1
        k95 = (cum < 0.95).sum(1).float().mean().item() + 1
        k99 = (cum < 0.99).sum(1).float().mean().item() + 1
        frac90.append(k90 / I)
        frac95.append(k95 / I)
        frac99.append(k99 / I)

        # static ceiling: importance co dinh = mean energy tren token; giu top-30%, do nang
        # luong/token giu lai (so voi oracle per-token cung giu 30%)
        k30 = int(I * 0.30)
        static_imp = energy.mean(0)           # [I]
        static_top = static_imp.topk(k30).indices
        static_mask = torch.zeros(I, dtype=torch.bool)
        static_mask[static_top] = True
        static_energy_kept = (energy[:, static_mask].sum(1) / tot.squeeze(1)).mean().item()
        oracle_energy_kept = (sorted_e[:, :k30].sum(1) / tot.squeeze(1)).mean().item()
        static_ret_at30.append((static_energy_kept, oracle_energy_kept))

        # Jaccard giua top-30% set cua cac cap token ngau nhien
        top_sets = sorted_idx[:, :k30]        # [T, k30]
        pairs = min(50, T * (T - 1) // 2)
        jac = []
        for _ in range(pairs):
            i = int(torch.randint(0, T, (1,), generator=g))
            j = int(torch.randint(0, T, (1,), generator=g))
            if i == j:
                continue
            si = set(top_sets[i].tolist())
            sj = set(top_sets[j].tolist())
            jac.append(len(si & sj) / len(si | sj))
        jaccard30.append(sum(jac) / max(len(jac), 1))
        per_layer.append({"L": L, "frac90": round(k90 / I, 3), "frac95": round(k95 / I, 3),
                          "frac99": round(k99 / I, 3),
                          "static_kept@30%": round(static_energy_kept, 3),
                          "oracle_kept@30%": round(oracle_energy_kept, 3),
                          "jaccard30": round(jaccard30[-1], 3)})

    def avg(x):
        return sum(x) / len(x)

    log("\n=== KET QUA (trung binh tren 28 layer) ===")
    log(f"  % neuron can giu de dat 90% nang luong: {avg(frac90):.1%}")
    log(f"  % neuron can giu de dat 95% nang luong: {avg(frac95):.1%}")
    log(f"  % neuron can giu de dat 99% nang luong: {avg(frac99):.1%}")
    static_avg = avg([s for s, o in static_ret_at30])
    oracle_avg = avg([o for s, o in static_ret_at30])
    log(f"  Giu top-30% neuron: oracle per-token giu {oracle_avg:.1%} nang luong; "
        f"mask TINH (co dinh) giu {static_avg:.1%}")
    log(f"  Jaccard top-30% giua cap token ngau nhien: {avg(jaccard30):.2f} "
        f"(thap=contextual, cao=tinh)")

    log("\n=== DIEN GIAI ===")
    if avg(frac95) > 0.6:
        verdict_sparse = "KHONG thua - can giu >60% neuron cho 95% nang luong -> tran thap, khong dang"
    elif avg(frac95) > 0.4:
        verdict_sparse = "thua VUA - giu 40-60% neuron du 95% -> tran ~1.5-2x, dang xem xet"
    else:
        verdict_sparse = "thua MANH - <40% neuron du 95% -> tran >2x, DANG theo duoi"
    log(f"  Do thua: {verdict_sparse}")
    gap = oracle_avg - static_avg
    if gap > 0.15:
        verdict_ctx = f"CONTEXTUAL manh (oracle hon mask tinh {gap:.0%}) - phai du doan per-token"
    elif gap > 0.05:
        verdict_ctx = f"contextual vua (hon {gap:.0%}) - du doan per-token co loi vua"
    else:
        verdict_ctx = f"chu yeu TINH (chi hon {gap:.0%}) - prune tinh la du, khong can du doan"
    log(f"  Tinh contextual: {verdict_ctx}")

    out = {"model": "Qwen3-0.6B", "n_layers": NL, "intermediate": I,
           "avg_frac_neurons_for_energy": {"90%": round(avg(frac90), 4),
                                           "95%": round(avg(frac95), 4),
                                           "99%": round(avg(frac99), 4)},
           "top30pct_energy_kept": {"oracle_per_token": round(oracle_avg, 4),
                                    "static_fixed_mask": round(static_avg, 4)},
           "avg_jaccard_top30": round(avg(jaccard30), 4),
           "verdict_sparsity": verdict_sparse, "verdict_contextual": verdict_ctx,
           "per_layer": per_layer}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"da ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
