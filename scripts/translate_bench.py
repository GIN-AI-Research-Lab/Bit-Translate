#!/usr/bin/env python3
"""Dịch bộ bench (bench_new.jsonl) bằng checkpoint .pt hoặc GGUF — ghi hyp, KHÔNG cần ref.

Khác `hardbench_ckpt.py`: bộ bench mới không có bản dịch tham chiếu (judge mù chấm
thẳng từ câu nguồn), nên script này chỉ sinh hyp.

  python scripts/translate_bench.py D:/.../v3_avg.pt --label v3
  python scripts/translate_bench.py D:/.../v2_avg3.pt --label v2
  python scripts/translate_bench.py checkpoints/kd_avg5.pt --label 100m
  # GGUF (292M) chạy qua WSL bitnet.cpp:
  python scripts/translate_bench.py dist/w0_i2s.gguf --label 292m --gguf
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def run_ckpt(probe, ckpt, beam=0):
    import torch
    import sentencepiece as spm
    from bitnet import BitNetLM, BitNetConfig
    from text_norm import normalize_for_model
    torch.set_num_threads(6)
    sp = spm.SentencePieceProcessor(
        model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    bos, eos, vie = sp.bos_id(), sp.eos_id(), sp.piece_to_id(">>vie<<")
    ck = torch.load(ckpt, map_location="cpu")
    c = ck["cfg"]
    m = BitNetLM(BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"],
                              n_layers=c["n_layers"], n_heads=c["n_heads"],
                              d_ff=c["d_ff"], max_seq=c["max_seq"])).eval()
    m.load_state_dict(ck["model"])
    m.freeze_for_inference()
    print(f"  {c['n_layers']}L d{c['d_model']} | {m.num_params()/1e6:.1f}M", flush=True)
    out, t0 = [], time.time()
    for i, p in enumerate(probe):
        ids = torch.tensor([[bos, vie] + sp.encode(normalize_for_model(p["src"])) + [eos]])
        with torch.inference_mode():
            o = m.generate_cached(ids, max_new_tokens=200, eos_id=eos, rep_penalty=1.0)
        g = o[0, ids.shape[1]:].tolist()
        if eos in g:
            g = g[:g.index(eos)]
        out.append(sp.decode(g))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(probe)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def run_gguf(probe, gguf):
    """GGUF qua llama-server, đưa TOKEN IDS tokenize sẵn — KHÔNG để llama.cpp tự tokenize.

    Vì sao (ISSUES.md #4, đóng đinh 2026-07-29): spm_vija_32k.model là UNIGRAM, nhưng
    llm_tokenizer_spm của llama.cpp là greedy bigram-merge kiểu BPE — không tái tạo được
    phân đoạn unigram. Từ thường gặp bị xé vụn TRƯỚC khi vào model (してください 1594 ->
    して+く+だ+さい; ツイート 17062 -> ツ+イー+ト), ids lệch 30/30 câu, +19,4% token,
    kèm token ▁(262) bị đúp. Hậu quả: mọi format GGUF mất 16-22 điểm judge so PyTorch.
    Kiểm nhân quả HAI CHIỀU n=30: PyTorch ăn ids-của-llama -> tái tạo output hỏng
    (chrF 96,1); GGUF ăn ids chuẩn qua /completion -> hồi phục về mức PyTorch (chrF 97,0).

    Đường đúng duy nhất: tokenize bằng sentencepiece + normalize_for_model (y hệt lúc
    train) rồi POST mảng ids vào /completion. Model nạp MỘT lần -> nhanh hơn ~10x so
    spawn llama-cli mỗi câu. -t 4..8 (14 luồng sụp 0,32 tok/s, đo 2026-07-28)."""
    import urllib.error
    import urllib.request

    import sentencepiece as spm
    from text_norm import normalize_for_model

    BIN = "/home/tuent/BitNet-test/build/bin/llama-server"
    PORT = 8807
    wsl_model = "/mnt/" + str(gguf).replace(":", "").replace("\\", "/").lower()[0] + \
                str(gguf).replace(":", "").replace("\\", "/")[1:]
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "spm_vija_32k.model"))
    bos, eos = sp.piece_to_id("<s>"), sp.piece_to_id("</s>")
    vie = sp.piece_to_id(">>vie<<")

    srv = subprocess.Popen(
        ["wsl", "-e", "bash", "-lc",
         f"exec {BIN} -m {wsl_model} -t 4 -c 256 --host 127.0.0.1 --port {PORT} 2>/dev/null"])
    try:
        # Chờ server sẵn sàng. HTTPError (404...) nghĩa là server ĐÃ nhận kết nối
        # (endpoint /health có thể không tồn tại ở bản cũ) -> cũng coi là sẵn sàng.
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server không lên sau 120s")

        out, t0 = [], time.time()
        for i, p in enumerate(probe):
            ids = [bos, vie] + sp.encode(normalize_for_model(p["src"])) + [eos]
            body = json.dumps({"prompt": ids, "n_predict": 200, "temperature": 0.0,
                               "repeat_penalty": 1.0, "cache_prompt": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.loads(r.read().decode("utf-8"))
            out.append(j["content"].strip())
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(probe)} ({time.time()-t0:.0f}s)", flush=True)
        return out
    finally:
        srv.terminate()
        subprocess.run(["wsl", "-e", "bash", "-lc", "pkill -f llama-server"],
                       capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--label", required=True)
    ap.add_argument("--probe", default=str(ROOT / "eval" / "bench_new.jsonl"))
    ap.add_argument("--gguf", action="store_true")
    a = ap.parse_args()

    probe = [json.loads(l) for l in open(a.probe, encoding="utf-8")]
    print(f"[{a.label}] dịch {len(probe)} câu...", flush=True)
    hyps = run_gguf(probe, a.model) if a.gguf else run_ckpt(probe, a.model)

    outp = ROOT / "eval" / f"bench_{a.label}.jsonl"
    with open(outp, "w", encoding="utf-8") as f:
        for p, h in zip(probe, hyps):
            f.write(json.dumps({**p, "hyp": h, "sys": a.label}, ensure_ascii=False) + "\n")
    print(f"[{a.label}] -> {outp}")


if __name__ == "__main__":
    main()
