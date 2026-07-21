#!/usr/bin/env python3
"""Đo headroom rerank: sinh N bản dịch bằng sampling cho FLORES ja2vi, so 3 mức:
  - greedy  : bản dịch temperature=0 hiện tại (những gì đã judge)
  - rerank  : trong (greedy + N sample) chọn bản có LaBSE cosine(src, cand) cao nhất
              — CHẠY ĐƯỢC THẬT lúc inference, không cần biết ref
  - oracle  : bản chrF cao nhất so ref — TRẦN lý thuyết của rerank (không đạt được
              thật, chỉ để biết trong weights còn giấu bao nhiêu chất lượng)

Trả lời câu hỏi: "model đã BIẾT dịch tốt hơn nhưng greedy không lấy ra được,
hay là không biết?" — nếu oracle >> greedy thì rerank/decoding là đòn bẩy thật;
nếu oracle ≈ greedy thì vấn đề là kiến thức trong weights, decoding vô ích.

  LLAMA_SERVER=~/BitNet-test/build/bin/llama-server \
  python3 eval/rerank_probe.py <gguf> <label> [n_samples]

Ghi eval/rerank_probe_<label>.jsonl (mọi candidate + điểm) + in tổng kết.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from text_norm import normalize_for_model  # noqa: E402

SERVER = os.environ.get("LLAMA_SERVER", "/home/tuent/BitNet-test/build/bin/llama-server")
PORT = 8082
UP = f"http://127.0.0.1:{PORT}"
BOS, EOS, VIE = 2, 3, 4
ROOT = Path(__file__).parent.parent
GGUF, LABEL = sys.argv[1], sys.argv[2]
N_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def up(path, payload):
    req = urllib.request.Request(UP + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    # ĐỀ: 100 câu FLORES ja2vi (src/ref từ file google, greedy hyp từ file v3)
    google = [json.loads(l) for l in (ROOT / "eval" / "hardbench_flores_google.jsonl").open(encoding="utf-8")]
    v3 = {r["id"]: r for r in (json.loads(l) for l in (ROOT / "eval" / "hardbench_v3_flores.jsonl").open(encoding="utf-8"))}
    probes = [{"id": r["id"], "src": r["src"], "ref": r["ref"], "greedy": v3[r["id"]]["hyp"]}
              for r in google if r["dir"] == "ja2vi"]
    print(f"{len(probes)} câu, {N_SAMPLES} sample/câu", flush=True)

    proc = subprocess.Popen([SERVER, "-m", GGUF, "--host", "127.0.0.1", "--port", str(PORT),
                             "-t", "6", "-c", "384", "--parallel", "1"],
                            stdout=open("/tmp/rerank_server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            urllib.request.urlopen(UP + "/health", timeout=2)
            break
        except Exception:
            time.sleep(0.5)

    rows = []
    t0 = time.time()
    for i, p in enumerate(probes):
        s = normalize_for_model(p["src"])
        ids = [BOS, VIE] + up("/tokenize", {"content": s})["tokens"] + [EOS]
        cands = [p["greedy"]]
        for k in range(N_SAMPLES):
            out = up("/completion", {"prompt": ids, "n_predict": 120, "temperature": 0.8,
                                     "top_p": 0.95, "seed": p["id"] * 100 + k,
                                     "cache_prompt": False})
            cands.append((out.get("content") or "").strip())
        chrfs = [sacrebleu.sentence_chrf(c, [p["ref"]]).score if c else 0.0 for c in cands]
        rows.append({**p, "cands": cands, "chrfs": [round(x, 1) for x in chrfs]})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(probes)} ({time.time()-t0:.0f}s)", flush=True)
    proc.terminate()

    # LaBSE rerank (cosine JA src vs VI candidate — không nhìn ref)
    print("LaBSE rerank...", flush=True)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(6)
    model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
    model.max_seq_length = 128
    for r in rows:
        texts = [r["src"]] + r["cands"]
        emb = model.encode(texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
        sims = emb[1:] @ emb[0]
        r["labse"] = [round(float(x), 4) for x in sims]
        r["pick_rerank"] = int(np.argmax(sims))
        r["pick_oracle"] = int(np.argmax(r["chrfs"]))

    outp = ROOT / "eval" / f"rerank_probe_{LABEL}.jsonl"
    outp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    n = len(rows)
    g = sum(r["chrfs"][0] for r in rows) / n
    rr = sum(r["chrfs"][r["pick_rerank"]] for r in rows) / n
    oc = sum(r["chrfs"][r["pick_oracle"]] for r in rows) / n
    moved = sum(1 for r in rows if r["pick_rerank"] != 0)
    print(f"=== rerank probe [{LABEL}] n={n}, {N_SAMPLES} sample ===", flush=True)
    print(f"  chrF greedy : {g:.2f}", flush=True)
    print(f"  chrF rerank : {rr:.2f}  (LaBSE chọn khác greedy ở {moved}/{n} câu)", flush=True)
    print(f"  chrF oracle : {oc:.2f}  (trần lý thuyết)", flush=True)
    print(f"done -> {outp}", flush=True)


if __name__ == "__main__":
    main()
