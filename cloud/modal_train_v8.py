"""V8 — KD nhắm-lỗi (targeted KD) tiếp từ v7a_avg (18L/152,1M), step 8750 -> 16750.

Khác các vòng trước: KHÔNG train ở account trituenguyen97 nữa (volume vija-100m-kd-vol
theo account đó). Vòng này chạy trên account **thaovyh2t** ($30), volume RIÊNG
**vija-v8-vol** phải tự upload từ Máy A:
  modal volume put vija-v8-vol data/bin/train.tokens.u16 bin/train.tokens.u16
  modal volume put vija-v8-vol data/bin/train.index.npy  bin/train.index.npy
  modal volume put vija-v8-vol data/bin/dev.tokens.u16   bin/dev.tokens.u16
  modal volume put vija-v8-vol data/bin/dev.index.npy    bin/dev.index.npy
  modal volume put vija-v8-vol E:/Bit-Translate-data/checkpoints_v7a/v7a_avg.pt checkpoints_v8/last.pt

Data: bin MIX = base bin_v7g (15,88M seq) + KD V8 ×8 oversample (104,8k seq ja->vi).
Điểm khởi đầu: v7a_avg (averaged milestone, step 8750). LR-RESTART nhẹ: lr 5e-5 đỉnh
(warmup 200) -> cosine tới 5e-6, anchor 8750 (schedule tính từ 8750). Effective batch
8192×16 = 131k tok/step = Y HỆT run nhà (2048×64) -> quỹ đạo tối ưu KHÔNG đổi, chỉ
nhanh ~15× nhờ L40S + compile.

Chạy (detached, sống độc lập CLI):
  modal run --detach cloud/modal_train_v8.py::train_v8
Theo dõi:
  modal run cloud/modal_train_v8.py::status
"""
import modal

app = modal.App("vija-v8")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy<2", "sentencepiece", "sacrebleu")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("scripts", "/root/bt/scripts")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
)

vol = modal.Volume.from_name("vija-v8-vol", create_if_missing=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=12 * 3600)
def train_v8(steps: int = 16750, lr: float = 5e-5, n_layers: int = 18,
             bin_dir: str = "bin", ckpt_dir: str = "checkpoints_v8",
             compile: bool = True):
    """V8 targeted-KD continue-train từ v7a_avg step 8750 tới `steps`."""
    import os
    import subprocess
    import threading

    assert os.path.exists(f"/persist/{bin_dir}/train.tokens.u16"), \
        f"chưa có data {bin_dir}! chạy: modal volume put vija-v8-vol data/bin/... bin/..."
    assert os.path.exists(f"/persist/{ckpt_dir}/last.pt"), \
        f"chưa có {ckpt_dir}/last.pt! upload v7a_avg.pt -> {ckpt_dir}/last.pt"

    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [(f"/persist/{bin_dir}", f"/root/bt/data/{bin_dir}"),
                     (f"/persist/{ckpt_dir}", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    import torch
    ck = torch.load(f"/persist/{ckpt_dir}/last.pt", map_location="cpu")
    got_L = ck.get("cfg", {}).get("n_layers")
    if got_L is not None:
        assert got_L == n_layers, f"checkpoint {got_L}L != --n-layers {n_layers}"
    start_step = int(ck.get("step", 0))
    print(f">> V8 targeted-KD: {n_layers}L, từ '{ck.get('from','?')}' step={start_step} "
          f"-> {steps} | lr {lr} anchor {start_step}", flush=True)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    anchor = int(ck["step"])   # 8750 — LR restart tính từ điểm resume
    args = (f"--d-model 768 --n-layers {n_layers} --n-heads 12 --d-ff 2048 "
            f"--vocab-size 32001 --max-tokens 8192 --grad-accum 16 --max-seq 256 "
            f"{'--compile --fixed-shapes ' if compile else ''}"
            f"--pad-multiple 32 --label-smoothing 0.1 --bin-dir data/{bin_dir} "
            f"--lr {lr} --min-lr {lr/10:.2e} --warmup 200 --lr-anchor {anchor} "
            f"--max-steps {steps} --save-every 250 --milestone-every 1000 --log-every 10 "
            f"--dev-every 500")
    cmd = f"cd /root/bt && python3 scripts/train.py {args}"
    print(f"   {cmd}", flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    stop = threading.Event()

    def guardian():
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[guardian] commit lỗi:", e, flush=True)

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"V8 kết thúc rc={rc}. Checkpoint ở /persist/{ckpt_dir}.", flush=True)
    log = f"/persist/{ckpt_dir}/train.log"
    if os.path.exists(log):
        print(subprocess.run(["tail", "-10", log], capture_output=True, text=True).stdout,
              flush=True)


@app.function(image=image, volumes={"/persist": vol}, gpu="L40S",
              cpu=8.0, memory=16384, timeout=3600)
def eval_1200(ckpt: str = "last.pt", ckpt_dir: str = "checkpoints_v8", tag: str = ""):
    """Dịch 1199 câu bench bằng checkpoint `ckpt`, chrF vs Google (toàn bộ + 64 câu
    lỗi v7a), kèm baseline v7a từ output đã lưu. Ghi bản dịch v8 về /persist/eval/out.
    Đây là hành vi ternary THẬT (BitLinear forward = bitnet.cpp)."""
    import sys
    import os
    import json
    import time
    sys.path.insert(0, "/root/bt/src")
    sys.path.insert(0, "/root/bt/scripts")
    import torch
    import sentencepiece as spm
    import sacrebleu
    from bitnet import BitNetLM, BitNetConfig
    from text_norm import normalize_for_model

    ckpath = f"/persist/{ckpt_dir}/{ckpt}"
    assert os.path.exists(ckpath), f"không thấy {ckpath}"
    ck = torch.load(ckpath, map_location="cpu")
    c = ck["cfg"]
    step = ck.get("step", "?")
    dev = "cuda"
    sp = spm.SentencePieceProcessor(model_file="/root/bt/tokenizer/spm_vija_32k.model")
    bos, eos, vie = sp.bos_id(), sp.eos_id(), sp.piece_to_id(">>vie<<")
    cfg = BitNetConfig(vocab_size=c["vocab_size"], d_model=c["d_model"],
                       n_layers=c["n_layers"], n_heads=c["n_heads"],
                       d_ff=c["d_ff"], max_seq=c["max_seq"])
    m = BitNetLM(cfg).eval().to(dev)
    m.load_state_dict(ck["model"])
    m.freeze_for_inference()

    ja = [l.strip() for l in open("/persist/eval/combined_1200.txt", encoding="utf-8")
          if l.strip()]
    gg = {}
    for l in open("/persist/eval/google_1200.jsonl", encoding="utf-8"):
        o = json.loads(l)
        gg[o["ja"]] = o["gg"]
    v7a = {}
    for l in open("/persist/eval/v7a_1200.jsonl", encoding="utf-8"):
        o = json.loads(l)
        v7a[o["ja"]] = o["vi"]
    err_ja = set()
    for l in open("/persist/eval/v7a_errors.jsonl", encoding="utf-8"):
        err_ja.add(json.loads(l)["ja"])

    def tr(s):
        ids = torch.tensor([[bos, vie] + sp.encode(normalize_for_model(s)) + [eos]],
                           device=dev)
        with torch.inference_mode():
            out = m.generate_cached(ids, max_new_tokens=200, eos_id=eos, rep_penalty=1.0)
        g = out[0, ids.shape[1]:].tolist()
        if eos in g:
            g = g[:g.index(eos)]
        return sp.decode(g)

    t0 = time.time()
    outs = []
    for i, s in enumerate(ja):
        outs.append(tr(s))
        if (i + 1) % 300 == 0:
            print(f"  dịch {i+1}/{len(ja)} ({time.time()-t0:.0f}s)", flush=True)

    # chrF: v8 vs google + v7a baseline vs google, trên CÙNG tập ref
    h8, hv, ref, h8_64, hv_64, ref_64 = [], [], [], [], [], []
    for s, o in zip(ja, outs):
        if s in gg and s in v7a:
            h8.append(o); hv.append(v7a[s]); ref.append(gg[s])
            if s in err_ja:
                h8_64.append(o); hv_64.append(v7a[s]); ref_64.append(gg[s])

    def chrf(h, r):
        return sacrebleu.corpus_chrf(h, [r]).score if h else 0.0

    r = {
        "step": step,
        "v8_chrf_all": round(chrf(h8, ref), 2),
        "v7a_chrf_all": round(chrf(hv, ref), 2),
        "v8_chrf_64": round(chrf(h8_64, ref_64), 2),
        "v7a_chrf_64": round(chrf(hv_64, ref_64), 2),
        "n": len(ref), "n64": len(ref_64),
        "sec": round(time.time() - t0),
    }
    os.makedirs("/persist/eval/out", exist_ok=True)
    outpath = f"/persist/eval/out/v8_step{step}{('_'+tag) if tag else ''}.jsonl"
    with open(outpath, "w", encoding="utf-8") as f:
        for i, (s, o) in enumerate(zip(ja, outs)):
            f.write(json.dumps({"id": i + 1, "ja": s, "vi": o},
                               ensure_ascii=False) + "\n")
    vol.commit()
    r["out"] = outpath
    print(f"EVAL step {step}: chrF_all v8={r['v8_chrf_all']} v7a={r['v7a_chrf_all']} "
          f"(Δ{r['v8_chrf_all']-r['v7a_chrf_all']:+.2f}) | chrF_64 v8={r['v8_chrf_64']} "
          f"v7a={r['v7a_chrf_64']} (Δ{r['v8_chrf_64']-r['v7a_chrf_64']:+.2f}) -> {outpath}",
          flush=True)
    return r


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status(ckpt_dir: str = "checkpoints_v8"):
    import os
    import subprocess
    vol.reload()
    log = f"/persist/{ckpt_dir}/train.log"
    print(f"==== V8 targeted-KD (thaovyh2t) — {ckpt_dir} ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step |^  DEV ' {log} | tail -14", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — đang compile warmup ~15')")
    else:
        print("(chưa có train.log — container đang khởi động / compile)")
    if os.path.exists(f"/persist/{ckpt_dir}"):
        ms = sorted(f for f in os.listdir(f"/persist/{ckpt_dir}") if f.startswith("step"))
        print(f"milestone: {ms[-6:] if ms else '(chưa có)'}")


@app.local_entrypoint()
def main():
    print("Upload : modal volume put vija-v8-vol data/bin/train.tokens.u16 bin/train.tokens.u16 (+ 3 file + ckpt)")
    print("Train  : modal run --detach cloud/modal_train_v8.py::train_v8")
    print("Status : modal run cloud/modal_train_v8.py::status")
