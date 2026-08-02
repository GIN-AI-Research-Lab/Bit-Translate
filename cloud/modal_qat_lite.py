"""QAT-lite e2e (exp_r) trên Modal — GPU L40S, tài khoản tuent1997 (profile trituekstns).

Chạy:
  set MODAL_PROFILE=trituekstns
  python -m modal run --detach cloud/modal_qat_lite.py::train
  python -m modal run cloud/modal_qat_lite.py::status
Tải checkpoint về (khi xong):
  python -m modal volume get qat-lite-vol out/qat_lite_n3.pt D:\\Bit-Translate-data\\qat_lite_n3.pt

Chi phí ước: L40S ~$1.95/h; S1 ~5-8 phút + S2 1500 step ~20-35 phút => ~$1-1.5/run.
"""
import modal

app = modal.App("qat-lite-n3")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==4.57.6", "numpy", "accelerate", "huggingface_hub")
    .add_local_file("eval/lowbit_ptq/exp_r_qat_lite.py", "/root/exp_r_qat_lite.py")
    .add_local_dir("cloud/qat_data", "/root/qat_data")
)

vol = modal.Volume.from_name("qat-lite-vol", create_if_missing=True)


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=3 * 3600)
def train(steps: int = 1500, train_sents: int = 8000, lr: float = 2e-4,
          freeze_scales: int = 0, freeze_norms: int = 0,
          nm_n: int = 2, nm_m: int = 4, batch: int = 8, out_name: str = "qat_lite_n3.pt",
          fast: int = 0, cosine: int = 0, kd_temp: float = 1.0, ce_w: float = 0.0,
          ema: float = 0.0):
    import os
    import subprocess
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = "/vol/hf"
    mdir = snapshot_download("Qwen/Qwen3-0.6B")
    os.makedirs("/vol/out", exist_ok=True)
    os.environ["EXPR_OUT_DIR"] = "/vol/out"
    cmd = [
        "python", "/root/exp_r_qat_lite.py",
        "--device", "cuda",
        "--model-glob", mdir,
        "--train-vi", "/root/qat_data/train_slice.vi",
        "--train-ja", "/root/qat_data/train_slice.ja",
        "--dev-vi", "/root/qat_data/dev.vi",
        "--dev-ja", "/root/qat_data/dev.ja",
        "--train-skip", "0",
        "--steps", str(steps),
        "--train-sents", str(train_sents),
        "--lr", str(lr),
        "--freeze-scales", str(freeze_scales),
        "--freeze-norms", str(freeze_norms),
        "--nm-n", str(nm_n),
        "--nm-m", str(nm_m),
        "--batch", str(batch),
        "--fast", str(fast),
        "--cosine", str(cosine),
        "--kd-temp", str(kd_temp),
        "--ce-w", str(ce_w),
        "--ema", str(ema),
        "--out", f"/vol/out/{out_name}",
    ]
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    vol.commit()
    print(f"exit={r.returncode}")
    return r.returncode


@app.function(image=image, volumes={"/vol": vol}, timeout=120)
def status():
    import io as _io
    import json
    import os
    p = "/vol/out/exp_r_results.json"
    if not os.path.exists(p):
        print("chưa có kết quả")
        return
    with _io.open(p, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), ensure_ascii=False, indent=2))
    ck = "/vol/out/qat_lite_n3.pt"
    if os.path.exists(ck):
        print(f"checkpoint: {ck} ({os.path.getsize(ck)/1024/1024:.0f} MB)")
