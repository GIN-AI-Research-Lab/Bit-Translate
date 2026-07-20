"""Đợt 0 (PLAN_KD_JA2VI §2) — train 292M +2500 step LR ÊM trên Modal L40S.

Mục đích: chẩn đoán vụ 5 domain tụt ở step30000 (nhiễu LR-restart vs trần sức chứa)
+ tạo 5 milestone cho checkpoint averaging. KHÔNG data mới — dùng nguyên premix
vòng 3a + last.pt@30000 đã có trên Volume vija-vol (từ Phase 1, không cần setup lại).

LR: warmup 50 -> đỉnh 6e-5 -> cosine về 3e-5 tại 32500 (TB ~4.5e-5, bằng nửa đỉnh
1.2e-4 của Phase 1 — không có cú sốc restart). Milestone mỗi 500 step -> 30500..32500.

Chạy:
    python3 -m modal run --detach cloud/modal_train_wave0.py::train    # ~2h L40S
    python3 -m modal run cloud/modal_train_wave0.py::status            # tiến độ
    python3 -m modal run cloud/modal_train_wave0.py::publish           # avg 5 milestone + upload release wave0-lowlr

Autosave: last.pt lên release autosave-scale300m nhưng với TÊN RIÊNG w0_aa/w0_ab —
KHÔNG đè bộ p_aa/p_ab (step30000 lành, nguồn của BT vòng 3b).
"""
import modal

app = modal.App("vija-wave0")
REPO = "trituenguyen97/Bit-Translate"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "zstd")
    .run_commands(
        "curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz "
        "| tar xz -C /tmp && cp /tmp/gh_2.63.0_linux_amd64/bin/gh /usr/local/bin/gh")
    .pip_install("torch", "numpy<2", "sentencepiece", "sacrebleu")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("scripts", "/root/bt/scripts")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
)

vol = modal.Volume.from_name("vija-vol", create_if_missing=False)
gh = modal.Secret.from_name("github-token")

DIMS = "--d-model 1152 --n-layers 16 --n-heads 18 --d-ff 3072"
# save-every PHẢI là ước của milestone-every: train.py chỉ ghi stepN.pt khi step
# chia hết cho CẢ HAI (checkpoint-save lồng trong save_every). 100|500 OK; 60 thì KHÔNG.
WAVE0 = ("--max-tokens 4096 --grad-accum 32 --compile "
         "--max-steps 32500 --lr-anchor 30000 --lr 6e-5 --min-lr 3e-5 --warmup 50 "
         "--save-every 100 --milestone-every 500 --log-every 5")
MILESTONES = [30500, 31000, 31500, 32000, 32500]
W0_TAG = "wave0-lowlr"   # release chứa avg_wave0.pt + step32500.pt (model-only, ~1.2GB/file)


def _symlink_persist():
    import os
    import subprocess
    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [("/persist/bin", "/root/bt/data/bin"),
                     ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=4 * 3600)
def train():
    """Resume last.pt@30000 -> 32500 @ LR êm. Volume commit 5' + backup w0_* 10'."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/checkpoints/last.pt"), \
        "Volume không có last.pt — Phase 1 chưa chạy trên Volume này?"
    assert os.path.exists("/persist/bin/train.tokens.u16"), "Volume không có data premix"
    _symlink_persist()

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Linux thật, OK

    cmd = f"cd /root/bt && python3 scripts/train.py {DIMS} {WAVE0}"
    print(">> WAVE0:", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        """Upload last.pt (split 2 phần <2GB) với tên RIÊNG w0_aa/w0_ab."""
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            r = subprocess.run(
                "cd /persist/checkpoints && rm -f w0_a? && "
                "split -b 1900m last.pt w0_ && "
                f'GH_TOKEN={os.environ["GH_TOKEN"]} gh release upload autosave-scale300m '
                f"-R {REPO} --clobber w0_aa w0_ab",
                shell=True, timeout=900, capture_output=True, text=True)
            subprocess.run("rm -f /persist/checkpoints/w0_a?", shell=True)
            if r.returncode == 0:
                print(f"[{tag}] backup w0_* OK", flush=True)
                return True
            print(f"[{tag}] backup FAIL: {r.stderr[-300:]}", flush=True)
            return False
        except Exception as e:
            print(f"[{tag}] backup lỗi: {e}", flush=True)
            return False

    stop = threading.Event()

    def guardian():
        last_up = 0
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:
                print("[guardian] commit lỗi:", e, flush=True)
            if time.time() - last_up > 600:
                last_up = time.time()
                backup_release("guardian")

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"WAVE0 train kết thúc rc={rc}. Volume đã commit.", flush=True)

    ok = False
    for attempt in range(1, 4):
        ok = backup_release("final")
        if ok:
            break
        print(f"[final] lần {attempt}/3 fail, chờ 30s...", flush=True)
        time.sleep(30)
    if not ok:
        print("[final] CẢNH BÁO: backup release fail — checkpoint vẫn trong Volume, "
              "gọi lại ::train hoặc ::publish.", flush=True)

    log = "/persist/checkpoints/train.log"
    if os.path.exists(log):
        tail = subprocess.run(["tail", "-3", log], capture_output=True, text=True).stdout
        print("train.log:\n" + tail, flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh],
              cpu=8.0, memory=32768, timeout=3600)
def publish():
    """Avg 5 milestone -> avg_wave0.pt; upload avg + step32500.pt lên release wave0-lowlr.
    CPU-only (không tốn GPU)."""
    import os
    import subprocess

    vol.reload()
    _symlink_persist()
    cks = [f"/persist/checkpoints/step{s}.pt" for s in MILESTONES]
    missing = [c for c in cks if not os.path.exists(c)]
    assert not missing, f"thiếu milestone: {missing} — train chưa xong?"

    avg = "/persist/checkpoints/avg_wave0.pt"
    subprocess.run(
        f"cd /root/bt && python3 scripts/avg_checkpoints.py --out {avg} " + " ".join(cks),
        shell=True, check=True)
    vol.commit()

    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
    exists = subprocess.run(f"{env} gh release view {W0_TAG} -R {REPO}",
                            shell=True, capture_output=True).returncode == 0
    files = f"{avg} /persist/checkpoints/step32500.pt"
    verb = (f"gh release upload {W0_TAG} {files} -R {REPO} --clobber") if exists else (
           f"gh release create {W0_TAG} {files} -R {REPO} "
           f"--title 'Wave0 292M lowLR 30000->32500 + avg' --notes 'PLAN_KD_JA2VI dot 0'")
    subprocess.run(f"{env} {verb}", shell=True, check=True, timeout=1800)
    print(f"OK: release {W0_TAG} có avg_wave0.pt + step32500.pt", flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status():
    import os
    import subprocess
    vol.reload()
    log = "/persist/checkpoints/train.log"
    print("==== WAVE0 (30000 -> 32500) ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -6", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — compile warmup ~2-3')")
    ms = [s for s in MILESTONES if os.path.exists(f"/persist/checkpoints/step{s}.pt")]
    print(f"milestone có: {ms} (đủ 5 -> chạy ::publish)")
    print("(số trễ tối đa ~5' do Volume commit định kỳ)")


@app.local_entrypoint()
def main():
    print("modal run --detach cloud/modal_train_wave0.py::train  ->  ::status  ->  ::publish")
