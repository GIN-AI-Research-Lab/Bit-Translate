"""Train 100M v3 — task >>fix<< + TOÀN BỘ KD đã QC xong, TIẾP TỤC từ checkpoint
v1 (step10500, vocab đã mở 32001), KHÔNG train lại từ đầu (PLAN_KD_JA2VI §5).

Data: premix v3 = base ja->vi thuần (11,48M seq) + 19.584 câu KD sạch (778 v1 +
18.806 qwen-max/turbo/flash qua đủ rule filter + Haiku review + LaBSE, oversample
x3) + 2.759 cặp >>fix<< (1.786 v2 + 973 lỗi thật mới tìm, oversample x8) —
release `train-assets-ja2vi-100m` (`bin_ja2vi_v3.tar.zst`, 11.556.818 seq,
352,3M token).
Checkpoint nền: `last_100m_v2seed.pt` — GIỐNG v2 (v1 step10500 đã mở vocab
32000->32001, không kèm optimizer state cũ — train.py tự khởi động optimizer
mới). Volume RIÊNG (vija-100m-v3-vol) để tránh lẫn state cũ của v2.
Tokenizer: `tokenizer/spm_vija_32k.model` đã có `>>fix<<` (id 32000).

LR êm, KHÔNG restart sốc: +2500 step (10500->13000), lr đỉnh 1e-4, lr-anchor
10500 — data mới thật (>>fix<< + 18.806 KD chưa từng thấy) nên rủi ro overfit
thấp. Đo judge ở milestone (mỗi 500 step), không chạy mù tới hết.

Chuẩn bị 1 lần:
  python3 -m modal run cloud/modal_train_100m_v3.py::setup
Train:
  python3 -m modal run --detach cloud/modal_train_100m_v3.py::train
  python3 -m modal run cloud/modal_train_100m_v3.py::status
"""
import modal

app = modal.App("vija-100m-v3")
REPO = "trituenguyen97/Bit-Translate"
PREMIX_TAG = "train-assets-ja2vi-100m"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "zstd")
    .run_commands(
        "curl -fsSL https://github.com/cli/cli/releases/download/v2.63.0/gh_2.63.0_linux_amd64.tar.gz "
        "| tar xz -C /tmp && cp /tmp/gh_2.63.0_linux_amd64/bin/gh /usr/local/bin/gh")
    .pip_install("torch", "numpy<2", "sentencepiece", "sacrebleu")
    .add_local_dir("src", "/root/bt/src")
    .add_local_dir("scripts", "/root/bt/scripts")
    .add_local_dir("tokenizer", "/root/bt/tokenizer")  # PHẢI có >>fix<< (add_fix_token.py đã chạy local)
)

vol = modal.Volume.from_name("vija-100m-v3-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

DIMS = "--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048 --vocab-size 32001"
TRAIN_ARGS = ("--max-tokens 8192 --grad-accum 16 --compile "
              "--max-steps 13000 --lr-anchor 10500 --lr 1e-4 --min-lr 1e-5 --warmup 100 "
              "--save-every 100 --milestone-every 500 --log-every 10")


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], timeout=3600)
def setup():
    """Tải premix v3 + checkpoint seed (idempotent)."""
    import os
    import subprocess

    def sh(c):
        print(">>", c, flush=True)
        subprocess.run(c, shell=True, check=True)

    os.makedirs("/persist/bin", exist_ok=True)
    os.makedirs("/persist/checkpoints", exist_ok=True)
    env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'

    if not os.path.exists("/persist/bin/train.tokens.u16"):
        sh(f"cd /tmp && {env} gh release download {PREMIX_TAG} -R {REPO} "
           f"--pattern bin_ja2vi_v3.tar.zst --clobber")
        sh("tar -C /persist -I zstd -xf /tmp/bin_ja2vi_v3.tar.zst && rm /tmp/bin_ja2vi_v3.tar.zst")
        print("data OK:", os.listdir("/persist/bin"), flush=True)
    else:
        print("data đã có, bỏ qua.", flush=True)

    if not os.path.exists("/persist/checkpoints/last.pt"):
        sh(f"cd /persist/checkpoints && {env} gh release download {PREMIX_TAG} -R {REPO} "
           f"--pattern last_100m_v2seed.pt --clobber && mv last_100m_v2seed.pt last.pt")
        sz = os.path.getsize("/persist/checkpoints/last.pt")
        print(f"checkpoint seed OK: {sz//2**20}MB", flush=True)
    else:
        print("checkpoint đã có, bỏ qua.", flush=True)

    vol.commit()
    print("SETUP XONG — Volume sẵn sàng cho ::train.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=6 * 3600)
def train():
    """Resume checkpoint seed (step10500, vocab đã mở 32001) -> 13000, multi-task."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/checkpoints/last.pt"), "chưa setup! chạy ::setup trước."
    assert os.path.exists("/persist/bin/train.tokens.u16"), "chưa setup! chạy ::setup trước."
    os.makedirs("/root/bt/data", exist_ok=True)
    for src, dst in [("/persist/bin", "/root/bt/data/bin"),
                     ("/persist/checkpoints", "/root/bt/checkpoints")]:
        if os.path.islink(dst) or os.path.exists(dst):
            subprocess.run(["rm", "-rf", dst])
        os.symlink(src, dst)

    env = dict(os.environ)
    env["BITNET_OPT"] = "off"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    cmd = f"cd /root/bt && python3 scripts/train.py {DIMS} {TRAIN_ARGS}"
    print(">> TRAIN 100M v3 (KD lon + fix):", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            subprocess.run("cd /persist/checkpoints && rm -f v3_a? && split -b 1900m last.pt v3_",
                           shell=True, check=True, timeout=600)
            gh_env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
            r = subprocess.run(
                f"cd /persist/checkpoints && {gh_env} gh release upload autosave-100m-v3 "
                "-R " + REPO + " --clobber $(ls v3_a?)",
                shell=True, timeout=900, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[{tag}] backup OK", flush=True)
                return True
            if "release not found" in (r.stderr or "").lower():
                r2 = subprocess.run(
                    f"cd /persist/checkpoints && {gh_env} gh release create autosave-100m-v3 "
                    "$(ls v3_a?) -R " + REPO + " --title '100M v3 (KD lon+fix) autosave' "
                    "--notes 'auto backup'",
                    shell=True, timeout=900, capture_output=True, text=True)
                if r2.returncode == 0:
                    print(f"[{tag}] backup release CREATED", flush=True)
                    return True
                print(f"[{tag}] create fail: {r2.stderr[-300:]}", flush=True)
            print(f"[{tag}] backup FAIL: {r.stderr[-300:]}", flush=True)
            return False
        except Exception as e:
            print(f"[{tag}] backup lỗi: {e}", flush=True)
            return False
        finally:
            subprocess.run("rm -f /persist/checkpoints/v3_a?", shell=True)

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
    print(f"TRAIN v3 kết thúc rc={rc}. Volume đã commit.", flush=True)

    ok = False
    for attempt in range(1, 4):
        ok = backup_release("final")
        if ok:
            break
        print(f"[final] lần {attempt}/3 fail, chờ 30s...", flush=True)
        time.sleep(30)
    if not ok:
        print("[final] CẢNH BÁO: backup fail — checkpoint vẫn trong Volume, "
              "gọi lại ::train để backup lại.", flush=True)

    log = "/persist/checkpoints/train.log"
    if os.path.exists(log):
        tail = subprocess.run(["tail", "-5", log], capture_output=True, text=True).stdout
        print("train.log:\n" + tail, flush=True)


@app.function(image=image, volumes={"/persist": vol}, timeout=120)
def status():
    import os
    import subprocess
    vol.reload()
    log = "/persist/checkpoints/train.log"
    print("==== 100M v3 (KD lon + fix) 10500 -> 13000 ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -8", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — compile warmup ~1-2')")
    else:
        print("(chưa có train.log)")
    ms = [s for s in range(11000, 13000, 500)
          if os.path.exists(f"/persist/checkpoints/step{s}.pt")]
    print(f"milestone có: {ms}")


@app.local_entrypoint()
def main():
    print("modal run cloud/modal_train_100m_v3.py::setup  (1 lần)  "
          "->  --detach ::train  ->  ::status")
