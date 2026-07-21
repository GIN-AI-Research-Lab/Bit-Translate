"""Train 100M v4 — TIẾP TỤC từ checkpoint v3 (step13000, vocab 32001, KHÔNG đổi
kiến trúc/vocab lần này nên GIỮ NGUYÊN optimizer state, khác v1->v2 phải reset
vì mở vocab). KHÔNG train lại từ đầu (PLAN_KD_JA2VI §5).

Data: premix v4 = base ja->vi thuần (11,48M seq) + 32.390 câu KD sạch (19.584
đã dùng ở v3 + 12.806 mới — TOÀN BỘ backlog qwen-plus/gemini-lite còn lại, rule
filter + LaBSE thr=0.55, 98.9% đạt vì 2 thầy này ~0% lỗi audit — KHÔNG cần qua
Haiku review như qwen-max/turbo/flash, oversample x3) + 2.721 cặp >>fix<< (y
hệt v3, oversample x8, 38 câu bị bỏ vì > 320 token) — release
`train-assets-ja2vi-100m` (`bin_ja2vi_v4.tar.gz` — GZIP không phải zstd, máy
soạn thiếu zstd CLI cục bộ; setup() dùng tar -xzf), 11.595.236 seq, 354,2M
token.

Checkpoint nền: last.pt CUỐI của v3 (autosave-100m-v3 release, ghép split
v3_a?) — dùng NGUYÊN optimizer state vì kiến trúc + vocab không đổi lần này
(khác các lần trước reset optimizer do mismatch shape).
Volume RIÊNG (vija-100m-v4-vol) để tránh lẫn state cũ của v3.

2 FIX TỐC ĐỘ đã đo thật ở v3 (2026-07-21, L40S 80k->24k tok/s vì
torch._dynamo hit recompile_limit=8 do data KD+fix quá đa dạng độ dài):
  1. --pad-multiple 32: gom (B,T) về bội số 32 -> giảm mạnh số shape khác
     nhau (đã dùng thật cho 292M, xem help --pad-multiple trong train.py).
  2. train.py giờ tự set torch._dynamo.config.recompile_limit=64 trước khi
     compile (lưới an toàn thứ 2, phòng khi vẫn còn dư vài shape ngoài dự
     kiến sau khi bucket).
Kỳ vọng phục hồi phần lớn tốc độ (không chắc đúng 80k vì >>fix<< vẫn có phân
bố độ dài khác dịch thuần), nhưng phải > 24k rõ rệt.

LR êm, KHÔNG restart sốc: +2500 step (13000->15500), lr đỉnh 1e-4, lr-anchor
13000 — data mới thật (12.806 câu KD chưa từng thấy) nên rủi ro overfit thấp,
NHƯNG vẫn đo judge ở từng milestone (mỗi 500 step) như mọi vòng trước, KHÔNG
chạy mù tới hết — nếu judge tụt ở milestone nào thì dừng, dùng milestone đó.

Chuẩn bị 1 lần (SAU KHI v3 chạy xong step13000 + đã backup final release):
  python3 -m modal run cloud/modal_train_100m_v4.py::setup
Train:
  python3 -m modal run --detach cloud/modal_train_100m_v4.py::train
  python3 -m modal run cloud/modal_train_100m_v4.py::status
"""
import modal

app = modal.App("vija-100m-v4")
REPO = "trituenguyen97/Bit-Translate"
PREMIX_TAG = "train-assets-ja2vi-100m"
V3_BACKUP_TAG = "autosave-100m-v3"

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

vol = modal.Volume.from_name("vija-100m-v4-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

DIMS = "--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048 --vocab-size 32001"
TRAIN_ARGS = ("--max-tokens 8192 --grad-accum 16 --compile --max-seq 384 --pad-multiple 32 "
              "--max-steps 15500 --lr-anchor 13000 --lr 1e-4 --min-lr 1e-5 --warmup 100 "
              "--save-every 100 --milestone-every 500 --log-every 10")


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], timeout=3600)
def setup():
    """Tải premix v4 + ghép checkpoint cuối của v3 làm seed (idempotent)."""
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
           f"--pattern bin_ja2vi_v4.tar.gz --clobber")
        sh("tar -C /persist -xzf /tmp/bin_ja2vi_v4.tar.gz && rm /tmp/bin_ja2vi_v4.tar.gz")
        print("data OK:", os.listdir("/persist/bin"), flush=True)
    else:
        print("data đã có, bỏ qua.", flush=True)

    if not os.path.exists("/persist/checkpoints/last.pt"):
        sh(f"cd /tmp && {env} gh release download {V3_BACKUP_TAG} -R {REPO} --pattern 'v3_a*' --clobber")
        sh("cd /tmp && cat v3_a* > /persist/checkpoints/last.pt && rm -f v3_a*")
        sz = os.path.getsize("/persist/checkpoints/last.pt")
        print(f"checkpoint seed (v3 step13000) OK: {sz//2**20}MB", flush=True)
    else:
        print("checkpoint đã có, bỏ qua.", flush=True)

    vol.commit()
    print("SETUP XONG — Volume sẵn sàng cho ::train.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=6 * 3600)
def train():
    """Resume checkpoint v3 (step13000, vocab 32001, GIỮ optimizer state) -> 15500."""
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
    print(">> TRAIN 100M v4 (KD backlog + pad-multiple fix):", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            subprocess.run("cd /persist/checkpoints && rm -f v4_a? && split -b 1900m last.pt v4_",
                           shell=True, check=True, timeout=600)
            gh_env = f'GH_TOKEN={os.environ["GH_TOKEN"]}'
            r = subprocess.run(
                f"cd /persist/checkpoints && {gh_env} gh release upload autosave-100m-v4 "
                "-R " + REPO + " --clobber $(ls v4_a?)",
                shell=True, timeout=900, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"[{tag}] backup OK", flush=True)
                return True
            if "release not found" in (r.stderr or "").lower():
                r2 = subprocess.run(
                    f"cd /persist/checkpoints && {gh_env} gh release create autosave-100m-v4 "
                    "$(ls v4_a?) -R " + REPO + " --title '100M v4 (KD backlog) autosave' "
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
            subprocess.run("rm -f /persist/checkpoints/v4_a?", shell=True)

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
    print(f"TRAIN v4 kết thúc rc={rc}. Volume đã commit.", flush=True)

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
    print("==== 100M v4 (KD backlog + pad-multiple fix) 13000 -> 15500 ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -8", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — compile warmup ~1-2')")
    else:
        print("(chưa có train.log)")
    ms = [s for s in range(13500, 15500, 500)
          if os.path.exists(f"/persist/checkpoints/step{s}.pt")]
    print(f"milestone có: {ms}")


@app.local_entrypoint()
def main():
    print("modal run cloud/modal_train_100m_v4.py::setup  (1 lần, SAU KHI v3 xong)  "
          "->  --detach ::train  ->  ::status")
