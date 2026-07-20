"""Train 100M TỪ ĐẦU, CHỈ chiều ja->vi (PLAN_KD_JA2VI §5) trên Modal L40S.

Data: premix ja->vi-only (11,48M seq base+vòng1/2/3a lọc theo tag >>vie<<, đã
kiểm chứng khớp format mix_and_binarize.py) + 778 câu KD (gemini-flash-lite +
qwen-plus, oversample x20) — release `train-assets-ja2vi-100m`
(`scripts/filter_bin_ja2vi.py` + `scripts/pack_kd_into_bin.py`).

Model: d768/12L/12H/ff2048 = ~109.6M params (config gốc CLAUDE.md, đã chạy ổn ở
step4000-23000 trước khi scale 292M) — nhưng giờ TỪ ĐẦU trên corpus 1 CHIỀU.

Chuẩn bị 1 lần:
  TK=$(gh auth token); python3 -m modal secret create github-token GH_TOKEN=$TK
  python3 -m modal run cloud/modal_train_100m_ja2vi.py::setup
Train (tự resume nếu gọi lại giữa chừng — step=0 nếu chưa có last.pt):
  python3 -m modal run --detach cloud/modal_train_100m_ja2vi.py::train
  python3 -m modal run cloud/modal_train_100m_ja2vi.py::status

QUAN TRỌNG (bài học Đợt 0 2026-07-20): train.log ghi milestone mỗi 1000 step —
ĐO JUDGE/hardbench ở TỪNG milestone thay vì chạy cố định tới max-steps rồi mới
xem, corpus KD nhỏ có thể "vắt kiệt" sớm hơn dự kiến giống wave0.
"""
import modal

app = modal.App("vija-100m-ja2vi")
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
    .add_local_dir("tokenizer", "/root/bt/tokenizer")
)

vol = modal.Volume.from_name("vija-100m-ja2vi-vol", create_if_missing=True)
gh = modal.Secret.from_name("github-token")

DIMS = "--d-model 768 --n-layers 12 --n-heads 12 --d-ff 2048"
# max-tokens 8192 x grad-accum 16 = 131072 tok/step (model nhỏ hơn 292M nên
# batch lớn hơn vẫn thoải mái VRAM trên L40S 48GB). ~4 epoch trên 348,8M token
# corpus (base ja->vi + KD) -> 348.8M*4/131072 ~= 10.640 step, làm tròn 10.500.
TRAIN_ARGS = ("--max-tokens 8192 --grad-accum 16 --compile "
              "--max-steps 10500 --lr 3e-4 --min-lr 3e-5 --warmup 500 "
              "--save-every 100 --milestone-every 1000 --log-every 10")


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], timeout=3600)
def setup():
    """Tải premix ja->vi-only (idempotent). KHÔNG cần checkpoint — train từ đầu."""
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
           f"--pattern bin_ja2vi_v1.tar.zst --clobber")
        # LƯU Ý: tar này đóng gói bằng scripts/pack_kd_into_bin.py's caller (chỉ 1
        # cấp "bin/..."), KHÔNG phải "data/bin/..." như bin_mix_vong3a.tar.zst cũ ->
        # KHÔNG dùng --strip-components=1 (sẽ bóc nhầm "bin/" làm file lạc chỗ ở
        # /persist/ thay vì /persist/bin/ — đã xảy ra thật, xem lịch sử commit).
        sh("tar -C /persist -I zstd -xf /tmp/bin_ja2vi_v1.tar.zst "
           "&& rm /tmp/bin_ja2vi_v1.tar.zst")
        print("data OK:", os.listdir("/persist/bin"), flush=True)
    else:
        print("data đã có, bỏ qua.", flush=True)

    vol.commit()
    print("SETUP XONG — Volume sẵn sàng cho ::train.", flush=True)


@app.function(image=image, volumes={"/persist": vol}, secrets=[gh], gpu="L40S",
              cpu=8.0, memory=16384, timeout=8 * 3600)
def train():
    """Train 100M từ đầu (step=0 nếu chưa có last.pt, tự resume nếu gọi lại)."""
    import os
    import subprocess
    import threading
    import time

    assert os.path.exists("/persist/bin/train.tokens.u16"), \
        "Volume không có data — chạy ::setup trước."
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
    print(">> TRAIN 100M ja->vi:", cmd, flush=True)
    proc = subprocess.Popen(cmd, shell=True, env=env)

    def backup_release(tag):
        if not os.path.exists("/persist/checkpoints/last.pt"):
            return False
        try:
            r = subprocess.run(
                "cd /persist/checkpoints && rm -f m100_a? && "
                "split -b 1900m last.pt m100_ && "
                f'GH_TOKEN={os.environ["GH_TOKEN"]} gh release upload autosave-100m-ja2vi '
                f"-R {REPO} --clobber m100_aa m100_ab",
                shell=True, timeout=900, capture_output=True, text=True)
            subprocess.run("rm -f /persist/checkpoints/m100_a?", shell=True)
            if r.returncode == 0:
                print(f"[{tag}] backup OK", flush=True)
                return True
            # release chưa tồn tại lần đầu -> tạo mới
            if "release not found" in (r.stderr or "").lower():
                r2 = subprocess.run(
                    "cd /persist/checkpoints && "
                    f'GH_TOKEN={os.environ["GH_TOKEN"]} gh release create autosave-100m-ja2vi '
                    f"m100_aa m100_ab -R {REPO} --title '100M ja->vi autosave' "
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

    stop = threading.Event()

    def guardian():
        last_up = 0
        while not stop.wait(300):
            try:
                vol.commit()
                print("[guardian] Volume committed", flush=True)
            except Exception as e:
                print("[guardian] commit lỗi:", e, flush=True)
            if time.time() - last_up > 900:
                last_up = time.time()
                backup_release("guardian")

    threading.Thread(target=guardian, daemon=True).start()
    rc = proc.wait()
    stop.set()
    vol.commit()
    print(f"TRAIN kết thúc rc={rc}. Volume đã commit.", flush=True)

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
    print("==== 100M ja->vi TỪ ĐẦU (0 -> 10500) ====")
    if os.path.exists(log):
        out = subprocess.run(f"grep -E '^step ' {log} | tail -8", shell=True,
                             capture_output=True, text=True).stdout.strip()
        print(out or "(chưa có dòng step — compile warmup ~1-2')")
    else:
        print("(chưa có train.log)")
    ms = [s for s in range(1000, 10500, 1000)
          if os.path.exists(f"/persist/checkpoints/step{s}.pt")]
    print(f"milestone có: {ms}")


@app.local_entrypoint()
def main():
    print("modal run cloud/modal_train_100m_ja2vi.py::setup  (1 lần)  "
          "->  --detach ::train  ->  ::status")
