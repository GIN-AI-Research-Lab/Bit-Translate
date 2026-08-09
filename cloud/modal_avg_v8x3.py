"""Trung bình 4 mốc v8x3 (14000/15000/16000/16750) NGAY TRÊN VOLUME
-> /persist/checkpoints_v8x3/v8x3_avg.pt. Cùng công thức scripts/avg_checkpoints.py
(cộng fp64 chia đều, giữ cfg+step bản cuối).

Chạy:  MODAL_PROFILE=thaovyh2t modal run cloud/modal_avg_v8x3.py::avg_x3
"""
import modal

app = modal.App("vija-v8x3-avg")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "numpy<2")
vol = modal.Volume.from_name("vija-v8-vol")

STEPS = (14000, 15000, 16000, 16750)


@app.function(image=image, volumes={"/persist": vol}, cpu=4.0, memory=16384,
              timeout=1800)
def avg_x3():
    import torch

    acc, meta, first_nonfloat, keys = None, None, None, None
    for i, s in enumerate(STEPS):
        p = f"/persist/checkpoints_v8x3/step{s}.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck["model"]
        meta = {"step": ck.get("step", 0), "cfg": ck.get("cfg")}
        print(f"[{i+1}/{len(STEPS)}] step{s} (step ghi {meta['step']})", flush=True)
        if acc is None:
            keys = set(sd)
            first_nonfloat = {k: v for k, v in sd.items() if not v.is_floating_point()}
            acc = {k: v.to(torch.float64) for k, v in sd.items() if v.is_floating_point()}
        else:
            assert set(sd) == keys, f"step{s}: state_dict lệch key"
            for k in acc:
                acc[k] += sd[k].to(torch.float64)
        del ck, sd
    out = {k: (v / len(STEPS)).to(torch.float32) for k, v in acc.items()}
    out.update(first_nonfloat)
    torch.save({"model": out, "step": meta["step"], "cfg": meta["cfg"]},
               "/persist/checkpoints_v8x3/v8x3_avg.pt")
    vol.commit()
    print(f"OK: v8x3_avg = avg {len(STEPS)} mốc -> checkpoints_v8x3/v8x3_avg.pt")
