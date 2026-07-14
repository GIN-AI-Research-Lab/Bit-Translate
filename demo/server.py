#!/usr/bin/env python3
"""Sidecar cho demo re-translate: serve UI + proxy dịch sang llama-server + đo tài nguyên.

Chạy sau khi llama-server đã lên (bash demo/run_demo.sh lo cả hai).
- /            -> ui.html
- /translate   -> POST {text, dir: "jpn"|"vie", n_predict?} -> gọi /tokenize + /completion upstream
- /stats       -> RSS / CPU% / threads của tiến trình llama-server (đọc /proc)

Prompt build bằng TOKEN IDS (không nhờ server parse special token) — mirror chính xác
PyTorch: [BOS=2, tag, <ids của text>, EOS=3]. Tag id lấy từ tokenizer SPM của dự án:
>>vie<< = 4, >>jpn<< = 5 (đã verify khớp sp.piece_to_id).
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UPSTREAM = "http://127.0.0.1:8080"
BOS, EOS = 2, 3
TAG = {"jpn": 5, "vie": 4}
HERE = Path(__file__).parent


def upstream_json(path, payload=None, timeout=60):
    req = urllib.request.Request(
        UPSTREAM + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


_pid = None
_last_cpu = (0.0, 0.0)  # (giây CPU tiến trình, wall time)


def server_pid():
    global _pid
    if _pid and Path(f"/proc/{_pid}").exists():
        return _pid
    out = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True).stdout.split()
    _pid = int(out[0]) if out else None
    return _pid


def stats():
    global _last_cpu
    pid = server_pid()
    if not pid:
        return {"error": "llama-server không chạy"}
    st = Path(f"/proc/{pid}/stat").read_text().split()
    proc_s = (int(st[13]) + int(st[14])) / 100.0  # utime+stime, HZ=100
    now = time.time()
    lp, lw = _last_cpu
    cpu_pct = ((proc_s - lp) / (now - lw) * 100.0) if lw and now > lw else 0.0
    _last_cpu = (proc_s, now)
    status = Path(f"/proc/{pid}/status").read_text()
    rss_mb = int(re.search(r"VmRSS:\s+(\d+)", status).group(1)) / 1024
    threads = int(re.search(r"Threads:\s+(\d+)", status).group(1))
    return {"rss_mb": round(rss_mb, 1), "cpu_pct": round(cpu_pct, 1), "threads": threads, "pid": pid}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # im lặng
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "ui.html").read_bytes(), "text/html")
        elif self.path == "/stats":
            self._send(200, stats())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/translate":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            text = (req.get("text") or "").strip()
            tag = TAG.get(req.get("dir", "jpn"), TAG["jpn"])
            if not text:
                return self._send(200, {"translation": "", "timings": {}, "wall_ms": 0})
            t0 = time.time()
            ids = [BOS, tag] + upstream_json("/tokenize", {"content": text})["tokens"] + [EOS]
            out = upstream_json("/completion", {
                "prompt": ids,
                "n_predict": int(req.get("n_predict", 96)),
                "temperature": 0.0,
                "cache_prompt": True,  # KV prefix reuse: gõ thêm chỉ tốn phần chênh
            })
            self._send(200, {
                "translation": (out.get("content") or "").strip(),
                "timings": out.get("timings", {}),
                "wall_ms": round((time.time() - t0) * 1000, 1),
                "prompt_len": len(ids),
            })
        except Exception as e:  # noqa: BLE001 - demo: trả lỗi về UI
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"UI:  http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
