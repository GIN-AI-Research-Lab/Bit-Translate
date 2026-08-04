# -*- coding: utf-8 -*-
"""Vá các entry zip bị hỏng CRC-32 trong ckpt 30B mà KHÔNG cần tải lại toàn bộ file
xuống đĩa lần 2 (không đủ dung lượng D: cho 2 bản 61GB cùng lúc).

Cách làm: `modal volume get ... -` stream ra STDOUT (không ghi đĩa), đọc theo chunk,
theo dõi offset lũy kế; khi offset chạm vào 1 target (offset,length) đã biết từ
patch_targets.json (tính từ chính file hiện có trên đĩa — cùng nguồn nên offset khớp
byte-for-byte), chụp lại đúng đoạn byte đó, verify CRC32 khớp kỳ vọng rồi vá TRỰC TIẾP
vào file .bin hiện có (seek + write tại đúng offset). Phần còn lại của stream bị vứt bỏ
ngay, không giữ trong RAM/đĩa.
"""
import json
import subprocess
import sys
import time
import zlib

sys.stdout.reconfigure(encoding="utf-8")

CKPT = "D:/Bit-Translate-data/pack_local/pytorch_model.bin"
TARGETS_JSON = "D:/Bit-Translate-data/pack_local/patch_targets.json"
CHUNK = 8 * 1024 * 1024


def main():
    targets = json.load(open(TARGETS_JSON, encoding="utf-8"))
    targets.sort(key=lambda t: t["offset"])
    print(f"can va {len(targets)} entry, tong {sum(t['length'] for t in targets)/1e6:.1f} MB")

    import os
    env = dict(os.environ, MODAL_PROFILE="free30")
    proc = subprocess.Popen(
        ["modal", "volume", "get", "qat-lite-vol",
         "out/expv_Qwen3-30B-A3B_2x4_tq2native.pt", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, bufsize=0,
    )

    pos = 0
    ti = 0  # con tro vao targets (da sort theo offset, doc stream tuan tu nen dung 1 con tro)
    cur_buf = bytearray()
    patched, still_bad = [], []
    t0 = time.time()

    out = open(CKPT, "r+b")
    try:
        first = True
        while ti < len(targets):
            chunk = proc.stdout.read(CHUNK)
            if not chunk:
                break
            if first:
                print("4 byte dau stream (ky vong PK\\x03\\x04):", chunk[:4])
                assert chunk[:4] == b"PK\x03\x04", "stdout KHONG phai raw zip bytes — kiem tra lai lenh modal!"
                first = False
            chunk_start, chunk_end = pos, pos + len(chunk)
            pos = chunk_end
            # gom byte roi vao target hien tai (co the trai qua nhieu chunk lien tiep)
            while ti < len(targets):
                t = targets[ti]
                tgt_start, tgt_end = t["offset"], t["offset"] + t["length"]
                if tgt_start >= chunk_end:
                    break  # target nay con o phia truoc, doi chunk sau
                # phan giao giua [tgt_start,tgt_end) va [chunk_start,chunk_end)
                lo = max(tgt_start, chunk_start)
                hi = min(tgt_end, chunk_end)
                if hi > lo:
                    cur_buf.extend(chunk[lo - chunk_start:hi - chunk_start])
                if len(cur_buf) >= t["length"]:
                    data = bytes(cur_buf[:t["length"]])
                    cur_buf = bytearray(cur_buf[t["length"]:])
                    crc = zlib.crc32(data) & 0xFFFFFFFF
                    if crc == t["expected_crc"]:
                        out.seek(t["offset"])
                        out.write(data)
                        patched.append(t["name"])
                    else:
                        still_bad.append(t["name"])
                    ti += 1
                    print(f"  [{ti}/{len(targets)}] {t['name']} "
                          f"{'VA OK' if crc == t['expected_crc'] else 'VAN HONG (crc lech tiep)'} "
                          f"({time.time()-t0:.0f}s, pos={pos/1e9:.2f}GB)", flush=True)
                    continue
                break  # target chua du du lieu, doi chunk sau
    finally:
        proc.stdout.close()
        proc.terminate()
        out.close()

    print(f"\nXONG: da va {len(patched)}/{len(targets)}, con hong: {len(still_bad)}")
    if still_bad:
        print("CAC ENTRY VAN HONG SAU LAN VA NAY (can lap lai voi 1 stream khac):")
        for n in still_bad:
            print(" ", n)
    json.dump(still_bad, open("D:/Bit-Translate-data/pack_local/still_bad.json", "w"))


if __name__ == "__main__":
    main()
