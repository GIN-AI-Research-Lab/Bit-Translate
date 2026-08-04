# -*- coding: utf-8 -*-
"""RESEARCH_TQ33_OUTLIER_FIX.md Giai doan 1 — doc file thong ke nhi phan do boi
qwen3moe_runner_tq33.exe che do "stats" (g_chan_absmax_cur[48][2048], g_chan_absmax_h[48][768],
tich luy MAX(|gia tri|) qua 470 token THAT cua 1 prompt tieng Viet, tung KENH rieng biet).

In ra: layer nao co outlier ro ret nhat, chi so kenh cu the, ty le max-kenh/median-cac-kenh-
khac TRONG CUNG 1 nhom-64 chua no (dung tieu chi nay vi day CHINH LA nhom ma quantize_x_int8()
dung de tinh scale — ty le nay do TRUC TIEP muc do "keo meo scale" ma outlier gay ra). Kiem tra
gia thuyet "kenh outlier la CO DINH xuyen nhieu layer" (van lieu massive-activations noi vay,
nhung PHAI do, khong duoc gia su) bang cach dem so lan moi chi so kenh xuat hien la "top-1" o
tung layer.

KHONG doan truoc chi so kenh cho 30B (hidden=2048) tu bao cao 0.6B (hidden=1024, dim~48/52) —
day la muc dich chinh cua script nay."""
import struct
from collections import Counter

import numpy as np

STATS_PATH = "D:/Bit-Translate-data/tq33_30b/genqa/outlier_stats.bin"


def group_ratio(absmax_row, ch, group_size=64):
    """Ty le max-kenh / median-cac-kenh-KHAC trong CUNG 1 nhom-group_size chua kenh `ch`."""
    b = ch // group_size
    group = absmax_row[b * group_size:(b + 1) * group_size].copy()
    val = group[ch % group_size]
    others = np.delete(group, ch % group_size)
    med = np.median(others)
    return val, med, (val / med if med > 1e-12 else float("inf"))


def analyze(name, arr, group_size, top_k_report=5):
    n_layer, n_dim = arr.shape
    n_block = n_dim // group_size
    print(f"\n{'=' * 90}\n=== {name}  shape=({n_layer},{n_dim})  group_size={group_size} "
          f"({n_block} nhom/layer) ===\n{'=' * 90}")

    top1_channel_per_layer = []
    layer_reports = []
    for l in range(n_layer):
        row = arr[l]
        top_idx = np.argsort(-row)[:top_k_report]
        top1_channel_per_layer.append(int(top_idx[0]))
        ratios = []
        for ch in top_idx:
            val, med, ratio = group_ratio(row, ch, group_size)
            ratios.append((int(ch), float(val), float(med), float(ratio)))
        layer_reports.append(ratios)

    # in bang cho vai layer dai dien (dau/giua/cuoi, giong style bao cao da co) + layer co
    # ty le lon nhat toan bo
    all_max_ratio = [(l, layer_reports[l][0][3]) for l in range(n_layer)]
    all_max_ratio.sort(key=lambda t: -t[1])
    print(f"\n--- Top 10 layer co ty le outlier/median LON NHAT (kenh top-1 cua layer do) ---")
    print(f"{'layer':>5} {'kenh':>6} {'absmax':>10} {'median-nhom':>12} {'ty le':>10}")
    for l, _ in all_max_ratio[:10]:
        ch, val, med, ratio = layer_reports[l][0]
        print(f"{l:5d} {ch:6d} {val:10.4f} {med:12.4f} {ratio:10.1f}x")

    print(f"\n--- Layer 0, {n_layer//2} (giua), {n_layer-1} (cuoi) — top-{top_k_report} kenh moi layer ---")
    for l in [0, 1, n_layer // 2, n_layer - 2, n_layer - 1]:
        print(f"layer {l}:")
        for ch, val, med, ratio in layer_reports[l]:
            print(f"    kenh {ch:5d}  absmax={val:9.4f}  median-nhom={med:8.4f}  ty le={ratio:8.1f}x")

    # gia thuyet "kenh CO DINH xuyen layer" — dem tan suat cua TUNG chi so kenh la top-1
    cnt = Counter(top1_channel_per_layer)
    print(f"\n--- Kenh nao la 'top-1' (absmax lon nhat layer) o BAO NHIEU / {n_layer} layer? ---")
    for ch, c in cnt.most_common(10):
        print(f"    kenh {ch:5d}: top-1 o {c:3d}/{n_layer} layer "
              f"({100.0*c/n_layer:5.1f}%)  -> block-64 chua no = {ch // 64}")

    # cung kiem tra top-K (K=5) moi layer xem co lap lai tap con nho khong (khong chi top-1)
    all_topk_channels = Counter()
    for l in range(n_layer):
        for ch, *_ in layer_reports[l]:
            all_topk_channels[ch] += 1
    print(f"\n--- Kenh xuat hien trong top-{top_k_report} o NHIEU layer nhat (khong chi rieng top-1) ---")
    for ch, c in all_topk_channels.most_common(10):
        print(f"    kenh {ch:5d}: xuat hien trong top-{top_k_report} o {c:3d}/{n_layer} layer")

    return top1_channel_per_layer, all_topk_channels, layer_reports


def main():
    with open(STATS_PATH, "rb") as f:
        hdr = np.frombuffer(f.read(16), dtype=np.int32)
        n_layer, hidden, moe_ffn, prompt_len = hdr.tolist()
        print(f"header: N_LAYER={n_layer} HIDDEN={hidden} MOE_FFN={moe_ffn} prompt_len={prompt_len}")
        cur_arr = np.frombuffer(f.read(4 * n_layer * hidden), dtype=np.float32).reshape(n_layer, hidden)
        h_arr = np.frombuffer(f.read(4 * n_layer * moe_ffn), dtype=np.float32).reshape(n_layer, moe_ffn)

    top1_cur, topk_cur, rep_cur = analyze("g_chan_absmax_cur (input cho ROUTER + expert GATE/UP, HIDDEN=2048)",
                                            cur_arr, 64)
    top1_h, topk_h, rep_h = analyze("g_chan_absmax_h (input cho DOWN_PROJ, MOE_FFN=768)", h_arr, 64)

    print(f"\n{'#'*90}\n# KET LUAN CHO setup_outlier_protect() TRONG qwen3moe_runner_tq33.c\n{'#'*90}")
    print("Chon K kenh co tan suat xuat hien trong top-5 CAO NHAT VA ty le outlier nhat quan qua")
    print("nhieu layer (khong chi 1 layer don le) — xem bang '--- Kenh xuat hien trong top-... ---' o tren.")
    print(f"\n>>> Goi y PROTECT_CUR (tu topk_cur, sap theo tan suat): "
          f"{[ch for ch, _ in topk_cur.most_common(8)]}")
    print(f">>> Goi y PROTECT_H   (tu topk_h,   sap theo tan suat): "
          f"{[ch for ch, _ in topk_h.most_common(8)]}")


if __name__ == "__main__":
    main()
