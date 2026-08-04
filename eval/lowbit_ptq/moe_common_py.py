# -*- coding: utf-8 -*-
"""Ban PyTorch/batched cua moe_common.h — dung trong validate_30b_layers.py (oracle) de
DUNG Y HET thuat toan dieu phoi da validate o Giai doan A (validate_moe_routing.c/.exe):
softmax TOAN BO n_expert -> chon top-k LON NHAT -> renormalize (KHONG re-softmax). Tach
rieng file nay (khong nhet vao script chinh) de dam bao chi 1 noi dinh nghia thuat toan nay
o phia Python, dung chung boi validate_moe_routing_gen.py (an toan vi genscript da chay xong
va so khop truoc do) va validate_30b_layers.py.
"""
import torch
import torch.nn.functional as F


def moe_softmax(logits):
    """logits: [..., n_expert] -> softmax tren chieu cuoi (KHOP moe_softmax_inplace)."""
    return F.softmax(logits, dim=-1)


def moe_topk_renorm(probs, k, eps=6.103515625e-5):
    """probs: [..., n_expert] (DA la softmax) -> (idx[...,k], weight[...,k]) giam dan theo
    gia tri, weight da renormalize ve tong=1 (co clamp chong chia 0, KHOP dong 2055-2069
    llama-graph.cpp va moe_topk_renorm trong moe_common.h)."""
    topv, topi = torch.topk(probs, k=k, dim=-1)   # mac dinh: LON NHAT, GIAM DAN, sorted=True
    wsum = topv.sum(dim=-1, keepdim=True).clamp(min=eps)
    weight = topv / wsum
    return topi, weight
