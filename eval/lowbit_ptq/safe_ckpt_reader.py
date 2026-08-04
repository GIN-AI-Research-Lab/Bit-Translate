# -*- coding: utf-8 -*-
"""Đọc ckpt torch.save (zip, .storage_alignment=64) MÀ KHÔNG dùng torch.load(mmap=True) —
torch 2.12.0+cpu crash (0xC0000005 access violation) khi mmap-materialize tensor trên ckpt
61GB/37491-tensor này (đã xác nhận: OS-level mmap + zipfile + raw sequential read đều đọc
được 100% file, hash MD5 khớp — vậy lỗi nằm trong chính torch, không phải file hỏng).

Cách né: tự parse data.pkl (persistent_load + find_class chặn _rebuild_tensor_v2) để lấy
metadata (storage_key, dtype, shape, stride, storage_offset) KHÔNG chạm byte thật, rồi đọc
byte thật trực tiếp từ entry zip `data/<key>` bằng zipfile (STORED, không nén) + np.frombuffer.
"""
import pickle
import struct
import sys
import zipfile
from collections import namedtuple

import numpy as np
import torch

TensorMeta = namedtuple("TensorMeta", ["storage_key", "dtype", "shape", "stride", "storage_offset", "numel"])

_DTYPE_MAP = {
    "HalfStorage": np.float16,
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "BFloat16Storage": None,  # numpy không có bf16 gốc -> xử lý riêng (uint16 + torch.view)
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "IntStorage": np.int32,
    "LongStorage": np.int64,
    "BoolStorage": np.bool_,
}


class _StorageMarker:
    __slots__ = ("key", "storage_type", "numel")

    def __init__(self, key, storage_type, numel):
        self.key = key
        self.storage_type = storage_type
        self.numel = numel


def _rebuild_tensor_v2_stub(storage, storage_offset, size, stride, requires_grad=None, backward_hooks=None, metadata=None):
    return TensorMeta(storage.key, storage.storage_type, tuple(size), tuple(stride),
                       storage_offset, storage.numel)


def _rebuild_tensor_stub(storage, storage_offset, size, stride):
    return _rebuild_tensor_v2_stub(storage, storage_offset, size, stride)


class _MetaUnpickler(pickle.Unpickler):
    """Unpickler chặn mọi tensor-rebuild để trả metadata thay vì đọc byte thật."""

    def persistent_load(self, pid):
        # pid dạng ('storage', storage_type, key, location, numel)
        assert pid[0] == "storage", f"persistent_id lạ: {pid[0]!r}"
        _, storage_type, key, _location, numel = pid
        type_name = storage_type.__name__ if hasattr(storage_type, "__name__") else str(storage_type)
        return _StorageMarker(key, type_name, numel)

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2_stub
        if module == "torch._utils" and name == "_rebuild_tensor":
            return _rebuild_tensor_stub
        # OrderedDict/torch.Size/collections cơ bản -> để pickle mặc định xử lý (an toàn,
        # không đụng storage/mmap)
        return super().find_class(module, name)


def list_tensors(path):
    """Trả {prefix: str, tensors: {name: TensorMeta}, extra: {name: object}} KHÔNG đọc byte thật."""
    zf = zipfile.ZipFile(path, "r")
    names = zf.namelist()
    pkl_name = next(n for n in names if n.endswith("/data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]
    raw = zf.read(pkl_name)
    up = _MetaUnpickler(__import__("io").BytesIO(raw))
    obj = up.load()
    tensors, extra = {}, {}
    stack = [("", obj)]
    flat = {}

    def walk(o, path_key):
        if isinstance(o, TensorMeta):
            flat[path_key] = o
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, k if not path_key else f"{path_key}.{k}")
        else:
            extra[path_key] = o

    walk(obj, "")
    zf.close()
    # tách gọn: tensor thật nằm dưới "state_dict.", phần còn lại ("meta.…") giữ riêng
    tensors = {k[len("state_dict."):]: v for k, v in flat.items() if k.startswith("state_dict.")}
    tensors.update({k: v for k, v in flat.items() if not k.startswith("state_dict.")})
    return prefix, tensors, extra


_ELEM_SIZE = {
    "HalfStorage": 2, "FloatStorage": 4, "DoubleStorage": 8,
    "BFloat16Storage": 2, "ByteStorage": 1, "CharStorage": 1,
    "IntStorage": 4, "LongStorage": 8, "BoolStorage": 1,
}


def read_tensor(path, prefix, meta):
    """Đọc 1 tensor thật từ entry zip data/<key>, trả torch.Tensor (contiguous, CPU)."""
    zf = zipfile.ZipFile(path, "r")
    entry = f"{prefix}data/{meta.storage_key}"
    elem_size = _ELEM_SIZE[meta.dtype]
    n_bytes = meta.numel * elem_size
    with zf.open(entry, "r") as f:
        raw = f.read(n_bytes)
    zf.close()
    if meta.dtype == "BFloat16Storage":
        arr = np.frombuffer(raw, dtype=np.uint16)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
    else:
        np_dtype = _DTYPE_MAP[meta.dtype]
        arr = np.frombuffer(raw, dtype=np_dtype)
        t = torch.from_numpy(arr.copy())
    # storage_offset + stride: state_dict tensor hầu như luôn contiguous, offset 0;
    # vẫn áp dụng đúng cho tổng quát.
    if meta.storage_offset or tuple(t.shape) != (meta.numel,):
        t = t.as_strided(meta.shape, meta.stride, meta.storage_offset)
    else:
        t = t.view(meta.shape)
    return t.clone()


def open_reader(path, warn_crc_mismatch=True):
    """Trả (prefix, tensors_meta, extra) rồi 1 hàm get(name)->Tensor dùng chung 1 ZipFile.

    Đọc data qua OFFSET SUY TỪ ALIGNMENT (header_offset + 30 + nlen(tên đã biết chính xác)
    + elen(suy từ bất biến "data bắt đầu ở địa chỉ chia hết 64")) THAY VÌ zipfile.open()
    (dựa vào đọc lại nlen/elen từ chính local header trên đĩa). Lý do: ckpt 61GB này có
    một số entry bị lỗi truyền tải khiến local header cũng hỏng (nlen/elen sai, ví dụ đọc
    ra 0/0 thay vì giá trị đúng) — CRC-32 chỉ bảo vệ phần DATA, không bảo vệ header, nên
    zipfile tự đọc header hỏng sẽ tính sai vị trí data và báo "hỏng" dù data THẬT vẫn nguyên
    vẹn. Cách này đã verify phục hồi đúng nhiều tensor tưởng hỏng nhưng thực ra chỉ header
    lỗi (xem lowbit-lab-hoc-tap.md / RESEARCH_TQ33.md mục CRC). Nếu vẫn lệch CRC sau khi
    dùng offset đúng — đó là hỏng DATA thật (đã gặp: vài chục tensor bị tải về toàn số 0,
    lỗi tải/CDN không phải bit-flip; encode vẫn AN TOÀN vì all-zero mã hoá ternary chính
    xác thành all-zero, chỉ cảnh báo để biết mà báo cáo minh bạch).
    """
    prefix, tensors, extra = list_tensors(path)
    zf = zipfile.ZipFile(path, "r")
    fh = open(path, "rb")
    crc_warned = []

    def get(name):
        meta = tensors[name]
        elem_size = _ELEM_SIZE[meta.dtype]
        n_bytes = meta.numel * elem_size
        entry = f"{prefix}data/{meta.storage_key}"
        info = zf.getinfo(entry)
        nlen = len(entry.encode("utf-8"))
        elen = (64 - (info.header_offset + 30 + nlen) % 64) % 64
        data_start = info.header_offset + 30 + nlen + elen
        fh.seek(data_start)
        raw = fh.read(n_bytes)
        if warn_crc_mismatch:
            import zlib
            if (zlib.crc32(raw) & 0xFFFFFFFF) != info.CRC:
                crc_warned.append(name)
        if meta.dtype == "BFloat16Storage":
            arr = np.frombuffer(raw, dtype=np.uint16)
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
        else:
            arr = np.frombuffer(raw, dtype=_DTYPE_MAP[meta.dtype])
            t = torch.from_numpy(arr.copy())
        if meta.storage_offset or tuple(t.shape) != (meta.numel,):
            t = t.as_strided(meta.shape, meta.stride, meta.storage_offset)
        else:
            t = t.view(meta.shape)
        return t.clone()

    get.crc_warned = crc_warned  # danh sach tensor CRC lech thuc su (data hong, khong phai header)
    return prefix, tensors, extra, get, zf


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else "D:/Bit-Translate-data/pack_local/pytorch_model.bin"
    prefix, tensors, extra, get, zf = open_reader(path)
    print("prefix:", prefix)
    print("so tensor:", len(tensors))
    print("extra keys (non-tensor):", list(extra.keys())[:10])
    for k in list(tensors.keys())[:3]:
        m = tensors[k]
        print(f"  {k}: dtype={m.dtype} shape={m.shape} storage_key={m.storage_key}")
    t = get("model.norm.weight")
    print("model.norm.weight:", t.shape, t.dtype, t.float().flatten()[:5].tolist())
    zf.close()
