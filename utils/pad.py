# utils/pad.py
import numpy as np
import torch

def _alloc_cpu(shape, dtype=torch.float32, pin=True):
    t = torch.zeros(shape, dtype=dtype)
    if pin and torch.cuda.is_available():
        t = t.pin_memory()
    return t

def pad_3d_list_cpu(arrs, channels_first=True, pin=True, dtype=torch.float32):
    """
    arrs: list of numpy arrays or torch tensors of shape (C,D,H,W) or (1,D,H,W)
    [internal]
    """
    B = len(arrs)
    C = int(arrs[0].shape[0])
    D = max(int(a.shape[1]) for a in arrs)
    H = max(int(a.shape[2]) for a in arrs)
    W = max(int(a.shape[3]) for a in arrs)
    out = _alloc_cpu((B, C, D, H, W), dtype=dtype, pin=pin)

    for i, a in enumerate(arrs):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        d,h,w = a.shape[1], a.shape[2], a.shape[3]
        out[i, :, :d, :h, :w].copy_(a.to(dtype))
    return out  # CPU pinned

def pad_2d_list_cpu(arrs, pin=True, dtype=torch.float32):
    """
    arrs: list of numpy arrays or torch tensors of shape (C,H,W)
    [internal]
    """
    B = len(arrs)
    C = int(arrs[0].shape[0])
    H = max(int(a.shape[1]) for a in arrs)
    W = max(int(a.shape[2]) for a in arrs)
    out = _alloc_cpu((B, C, H, W), dtype=dtype, pin=pin)
    for i, a in enumerate(arrs):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        h,w = a.shape[1], a.shape[2]
        out[i, :, :h, :w].copy_(a.to(dtype))
    return out  # CPU pinned
