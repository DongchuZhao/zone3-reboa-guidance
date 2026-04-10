# utils/metrics.py
import numpy as np
import torch

# ---------------------------
# [internal note]
# ---------------------------

def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_tensor(x, device=None, dtype=None):
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

# ---------------------------
# [internal note]
# ---------------------------

def split_vec(x: torch.Tensor, S: int, N: int):
    """
    [internal]
      [internal]
      [internal]
    [internal]
      coords: (..., S, N, 3)  [internal]
      diam  : (..., S, N)     [internal]
      lens  : (..., S)        [internal]
    """
    S = int(S); N = int(N)
    head = S * N * 4
    xshape = x.shape
    cd = x[..., :head].reshape(*xshape[:-1], S, N, 4)   # (..., S, N, 4)
    coords = cd[..., :3]                                # (..., S, N, 3)
    diam   = cd[...,  3].reshape(*xshape[:-1], S, N)    # (..., S, N)
    lens   = x[..., head:head+S].reshape(*xshape[:-1], S)
    return coords, diam, lens

# ---------------------------
# [internal note]
# ---------------------------

def _rmse(t: torch.Tensor, dim=None, keepdim=False):
    return torch.sqrt(torch.mean(t * t, dim=dim, keepdim=keepdim) + 1e-12)

def mae_rmse_mm(pred_mm: torch.Tensor, gt_mm: torch.Tensor, S: int, N: int):
    """
    [internal]
    """
    p_c, p_d, p_l = split_vec(pred_mm, S, N)
    g_c, g_d, g_l = split_vec(gt_mm,  S, N)

    coord_diff = (p_c - g_c)
    diam_diff  = (p_d - g_d)
    len_diff   = (p_l - g_l)

    coord_mae  = coord_diff.abs().mean()
    # [internal note]
    if coord_diff.ndim >= 2:
        coord_rmse = _rmse(coord_diff.view(coord_diff.shape[0], -1), dim=1).mean()
    else:
        coord_rmse = _rmse(coord_diff).mean()

    diam_mae   = diam_diff.abs().mean()
    if diam_diff.ndim >= 2:
        diam_rmse  = _rmse(diam_diff.view(diam_diff.shape[0], -1), dim=1).mean()
    else:
        diam_rmse  = _rmse(diam_diff).mean()

    len_mae    = len_diff.abs().mean()
    len_rmse   = _rmse(len_diff, dim=-1).mean()

    return {
        "coord_mae_mm":  float(coord_mae),
        "coord_rmse_mm": float(coord_rmse),
        "diam_mae_mm":   float(diam_mae),
        "diam_rmse_mm":  float(diam_rmse),
        "len_mae_mm":    float(len_mae),
        "len_rmse_mm":   float(len_rmse),
    }

# ---------------------------
# [internal note]
# ---------------------------

def chamfer_distance_mm(p_coords: torch.Tensor, g_coords: torch.Tensor, squared: bool=False) -> torch.Tensor:
    """
    [internal]
      CD = mean_a min_b ||a-b|| + mean_b min_a ||b-a||
    [internal]
      (B,S,N,3) [internal]
    [internal]
      [internal]
    [internal]
      [internal]
    """
    # [internal note]
    if p_coords.ndim == 4:
        B, S, N, _ = p_coords.shape
        P = p_coords.reshape(B, S*N, 3)
    else:
        P = p_coords
    if g_coords.ndim == 4:
        B2, S2, N2, _ = g_coords.shape
        G = g_coords.reshape(B2, S2*N2, 3)
    else:
        G = g_coords

    # [internal note]
    P = P.float()
    G = G.float()

    dists = torch.cdist(P, G, p=2)      # (B, Pn, Gn)
    d1 = dists.min(dim=2).values.mean(dim=1)  # (B,)
    d2 = dists.min(dim=1).values.mean(dim=1)  # (B,)
    cd = d1 + d2
    if squared:
        cd = cd * cd
    # [internal note]
    return cd.mean()

# ---------------------------
# [internal note]
# ---------------------------

def _voxel_bbox(coords_np, margin_mm: float):
    lo = np.nanmin(coords_np, axis=0)
    hi = np.nanmax(coords_np, axis=0)
    lo = lo - margin_mm
    hi = hi + margin_mm
    return lo, hi

def _world_to_index(xyz_mm, origin_mm, spacing_mm):
    return np.round((xyz_mm - origin_mm) / spacing_mm).astype(np.int64)

def _rasterize_tube(mask, origin_mm, spacing_mm, pts_mm, diam_mm):
    if pts_mm.shape[0] == 0:
        return
    for p, d in zip(pts_mm, diam_mm):
        r = max(float(d) * 0.5, 0.0)
        if r <= 0:
            continue
        rad = np.array([r, r, r], dtype=np.float32)
        lo_mm = p - rad
        hi_mm = p + rad
        lo_idx = _world_to_index(lo_mm, origin_mm, spacing_mm)
        hi_idx = _world_to_index(hi_mm, origin_mm, spacing_mm)
        zlo, ylo, xlo = np.maximum(lo_idx[::-1], 0)
        zhi, yhi, xhi = np.minimum(hi_idx[::-1], np.array(mask.shape) - 1)
        if zhi < zlo or yhi < ylo or xhi < xlo:
            continue
        zz, yy, xx = np.ogrid[zlo:zhi+1, ylo:yhi+1, xlo:xhi+1]
        cen_idx = _world_to_index(p, origin_mm, spacing_mm)[::-1]
        cz, cy, cx = cen_idx
        dz = (zz - cz) * spacing_mm[2]
        dy = (yy - cy) * spacing_mm[1]
        dx = (xx - cx) * spacing_mm[0]
        dist2 = dx*dx + dy*dy + dz*dz
        inside = dist2 <= (r * r + 1e-6)
        mask[zlo:zhi+1, ylo:yhi+1, xlo:xhi+1] |= inside

def dice_from_local_tubes(
    p_coords, p_diam, g_coords, g_diam,
    voxel_mm: float = 2.0, margin_mm: float = 10.0
):
    """
    [internal]
    [internal]
      p_coords / g_coords : (B,S,N,3)  [internal]
      p_diam   / g_diam   : (B,S,N)    [internal]
    [internal]
      float（[internal]
    """
    PC = _to_numpy(p_coords)
    PD = _to_numpy(p_diam)
    GC = _to_numpy(g_coords)
    GD = _to_numpy(g_diam)

    B, S, N, _ = PC.shape
    spacing = np.array([voxel_mm, voxel_mm, voxel_mm], dtype=np.float32)  # (x,y,z)

    dices = []
    for b in range(B):
        pts_all = np.concatenate([PC[b].reshape(-1, 3), GC[b].reshape(-1, 3)], axis=0)
        lo_mm, hi_mm = _voxel_bbox(pts_all, margin_mm)
        size_xyz = np.maximum(np.ceil((hi_mm - lo_mm) / spacing).astype(np.int64) + 1, 1)
        size_zyx = size_xyz[::-1]
        mask_p = np.zeros(size_zyx, dtype=bool)
        mask_g = np.zeros(size_zyx, dtype=bool)
        for s in range(S):
            _rasterize_tube(mask_p, lo_mm, spacing, PC[b, s], PD[b, s])
            _rasterize_tube(mask_g, lo_mm, spacing, GC[b, s], GD[b, s])
        inter = np.logical_and(mask_p, mask_g).sum(dtype=np.float64)
        vp = mask_p.sum(dtype=np.float64); vg = mask_g.sum(dtype=np.float64)
        dice = (2.0 * inter) / (vp + vg + 1e-6)
        dices.append(dice)
    return float(np.mean(dices)) if len(dices) > 0 else 0.0

# ---------------------------
# [internal note]
# ---------------------------

def postprocess_savgol(diam_np: np.ndarray, window: int = 7, poly: int = 3) -> np.ndarray:
    """
    Savitzky–Golay [internal]
    """
    d = _to_numpy(diam_np).astype(np.float32, copy=False)
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(d, window_length=window, polyorder=min(poly, window-1), axis=-1, mode="interp")
    except Exception:
        k = max(1, window)
        pad = k // 2
        pad_mode = "edge"
        if d.ndim == 2:
            x = np.pad(d, ((0,0),(pad,pad)), mode=pad_mode)
            ker = np.ones((1,k), dtype=np.float32) / k
            y = np.apply_along_axis(lambda v: np.convolve(v, ker.ravel(), mode="valid"), axis=-1, arr=x)
        else:
            x = np.pad(d, ((0,0),)*(d.ndim-1) + ((pad,pad),), mode=pad_mode)
            ker = np.ones((k,), dtype=np.float32) / k
            y = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="valid"), axis=-1, arr=x)
        return y.astype(np.float32)

def _tv1d_denoise(signal: np.ndarray, lam: float, iters: int) -> np.ndarray:
    x = signal.astype(np.float32, copy=True)
    p = np.zeros_like(x)
    for _ in range(max(1, int(iters))):
        dx = np.diff(x, append=x[-1])
        p = np.clip(dx, -lam, lam)
        x = signal - np.concatenate([[0.0], np.cumsum(p[:-1] - p[1:])])
    return x

def postprocess_tv(diam_np: np.ndarray, lam: float = 0.01, iters: int = 60) -> np.ndarray:
    d = _to_numpy(diam_np).astype(np.float32, copy=True)
    if d.ndim == 1:
        return _tv1d_denoise(d, lam, iters)
    elif d.ndim == 2:
        S, _ = d.shape
        out = np.zeros_like(d)
        for s in range(S):
            out[s] = _tv1d_denoise(d[s], lam, iters)
        return out
    else:
        shape = d.shape
        flat = d.reshape(-1, shape[-1])
        for i in range(flat.shape[0]):
            flat[i] = _tv1d_denoise(flat[i], lam, iters)
        return flat.reshape(shape)
