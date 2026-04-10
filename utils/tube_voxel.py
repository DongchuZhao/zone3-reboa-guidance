# utils/tube_voxel.py
import torch
import numpy as np

def _to_t(x):
    return x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))

def _bbox_from_points(a, b, margin):
    pts = torch.cat([a.reshape(-1,3), b.reshape(-1,3)], 0)
    mn, _ = pts.min(0)
    mx, _ = pts.max(0)
    return mn - margin, mx + margin

def _grid(mn, mx, voxel_mm):
    size = torch.clamp(((mx - mn)/voxel_mm).ceil().long()+2, min=8, max=512)  # [internal]
    return size

def _rasterize_tube(coords, diam, mn, voxel_mm, size):
    """
    coords: (S,N,3)  diam: (S,N)  -> [internal]
    [internal]
    """
    S,N,_ = coords.shape
    device = coords.device
    grid = torch.zeros(tuple(size.tolist()), dtype=torch.bool, device=device)
    # [internal note]
    zz, yy, xx = torch.meshgrid(
        torch.arange(size[0], device=device),
        torch.arange(size[1], device=device),
        torch.arange(size[2], device=device),
        indexing='ij'
    )
    pts = torch.stack([xx,yy,zz], dim=-1).float()*voxel_mm + mn  # (Z,Y,X,3) in mm
    for s in range(S):
        r = (diam[s]/2.0).clamp(min=0.5)
        for i in range(N):
            d2 = ((pts - coords[s,i])**2).sum(-1).sqrt()
            grid |= (d2 <= r[i])
    return grid  # bool[Z,Y,X]

def tubes_to_local_masks(pC, pD, gC, gD, voxel_mm=2.0, margin_mm=10.0):
    """
    pC/gC: (B,S,N,3)   pD/gD: (B,S,N)
    [internal]
    """
    pC, pD, gC, gD = _to_t(pC), _to_t(pD), _to_t(gC), _to_t(gD)
    assert pC.shape == gC.shape
    B,S,N,_ = pC.shape
    masks_p, masks_g = [], []
    for b in range(B):
        mn, mx = _bbox_from_points(pC[b], gC[b], margin=torch.tensor([margin_mm,margin_mm,margin_mm], device=pC.device))
        size = _grid(mn, mx, torch.tensor(voxel_mm, device=pC.device))
        mp = _rasterize_tube(pC[b], pD[b], mn, torch.tensor(voxel_mm, device=pC.device), size)
        mg = _rasterize_tube(gC[b], gD[b], mn, torch.tensor(voxel_mm, device=pC.device), size)
        masks_p.append(mp); masks_g.append(mg)
    return masks_p, masks_g

def dice_coeff(masks_a, masks_b, eps=1e-6):
    """
    masks_*: list of bool 3D tensors
    Sørensen–Dice [internal]
    [internal]
    """
    dices = []
    for A,B in zip(masks_a, masks_b):
        inter = (A & B).sum().float()
        s = A.sum().float() + B.sum().float()
        dices.append((2*inter + eps)/(s + eps))
    return torch.stack(dices,0).mean().item()
