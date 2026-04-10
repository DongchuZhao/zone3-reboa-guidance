# utils/losses.py

import torch
import torch.nn as nn

# -------------------------------------------------
# [internal note]
# [internal note]
# -------------------------------------------------
def safe_split_vec(x: torch.Tensor, S: int, N: int):
    """
    [internal]
      coords: (..., S, N, 3)  [internal]
      diam  : (..., S, N)
      lens  : (..., S)
    [internal]
      - S*N*4 + S         : [internal]
      - S*N*4             : [internal]
      - S*N*3 + S         : [internal]
      - S*N*3             : [internal]
    [internal]
    """
    S = int(S); N = int(N)
    last = x.shape[-1]
    exp_full = S * N * 4 + S
    exp_cdz  = S * N * 4
    exp_cL   = S * N * 3 + S
    exp_c    = S * N * 3

    def _coords_to_len(c):  # (..., S, N, 3) → (..., S)
        # [internal note]
        d = c[..., 1:, :] - c[..., :-1, :]
        seg = torch.sqrt(torch.clamp((d * d).sum(dim=-1), min=0.0))  # (..., S, N-1)
        return seg.sum(dim=-1)                                       # (..., S)

    if last == exp_full:
        head = S * N * 4
        cd   = x[..., :head].view(*x.shape[:-1], S, N, 4)
        coords = cd[..., :3]
        diam   = cd[...,  3]
        lens   = x[..., head:head+S].view(*x.shape[:-1], S)
        return coords, diam, lens

    if last == exp_cdz:           # [internal]
        head = S * N * 4
        cd   = x.view(*x.shape[:-1], S, N, 4)
        coords = cd[..., :3]
        diam   = cd[...,  3]
        lens   = _coords_to_len(coords)
        return coords, diam, lens

    if last == exp_cL:            # [internal]
        head = S * N * 3
        c    = x[..., :head].view(*x.shape[:-1], S, N, 3)
        coords = c
        diam   = torch.zeros(*x.shape[:-1], S, N, device=x.device, dtype=x.dtype)
        lens   = x[..., head:head+S].view(*x.shape[:-1], S)
        return coords, diam, lens

    if last == exp_c:             # [internal]
        c = x.view(*x.shape[:-1], S, N, 3)
        coords = c
        diam   = torch.zeros(*x.shape[:-1], S, N, device=x.device, dtype=x.dtype)
        lens   = _coords_to_len(coords)
        return coords, diam, lens

    raise RuntimeError(
        f"[safe_split_vec] Unexpected last_dim={last}. "
        f"Expected one of {{S*N*4+S={exp_full}, S*N*4={exp_cdz}, S*N*3+S={exp_cL}, S*N*3={exp_c}}} "
        f"(S={S}, N={N})."
    )

# -------------------------------------------------
# [internal note]
# -------------------------------------------------
def smoothness_penalty_1st(diam: torch.Tensor) -> torch.Tensor:
    """
    [internal]
    [internal]
    [internal]
    """
    if diam.ndim < 1:
        return torch.tensor(0.0, device=diam.device if torch.is_tensor(diam) else "cpu")
    diff = diam[..., 1:] - diam[..., :-1]
    return (diff * diff).mean()

def smoothness_penalty_2nd(diam: torch.Tensor) -> torch.Tensor:
    """
    [internal]
    [internal]
    [internal]
    """
    if diam.ndim < 1 or diam.shape[-1] < 3:
        return torch.tensor(0.0, device=diam.device if torch.is_tensor(diam) else "cpu")
    d1 = diam[..., 2:] - 2.0 * diam[..., 1:-1] + diam[..., :-2]
    return (d1 * d1).mean()

# -------------------------------------------------
# [internal note]
# [internal note]
# -------------------------------------------------
class UncertaintyWeighter(nn.Module):
    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = bool(enabled)
        # [internal note]
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, losses_tuple):
        """
        losses_tuple: (Lc, Ld, Ll)  [internal]
        [internal]
        """
        Lc, Ld, Ll = losses_tuple
        if not self.enabled:
            total = Lc + Ld + Ll
            return total, (Lc, Ld, Ll, torch.tensor(0.0, device=total.device, dtype=total.dtype))

        # s_i = log σ_i^2
        s = self.log_vars
        weights = torch.exp(-s)           # (3,)
        extra   = s.sum()                 # [internal]
        stacked = torch.stack([Lc, Ld, Ll], dim=0)  # (3,)
        total   = (weights * stacked).sum() + extra
        return total, (Lc, Ld, Ll, extra)

# -------------------------------------------------
# [internal note]
# [internal note]
# [internal note]
# [internal note]
# -------------------------------------------------
class SeqLengthLoss(nn.Module):
    def __init__(
        self, S: int, N: int,
        coord_w: float = 1.0,
        diam_w:  float = 1.0,
        len_w:   float = 1.0,
        use_huber: bool = False,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.S = int(S); self.N = int(N)
        self.coord_w = float(coord_w)
        self.diam_w  = float(diam_w)
        self.len_w   = float(len_w)
        if use_huber:
            # [internal note]
            self.l1 = nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        else:
            self.l1 = None
            self.mse = nn.MSELoss(reduction="mean")

    def _crit(self, a, b):
        if self.l1 is not None:
            return self.l1(a, b)
        return self.mse(a, b)

    def forward(self, pred, target):
        """
        pred/target: (..., S*N*4 + S) [internal]
        [internal]
        """
        S, N = self.S, self.N
        pc, pd, pl = safe_split_vec(pred,   S, N)
        gc, gd, gl = safe_split_vec(target, S, N)

        Lc = self._crit(pc, gc) * self.coord_w
        Ld = self._crit(pd, gd) * self.diam_w
        Ll = self._crit(pl, gl) * self.len_w
        total = Lc + Ld + Ll
        return total, (Lc, Ld, Ll)

# -------------------------------------------------
# [internal note]
# [internal note]
# [internal note]
# [internal note]
# [internal note]
# [internal note]
# -------------------------------------------------
class CompositeLoss(nn.Module):
    def __init__(
        self, S, N,
        coord_w=1.0, diam_w=1.0, len_w=1.0,
        use_huber=False, huber_delta=1.0,
        use_uncertainty=False,
        diam_smooth_w1=0.0, diam_smooth_w2=0.0
    ):
        super().__init__()
        self.core = SeqLengthLoss(
            S=S, N=N,
            coord_w=coord_w, diam_w=diam_w, len_w=len_w,
            use_huber=use_huber, huber_delta=huber_delta
        )
        self.uw   = UncertaintyWeighter(enabled=use_uncertainty)
        self.ds1w = float(diam_smooth_w1)
        self.ds2w = float(diam_smooth_w2)
        self.S = int(S); self.N = int(N)

    def forward(self, pred, target, extra_terms=None):
        """
        extra_terms: [internal]
        [internal]
          total, parts_dict
        parts_dict [internal]
          'Lc','Ld','Ll','unc_reg','smooth1','smooth2','extras_sum'
        """
        total_core, (Lc, Ld, Ll) = self.core(pred, target)
        # [internal note]
        total_u, (Lc_u, Ld_u, Ll_u, unc_reg) = self.uw((Lc, Ld, Ll))

        # [internal note]
        pc, pd, pl = safe_split_vec(pred, self.S, self.N)
        sm1 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        sm2 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.ds1w > 0:
            sm1 = smoothness_penalty_1st(pd) * self.ds1w
        if self.ds2w > 0:
            sm2 = smoothness_penalty_2nd(pd) * self.ds2w

        # [internal note]
        extras_sum = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if extra_terms is not None and len(extra_terms) > 0:
            extras_sum = torch.stack([e for e in extra_terms], dim=0).sum()

        total = total_u + sm1 + sm2 + extras_sum
        parts = {
            "Lc": Lc_u, "Ld": Ld_u, "Ll": Ll_u,
            "unc_reg": unc_reg, "smooth1": sm1, "smooth2": sm2,
            "extras_sum": extras_sum
        }
        return total, parts
