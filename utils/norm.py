# utils/norm.py
import torch

class NormStats:
    """
    [internal]
      - [internal]
      - [internal]
      - [internal]

    [internal]
      [internal]
    """
    def __init__(
        self, S:int, N:int, mode:str="fixed",
        scale_coord_mm:float=200.0, scale_diam_mm:float=20.0, scale_len_mm:float=200.0,
        eps:float=1e-8
    ):
        self.S = int(S); self.N = int(N)
        self.mode = str(mode)
        self.sc_coord = float(scale_coord_mm)
        self.sc_diam  = float(scale_diam_mm)
        self.sc_len   = float(scale_len_mm)
        self.eps = float(eps)

    # [internal note]
    def _apply(self, x: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        S, N = self.S, self.N
        head = S * N * 4
        xshape = x.shape
        # [internal note]
        coords_d = x[..., :head].reshape(*xshape[:-1], S, N, 4)
        lens     = x[..., head:head+S].reshape(*xshape[:-1], S)

        if not inverse:
            coords_d[..., :3] = coords_d[..., :3] / (self.sc_coord + self.eps)
            coords_d[...,  3] = coords_d[...,  3] / (self.sc_diam  + self.eps)
            lens              = lens              / (self.sc_len   + self.eps)
        else:
            coords_d[..., :3] = coords_d[..., :3] * self.sc_coord
            coords_d[...,  3] = coords_d[...,  3] * self.sc_diam
            lens              = lens              * self.sc_len

        out_head = coords_d.reshape(*xshape[:-1], head)
        out_tail = lens.reshape(*xshape[:-1], S)
        return torch.cat([out_head, out_tail], dim=-1)

    def apply_t(self, x: torch.Tensor) -> torch.Tensor:
        """[see source]"""
        return self._apply(x, inverse=False)

    def apply_inv(self, x: torch.Tensor) -> torch.Tensor:
        """[see source]"""
        return self._apply(x, inverse=True)

    # [internal note]
    @staticmethod
    def _extract_scales(obj) -> tuple[float, float, float, float]:
        """
        [internal]
        """
        if isinstance(obj, NormStats):
            return obj.sc_coord, obj.sc_diam, obj.sc_len, obj.eps
        # [internal note]
        def _get(o, *keys, default=None):
            for k in keys:
                if hasattr(o, k): return getattr(o, k)
                if isinstance(o, dict) and k in o: return o[k]
            return default
        sc_coord = float(_get(obj, "scale_coord_mm", "sc_coord", default=200.0))
        sc_diam  = float(_get(obj, "scale_diam_mm",  "sc_diam",  default=20.0))
        sc_len   = float(_get(obj, "scale_len_mm",   "sc_len",   default=200.0))
        eps      = float(_get(obj, "eps", default=1e-8))
        return sc_coord, sc_diam, sc_len, eps

    @staticmethod
    def apply_t_static(x: torch.Tensor, S:int, N:int, stats_or_dict) -> torch.Tensor:
        """[see source]"""
        S = int(S); N = int(N); head = S * N * 4
        sc_coord, sc_diam, sc_len, eps = NormStats._extract_scales(stats_or_dict)
        xshape = x.shape
        coords_d = x[..., :head].reshape(*xshape[:-1], S, N, 4)
        lens    = x[..., head:head+S].reshape(*xshape[:-1], S)
        coords_d[..., :3] = coords_d[..., :3] / (sc_coord + eps)
        coords_d[...,  3] = coords_d[...,  3] / (sc_diam  + eps)
        lens              = lens              / (sc_len   + eps)
        out_head = coords_d.reshape(*xshape[:-1], head)
        out_tail = lens.reshape(*xshape[:-1], S)
        return torch.cat([out_head, out_tail], dim=-1)

    @staticmethod
    def apply_inv_static(x: torch.Tensor, S:int, N:int, stats_or_dict) -> torch.Tensor:
        """[see source]"""
        S = int(S); N = int(N); head = S * N * 4
        sc_coord, sc_diam, sc_len, _ = NormStats._extract_scales(stats_or_dict)
        xshape = x.shape
        coords_d = x[..., :head].reshape(*xshape[:-1], S, N, 4)
        lens    = x[..., head:head+S].reshape(*xshape[:-1], S)
        coords_d[..., :3] = coords_d[..., :3] * sc_coord
        coords_d[...,  3] = coords_d[...,  3] * sc_diam
        lens              = lens              * sc_len
        out_head = coords_d.reshape(*xshape[:-1], head)
        out_tail = lens.reshape(*xshape[:-1], S)
        return torch.cat([out_head, out_tail], dim=-1)
