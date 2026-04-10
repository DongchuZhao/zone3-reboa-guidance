# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqLengthLoss(nn.Module):
    """
    [internal]
      - [internal]
    reduction [internal]
    """
    def __init__(self, S, N, coord_w=1.0, diam_w=1.0, len_w=1.0, use_huber=False, huber_delta=1.0):
        super().__init__()
        self.S, self.N = int(S), int(N)
        self.coord_w, self.diam_w, self.len_w = float(coord_w), float(diam_w), float(len_w)
        self.use_huber = bool(use_huber)
        self.huber_delta = float(huber_delta)

    def _mse_or_huber(self, pred, target):
        if self.use_huber:
            return F.huber_loss(pred, target, delta=self.huber_delta, reduction='mean')
        else:
            return F.mse_loss(pred, target, reduction='mean')

    def forward(self, pred, target):
        B = pred.shape[0]
        K_point = self.S * self.N * 4
        pts_p = pred[:, :K_point]
        pts_t = target[:, :K_point]
        len_p = pred[:, K_point: K_point + self.S]
        len_t = target[:, K_point: K_point + self.S]

        # [internal note]
        pts_p = pts_p.view(B, self.S, self.N, 4)
        pts_t = pts_t.view(B, self.S, self.N, 4)

        coord_p, diam_p = pts_p[..., :3], pts_p[..., 3:]
        coord_t, diam_t = pts_t[..., :3], pts_t[..., 3:]

        loss_coord = self._mse_or_huber(coord_p, coord_t)
        loss_diam  = self._mse_or_huber(diam_p,  diam_t)
        loss_len   = self._mse_or_huber(len_p,   len_t)

        loss = self.coord_w*loss_coord + self.diam_w*loss_diam + self.len_w*loss_len
        stats = {
            "coord": loss_coord.detach(),
            "diam":  loss_diam.detach(),
            "len":   loss_len.detach()
        }
        return loss, stats
