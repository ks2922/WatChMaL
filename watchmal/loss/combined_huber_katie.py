import torch
import torch.nn as nn
import torch.nn.functional as F

class WCTECombinedLoss(nn.Module):
    def __init__(self, pos_delta=0.1, mom_delta=1.0, eps=1e-6):
        super().__init__()
        self.pos_delta = pos_delta
        self.mom_delta = mom_delta
        self.eps = eps

    def huber(self, x, y, delta):
        diff = x - y
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic**2 + delta * linear).mean()

    def relative_huber(self, x, y, delta):
        rel = (x - y) / (y + self.eps)
        abs_rel = torch.abs(rel)
        quadratic = torch.clamp(abs_rel, max=delta)
        linear = abs_rel - quadratic
        return (0.5 * quadratic**2 + delta * linear).mean()

    def forward(self, pred, target):
        # split channels
        pos_pred = pred[:, :3]
        pos_true = target[:, :3]

        mom_pred = pred[:, 3:]
        mom_true = target[:, 3:]

        pos_loss = self.huber(pos_pred, pos_true, self.pos_delta)
        mom_loss = self.relative_huber(mom_pred, mom_true, self.mom_delta)

        return pos_loss + mom_loss
