import torch
import torch.nn as nn

class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-6):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, input, target):
        # vector magnitude of true momentum
        norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)

        # relative error scaled by full vector magnitude (NOT per component)
        relative_error = (input - target) / (norm + self.eps)

        abs_error = torch.abs(relative_error)

        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()
