import torch
import torch.nn as nn

class SmoothRelativeHuberLoss(nn.Module):
    """
    Relative Huber loss with smooth transition between absolute and relative regimes.
    Below p0: behaves like absolute Huber (denominator clamped at p0)
    Above p0: behaves like relative Huber (denominator = |p_true|)
    Transition is smooth via sqrt(norm^2 + p0^2).
    """
    def __init__(self, delta=1.0, p0=20.0):
        super().__init__()
        self.delta = delta
        self.p0 = p0

    def forward(self, input, target):
        norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        effective_norm = torch.sqrt(norm**2 + self.p0**2)
        relative_error = (input - target) / effective_norm
        abs_error = torch.abs(relative_error)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()
