import torch.nn as nn
from watchmal.model.resnet import resnet50


class ResNetMultiRing(nn.Module):
    """
    ResNet50 (matches single-ring baseline backbone) with a doubled output
    head for two-ring baseline regression. Outputs 2 rings x 7 channels
    (position[3] + direction[3] + energy[1]) = 14 channels total.

    Accepts (and ignores) num_output_channels, since the shared
    wcte_regression.yaml base config injects model.num_output_channels=7
    via Hydra's _self_ merge order -- this collides with resnet50()'s own
    num_output_channels kwarg unless absorbed here first. The real value
    (num_rings * channels_per_ring) is always computed internally.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 num_output_channels=None, **kwargs):
        super().__init__()
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring

        self.backbone = resnet50(
            num_input_channels=num_input_channels,
            num_output_channels=num_rings * channels_per_ring,
            **kwargs,
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.view(-1, self.num_rings, self.channels_per_ring)
