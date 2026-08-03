import torch.nn as nn
from watchmal.model.resnet import resnet50


class ResNetMultiRing(nn.Module):
    """
    ResNet50 (matches single-ring baseline backbone) with a doubled output
    head for two-ring baseline regression. Outputs 2 rings x 7 channels
    (position[3] + direction[3] + energy[1]) = 14 channels total.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7, **kwargs):
        super().__init__()
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring
        num_output_channels = num_rings * channels_per_ring

        self.backbone = resnet50(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            **kwargs,
        )

    def forward(self, x):
        out = self.backbone(x)
        return out.view(-1, self.num_rings, self.channels_per_ring)
