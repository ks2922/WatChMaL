import torch
import torch.nn as nn
from watchmal.model.resnet import resnet50


class ResNetMultiRingV2Style(nn.Module):
    """
    ResNet50 backbone (unchanged from ResNetMultiRing), but replacing the
    single shared doubled head with: a learned per-ring slot embedding
    (concatenated into each ring's head input) and separate, independently-
    weighted per-ring MLP heads -- mirrors PointNet++'s V2 changes, kept
    isolated from cross-ring attention (tested separately) so any effect
    can be attributed specifically to this change.

    Same num_output_channels absorption pattern as ResNetMultiRing, for
    the same wcte_regression.yaml base-config collision reason.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, num_output_channels=None, **kwargs):
        super().__init__()
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring

        # backbone with a throwaway fc -- we only use its pooled feature,
        # accessed via feature-extraction below, not backbone.fc's output
        self.backbone = resnet50(
            num_input_channels=num_input_channels,
            num_output_channels=1,  # unused, discarded below
            **kwargs,
        )
        pooled_dim = self.backbone.fc.in_features  # 2048 for resnet50/Bottleneck

        self.slot_embeddings = nn.Parameter(torch.randn(num_rings, slot_embed_dim) * 0.02)

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pooled_dim + slot_embed_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, channels_per_ring),
            ) for _ in range(num_rings)
        ])

    def _extract_pooled_feature(self, x):
        b = self.backbone
        out = b.conv1(x)
        out = b.bn1(out)
        out = b.relu(out)
        out = b.maxpool(out)
        out = b.layer1(out)
        out = b.layer2(out)
        out = b.layer3(out)
        out = b.layer4(out)
        out = b.avgpool(out)
        out = torch.flatten(out, 1)
        return out  # (B, pooled_dim), NOT passed through backbone.fc

    def forward(self, x):
        pooled = self._extract_pooled_feature(x)

        outs = []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(pooled.shape[0], -1)
            head_in = torch.cat([pooled, slot_emb], dim=1)
            outs.append(self.main_heads[r](head_in))

        return torch.stack(outs, dim=1)  # (B, num_rings, channels_per_ring)
