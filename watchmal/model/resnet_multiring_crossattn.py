import torch
import torch.nn as nn
from watchmal.model.resnet import resnet50


class ResNetMultiRingCrossAttn(nn.Module):
    """
    ResNet50 backbone (unchanged), with a two-step head: since ResNet's
    global-average-pooled feature has no natural per-ring split (unlike
    PointNet++'s spatially-clustered ring features), two distinct per-ring
    projections are first created via slot-embedding-conditioned linear
    layers (both start from the SAME shared pooled feature, differentiated
    only by their learned slot embedding), and only then does cross-ring
    self-attention operate between them -- mirroring V8's mechanism, with
    this projection step as a necessary adaptation for an architecture
    without native per-ring features.

    Deliberately does NOT include V2-style separate-head or slot-embedding-
    in-final-head changes beyond what's needed to create the two attention
    tokens, so any effect can be attributed to cross-ring attention
    specifically, tested separately from ResNetMultiRingV2Style.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, proj_dim=256, cross_ring_attn_heads=4,
                 num_output_channels=None, **kwargs):
        super().__init__()
        assert num_rings == 2, "cross-ring attention here only implemented for num_rings=2"
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring

        self.backbone = resnet50(
            num_input_channels=num_input_channels,
            num_output_channels=1,
            **kwargs,
        )
        pooled_dim = self.backbone.fc.in_features

        self.slot_embeddings = nn.Parameter(torch.randn(num_rings, slot_embed_dim) * 0.02)
        self.ring_proj = nn.Sequential(
            nn.Linear(pooled_dim + slot_embed_dim, proj_dim), nn.ReLU(),
        )

        self.cross_ring_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=cross_ring_attn_heads, batch_first=True)
        self.cross_ring_norm = nn.LayerNorm(proj_dim)

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(proj_dim, 256), nn.ReLU(), nn.Dropout(0.3),
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
        return out

    def forward(self, x):
        pooled = self._extract_pooled_feature(x)  # (B, pooled_dim), shared by both rings

        ring_feats = []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(pooled.shape[0], -1)
            ring_feats.append(self.ring_proj(torch.cat([pooled, slot_emb], dim=1)))
        ring_feats = torch.stack(ring_feats, dim=1)  # (B, 2, proj_dim)

        attn_out, _ = self.cross_ring_attn(ring_feats, ring_feats, ring_feats)
        ring_feats = self.cross_ring_norm(ring_feats + attn_out)

        outs = [self.main_heads[r](ring_feats[:, r]) for r in range(self.num_rings)]
        return torch.stack(outs, dim=1)
