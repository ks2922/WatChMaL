import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2TDACrossAttn(nn.Module):
    """
    Cross-attention fusion between geometric and TDA features, instead of
    flat concatenation. The 256-d geometric feature and a projected 256-d
    TDA feature are treated as two tokens; a small multi-head attention
    block lets each attend to the other before the head, rather than the
    head learning one fixed linear combination of a concatenated vector.
    Same 5SA Config 1 backbone as PointNet2TDA.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_output_channels,
                 dropout=0.3, use_tda=False, tda_dim=0, attn_heads=4, **kwargs):
        super().__init__()
        self.use_tda = use_tda
        self.tda_dim = tda_dim if use_tda else 0

        in_feat = num_input_channels - 3

        self.SA_modules = nn.ModuleList([
            PointnetSAModule(npoint=1024, radius=15.0, nsample=32,
                mlp=[in_feat, 32, 32, 64], use_xyz=True),
            PointnetSAModule(npoint=512, radius=30.0, nsample=32,
                mlp=[64, 64, 64, 128], use_xyz=True),
            PointnetSAModule(npoint=256, radius=80.0, nsample=32,
                mlp=[128, 128, 128, 256], use_xyz=True),
            PointnetSAModule(npoint=64, radius=180.0, nsample=32,
                mlp=[256, 256, 256, 512], use_xyz=True),
            PointnetSAModule(mlp=[512, 512, 1024], use_xyz=True),
        ])

        self.backbone_head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
        )

        if use_tda:
            self.tda_proj = nn.Sequential(
                nn.Linear(tda_dim, 256), nn.ReLU(),
            )
            self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=attn_heads, batch_first=True)
            self.cross_norm = nn.LayerNorm(256)
            head_in = 512  # both attended tokens concatenated
        else:
            head_in = 256

        self.fusion_head = nn.Sequential(
            nn.Linear(head_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_output_channels),
        )

    def forward(self, x, tda_features=None):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None

        for sa in self.SA_modules:
            xyz, features = sa(xyz, features)

        geom_feat = features.squeeze(-1)
        geom_feat = self.backbone_head(geom_feat)

        if self.use_tda:
            if tda_features is None:
                raise ValueError("use_tda=True but tda_features=None")
            tda_feat = self.tda_proj(tda_features)

            tokens = torch.stack([geom_feat, tda_feat], dim=1)  # (B, 2, 256)
            attn_out, _ = self.cross_attn(tokens, tokens, tokens)
            tokens = self.cross_norm(tokens + attn_out)

            out = tokens.reshape(tokens.shape[0], -1)  # (B, 512)
        else:
            out = geom_feat

        return self.fusion_head(out)
