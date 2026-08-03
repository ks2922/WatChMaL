import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2TDA(nn.Module):
    """
    PointNet++ 5SA Config 1 backbone (matches the single-ring baseline
    exactly) with optional TDA feature fusion, concatenated before the
    regression head. Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_output_channels,
                 dropout=0.3, use_tda=False, tda_dim=0, **kwargs):
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

        self.fusion_head = nn.Linear(256 + self.tda_dim, num_output_channels)

    def forward(self, x, tda_features=None):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None

        for sa in self.SA_modules:
            xyz, features = sa(xyz, features)

        out = features.squeeze(-1)
        out = self.backbone_head(out)

        if self.use_tda:
            if tda_features is None:
                raise ValueError("use_tda=True but tda_features=None")
            out = torch.cat([out, tda_features], dim=1)

        return self.fusion_head(out)
