import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRing(nn.Module):
    """
    PointNet++ 5SA (Config 1 backbone) with a doubled output head for two-ring
    baseline regression. Outputs 2 rings x 7 channels (position[3] + direction[3]
    + energy[1]) = 14 channels total. Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 dropout=0.3, **kwargs):
        super().__init__()
        in_feat = num_input_channels - 3
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring
        num_output_channels = num_rings * channels_per_ring

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

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_output_channels),
        )

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None
        for sa in self.SA_modules:
            xyz, features = sa(xyz, features)
        out = self.fc(features.squeeze(-1))
        return out.view(-1, self.num_rings, self.channels_per_ring)
