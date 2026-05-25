import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2(nn.Module):
    """
    PointNet++ SSG regression model for WatChMaL.
    Input:  [B, num_input_channels, N]
    Output: [B, num_output_channels]
    The first 3 input channels are treated as xyz coordinates.
    Remaining channels are treated as point features.
    """
    def __init__(self, num_input_channels, num_output_channels, **kwargs):
        super().__init__()

        in_feat = num_input_channels - 3

        self.SA_modules = nn.ModuleList([
            PointnetSAModule(
                npoint=512,
                radius=15.0,    # ~local hit clustering scale
                nsample=64,
                mlp=[in_feat, 64, 64, 128],
                use_xyz=True,
             ),
             PointnetSAModule(
                npoint=128,
                radius=60.0,    # ~ring-scale structure
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            ),
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=True,
            ),
        ])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_output_channels),
        )

    def forward(self, x):
        # x: [B, C, N]
        xyz = x[:, :3, :].transpose(1, 2).contiguous()      # [B, N, 3]
        features = x[:, 3:, :].contiguous()                  # [B, C-3, N]

        if features.shape[1] == 0:
            features = None

        for sa in self.SA_modules:
            xyz, features = sa(xyz, features)

        out = features.squeeze(-1)
        return self.fc(out)
