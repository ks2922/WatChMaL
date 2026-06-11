import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius, global_max_pool
from torch_geometric.nn import MLP


class SetAbstraction(nn.Module):
    """
    PointNet++ set abstraction layer using PyTorch Geometric.
    Samples npoint points, groups neighbours within a radius, applies local PointNet.
    """
    def __init__(self, npoint, r, nsample, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.r = r
        self.nsample = nsample
        self.conv = PointNetConv(
            MLP([in_channels + 3] + mlp_channels),
            add_self_loops=False
        )

    def forward(self, x, pos, batch):
        # sample anchor points
        idx = fps(pos, batch, ratio=self.npoint / pos.shape[0])
        # group neighbours within radius
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.nsample)
        edge_index = torch.stack([col, row], dim=0)
        # apply local PointNet
        x_dst = None if x is None else x[idx]
        x_out = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x_out, pos[idx], batch[idx]


class GlobalSetAbstraction(nn.Module):
    """Global set abstraction — pools all remaining points."""
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.conv = PointNetConv(
            MLP([in_channels + 3] + mlp_channels),
            add_self_loops=False
        )

    def forward(self, x, pos, batch):
        # global: each point is its own group, pool over all
        row = torch.arange(pos.shape[0], device=pos.device)
        col = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_index = torch.stack([row, col], dim=0)
        x_out = self.conv((x, x[:1]), (pos, pos[:1]), edge_index)
        x_out = global_max_pool(x, batch)
        return x_out


class PointNet2(nn.Module):
    """
    PointNet++ SSG regression model for WatChMaL using PyTorch Geometric.
    Input:  [B, num_input_channels, N]
    Output: [B, num_output_channels]
    The first 3 input channels are treated as xyz coordinates.
    Remaining channels are treated as point features.
    No compiled CUDA extensions required.
    """
    def __init__(self, num_input_channels, num_output_channels, **kwargs):
        super().__init__()

        in_feat = num_input_channels - 3

        self.sa1 = SetAbstraction(
            npoint=512, r=15.0, nsample=64,
            in_channels=in_feat,
            mlp_channels=[64, 64, 128]
        )
        self.sa2 = SetAbstraction(
            npoint=128, r=60.0, nsample=64,
            in_channels=128,
            mlp_channels=[128, 128, 256]
        )
        self.sa3 = GlobalSetAbstraction(
            in_channels=256,
            mlp_channels=[256, 512, 1024]
        )

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
        B, C, N = x.shape
        xyz = x[:, :3, :].permute(0, 2, 1).reshape(-1, 3)        # [B*N, 3]
        feat = x[:, 3:, :].permute(0, 2, 1).reshape(-1, C - 3)   # [B*N, C-3]
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        if feat.shape[1] == 0:
            feat = None

        feat, xyz, batch = self.sa1(feat, xyz, batch)
        feat, xyz, batch = self.sa2(feat, xyz, batch)
        out = self.sa3(feat, xyz, batch)  # [B, 1024]

        return self.fc(out)
