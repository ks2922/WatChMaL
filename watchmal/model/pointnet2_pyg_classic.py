import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, radius, MLP
from torch_geometric.utils import to_dense_batch


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius_r, nsample, in_channels, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.radius_r = radius_r
        self.nsample = nsample

        self.conv = PointNetConv(
            local_nn=MLP([in_channels + 3] + mlp_channels, norm=None),
            add_self_loops=False
        )

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=None, K=self.npoint)

        row, col = radius(
            pos, pos[idx],
            self.radius_r,
            batch, batch[idx],
            max_num_neighbors=self.nsample
        )

        edge_index = torch.stack([row, col], dim=0)

        x_dst = None if x is None else x[idx]
        x_out = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        return x_out, pos[idx], batch[idx]


class GlobalSA(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.mlp = MLP([in_channels, *mlp_channels], norm=None)

    def forward(self, x, pos, batch):
        x = torch.cat([x, pos], dim=-1)
        x, _ = to_dense_batch(x, batch)
        x = self.mlp(x.permute(0, 2, 1))
        return x.max(dim=-1)[0]


class PointNet2_Canonical(nn.Module):
    def __init__(self, num_input_channels, num_outputs):
        super().__init__()
        c = num_input_channels - 3

        self.sa1 = SetAbstraction(512, 15.0, 64, c, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 60.0, 64, 128, [128, 128, 256])
        self.sa3 = GlobalSA(256 + 3, [256, 512, 1024])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        B, C, N = x.shape

        xyz = x[:, :3].permute(0, 2, 1).reshape(-1, 3)
        feat = x[:, 3:].permute(0, 2, 1).reshape(-1, C - 3)

        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        x, xyz, batch = self.sa1(feat, xyz, batch)
        x, xyz, batch = self.sa2(x, xyz, batch)
        x = self.sa3(x, xyz, batch)

        return self.fc(x)
