import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, knn, MLP
from torch_geometric.utils import to_dense_batch


class DetectorSetAbstraction(nn.Module):
    def __init__(self, k, in_channels, mlp_channels):
        super().__init__()
        self.k = k

        self.conv = PointNetConv(
            local_nn=MLP([in_channels + 3] + mlp_channels, norm=None),
            add_self_loops=False
        )

    def forward(self, x, pos, batch):
        # kNN graph instead of radius/FPS
        row, col = knn(pos, pos, self.k, batch, batch)

        edge_index = torch.stack([row, col], dim=0)

        x_dst = x  # no downsampling
        x_out = self.conv((x, x_dst), (pos, pos), edge_index)

        return x_out, pos, batch


class GlobalDetectorPool(nn.Module):
    def __init__(self, in_channels, mlp_channels):
        super().__init__()
        self.mlp = MLP([in_channels, *mlp_channels], norm=None)

    def forward(self, x, pos, batch):
        # preserve detector features explicitly
        x = torch.cat([x, pos], dim=-1)

        x, mask = to_dense_batch(x, batch)
        x = self.mlp(x.permute(0, 2, 1))

        return x.max(dim=-1)[0]


class PointNet2_WatChMaL(nn.Module):
    def __init__(self, num_input_channels, num_outputs):
        super().__init__()

        c = num_input_channels - 3

        # no FPS: keep physical sparsity intact
        self.sa1 = DetectorSetAbstraction(16, c, [64, 64, 128])
        self.sa2 = DetectorSetAbstraction(32, 128, [128, 128, 256])

        self.global_pool = GlobalDetectorPool(256 + 3, [512, 1024])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
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

        x = self.global_pool(x, xyz, batch)

        return self.fc(x)
