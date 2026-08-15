import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingV2(nn.Module):
    """
    5SA Config 1 backbone + Stage 1 multi-ring improvements: learned per-slot
    embeddings (reinjection), separate per-slot MLP heads, and an auxiliary
    intermediate-layer prediction head supervised with the same matched
    assignment as the main head. Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, dropout=0.3, **kwargs):
        super().__init__()
        in_feat = num_input_channels - 3
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring

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

        self.slot_embeddings = nn.Parameter(torch.randn(num_rings, slot_embed_dim) * 0.02)

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 + slot_embed_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, channels_per_ring),
            ) for _ in range(num_rings)
        ])

        self.aux_pool = nn.AdaptiveMaxPool1d(1)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + slot_embed_dim, 128), nn.ReLU(),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None

        intermediate_feat = None
        for i, sa in enumerate(self.SA_modules):
            xyz, features = sa(xyz, features)
            if i == 3:
                intermediate_feat = self.aux_pool(features).squeeze(-1)

        global_feat = features.squeeze(-1)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(global_feat.shape[0], -1)
            main_outs.append(self.main_heads[r](torch.cat([global_feat, slot_emb], dim=1)))
            aux_outs.append(self.aux_heads[r](torch.cat([intermediate_feat, slot_emb], dim=1)))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)
        return main_out, aux_out
