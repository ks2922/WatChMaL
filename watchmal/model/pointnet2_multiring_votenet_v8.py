import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNetV8(nn.Module):
    """
    Same as PointNet2MultiRingVoteNetV3, except the two per-ring cluster
    features exchange information with each other (2-token self-attention,
    i.e. cluster0 attends to cluster1 and vice versa) before being passed
    to main_heads.

    Motivation: unlike V5's self-attention over all 64 seed points (which
    caused broad degradation, plausibly by diluting per-point spatial
    precision with irrelevant long-range context), this operates on only
    the two already-pooled cluster summaries - a much smaller, cheaper
    attention problem that cannot smear fine per-point detail, since
    pooling has already happened. This directly tests whether letting each
    ring's summary "see" the other ring's summary (without touching raw
    per-point features) recovers some of the direction-resolution loss
    seen under clustering, while keeping clustering's position/momentum
    gains untouched.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, max_vote_offset=200.0, dropout=0.3,
                 cluster_temperature=20.0, cluster_penalty_scale=1e3,
                 cross_ring_attn_heads=4, **kwargs):
        super().__init__()
        assert num_rings == 2, "fixed-K anchor clustering below only implemented for num_rings=2"
        in_feat = num_input_channels - 3
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring
        self.max_vote_offset = max_vote_offset
        self.cluster_temperature = cluster_temperature
        self.cluster_penalty_scale = cluster_penalty_scale

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

        self.vote_offset_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.vote_feat_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
        )

        # NEW: 2-token cross-ring attention over [cluster0, cluster1]
        self.cross_ring_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=cross_ring_attn_heads, batch_first=True)
        self.cross_ring_norm = nn.LayerNorm(512)

        self.slot_embeddings = nn.Parameter(torch.randn(num_rings, slot_embed_dim) * 0.02)

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + slot_embed_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024 + slot_embed_dim, 128), nn.ReLU(),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

    def _cluster_votes(self, vote_xyz, vote_feat):
        B, N, _ = vote_xyz.shape
        centroid = vote_xyz.mean(dim=1, keepdim=True)
        d0 = torch.linalg.norm(vote_xyz - centroid, dim=-1)
        anchor0_idx = d0.argmax(dim=1)
        anchor0 = vote_xyz[torch.arange(B), anchor0_idx].unsqueeze(1)
        d1 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)
        anchor1_idx = d1.argmax(dim=1)
        anchor1 = vote_xyz[torch.arange(B), anchor1_idx].unsqueeze(1)

        dist_to_0 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)
        dist_to_1 = torch.linalg.norm(vote_xyz - anchor1, dim=-1)

        logits = torch.stack([-dist_to_0, -dist_to_1], dim=-1) / self.cluster_temperature
        weights = torch.softmax(logits, dim=-1)

        penalty0 = (1.0 - weights[..., 0]) * self.cluster_penalty_scale
        penalty1 = (1.0 - weights[..., 1]) * self.cluster_penalty_scale

        feat0 = vote_feat - penalty0.unsqueeze(-1)
        feat1 = vote_feat - penalty1.unsqueeze(-1)

        cluster0 = feat0.max(dim=1).values
        cluster1 = feat1.max(dim=1).values
        return torch.stack([cluster0, cluster1], dim=1)  # (B, 2, 512)

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None

        seed_xyz, seed_feat = None, None
        for i, sa in enumerate(self.SA_modules):
            xyz, features = sa(xyz, features)
            if i == 3:
                seed_xyz = xyz
                seed_feat = features.transpose(1, 2)

        global_feat = features.squeeze(-1)

        offsets = self.vote_offset_mlp(seed_feat) * self.max_vote_offset / 3.0
        vote_xyz = seed_xyz + offsets
        vote_feat = self.vote_feat_mlp(seed_feat)

        cluster_feats = self._cluster_votes(vote_xyz, vote_feat)  # (B, 2, 512)

        # NEW: cluster0 and cluster1 exchange information with each other
        attn_out, _ = self.cross_ring_attn(cluster_feats, cluster_feats, cluster_feats)
        cluster_feats = self.cross_ring_norm(cluster_feats + attn_out)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(global_feat.shape[0], -1)
            main_outs.append(self.main_heads[r](torch.cat([cluster_feats[:, r], slot_emb], dim=1)))
            aux_outs.append(self.aux_heads[r](torch.cat([global_feat, slot_emb], dim=1)))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)

        return main_out, aux_out, vote_xyz, seed_xyz
