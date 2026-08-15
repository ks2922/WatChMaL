import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNet(nn.Module):
    """
    Point-cloud-native alternative to slot embeddings alone: each of the 64
    seed points (post-SA4) predicts a learned offset ("vote") toward the
    center of the ring it belongs to. For fixed N=2 rings, two anchor votes
    are chosen via furthest-point selection, remaining votes assigned to
    their nearer anchor, and each group's features max-pooled -- an explicit
    spatial clustering step, rather than both rings reading off one shared
    pooled global vector.

    Also keeps a lightweight global-pooled auxiliary head (same role as
    PointNet2MultiRingV2's aux head), so this model is drop-in compatible
    with MultiRingRegressionEngineV2 (returns (main_out, aux_out), same
    shapes as V2).
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, max_vote_offset=200.0, dropout=0.3, **kwargs):
        super().__init__()
        assert num_rings == 2, "fixed-K anchor clustering below only implemented for num_rings=2"
        in_feat = num_input_channels - 3
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring
        self.max_vote_offset = max_vote_offset

        self.SA_modules = nn.ModuleList([
            PointnetSAModule(npoint=1024, radius=15.0, nsample=32,
                mlp=[in_feat, 32, 32, 64], use_xyz=True),
            PointnetSAModule(npoint=512, radius=30.0, nsample=32,
                mlp=[64, 64, 64, 128], use_xyz=True),
            PointnetSAModule(npoint=256, radius=80.0, nsample=32,
                mlp=[128, 128, 128, 256], use_xyz=True),
            PointnetSAModule(npoint=64, radius=180.0, nsample=32,
                mlp=[256, 256, 256, 512], use_xyz=True),   # seeds for voting
            PointnetSAModule(mlp=[512, 512, 1024], use_xyz=True),  # global feature (aux head)
        ])

        self.vote_offset_mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.vote_feat_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
        )

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
        """
        vote_xyz: (B, 64, 3), vote_feat: (B, 64, 512)
        Returns two (B, 512) cluster features, via furthest-point anchor
        selection + nearest-anchor assignment + max-pool per group.
        """
        B, N, _ = vote_xyz.shape
        centroid = vote_xyz.mean(dim=1, keepdim=True)  # (B,1,3)
        d0 = torch.linalg.norm(vote_xyz - centroid, dim=-1)  # (B,N)
        anchor0_idx = d0.argmax(dim=1)  # (B,)
        anchor0 = vote_xyz[torch.arange(B), anchor0_idx].unsqueeze(1)  # (B,1,3)
        d1 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)
        anchor1_idx = d1.argmax(dim=1)
        anchor1 = vote_xyz[torch.arange(B), anchor1_idx].unsqueeze(1)

        dist_to_0 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)  # (B,N)
        dist_to_1 = torch.linalg.norm(vote_xyz - anchor1, dim=-1)
        assign_to_0 = (dist_to_0 <= dist_to_1)  # (B,N) bool

        neg_inf = torch.finfo(vote_feat.dtype).min
        feat0 = torch.where(assign_to_0.unsqueeze(-1), vote_feat, torch.full_like(vote_feat, neg_inf))
        feat1 = torch.where(~assign_to_0.unsqueeze(-1), vote_feat, torch.full_like(vote_feat, neg_inf))
        cluster0 = feat0.max(dim=1).values  # (B,512)
        cluster1 = feat1.max(dim=1).values
        return torch.stack([cluster0, cluster1], dim=1)  # (B,2,512)

    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        features = x[:, 3:, :].contiguous()
        if features.shape[1] == 0:
            features = None

        seed_xyz, seed_feat = None, None
        for i, sa in enumerate(self.SA_modules):
            xyz, features = sa(xyz, features)
            if i == 3:
                seed_xyz = xyz                          # (B,64,3)
                seed_feat = features.transpose(1, 2)     # (B,64,512)

        global_feat = features.squeeze(-1)  # (B,1024), from SA5

        offsets = self.vote_offset_mlp(seed_feat) * self.max_vote_offset / 3.0  # bounded-ish, cm scale
        vote_xyz = seed_xyz + offsets
        vote_feat = self.vote_feat_mlp(seed_feat)  # (B,64,512)

        cluster_feats = self._cluster_votes(vote_xyz, vote_feat)  # (B,2,512)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(global_feat.shape[0], -1)
            main_outs.append(self.main_heads[r](torch.cat([cluster_feats[:, r], slot_emb], dim=1)))
            aux_outs.append(self.aux_heads[r](torch.cat([global_feat, slot_emb], dim=1)))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)
        return main_out, aux_out
