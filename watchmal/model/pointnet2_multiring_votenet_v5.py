import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNetV5(nn.Module):
    """
    Same as PointNet2MultiRingVoteNetV3, with one addition: a single
    multi-head self-attention layer applied over the 64 seed points
    (post-SA4), before the vote-offset MLP runs. Each seed point can now
    incorporate context from all other seed points -- e.g. a point near
    the boundary between two rings can "see" points clearly inside each
    ring before committing to a vote direction -- rather than the offset
    MLP acting on each point's local features in isolation.

    This is a more point-cloud-native mechanism for cross-ring information
    exchange than a DETR-style slot decoder: it operates directly on the
    point set itself (dense self-attention over all 64 seeds), rather than
    routing information through external learned query vectors.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, max_vote_offset=200.0, dropout=0.3,
                 cluster_temperature=20.0, cluster_penalty_scale=1e3,
                 attn_num_heads=8, **kwargs):
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
                mlp=[256, 256, 256, 512], use_xyz=True),   # seeds
            PointnetSAModule(mlp=[512, 512, 1024], use_xyz=True),
        ])

        # NEW: self-attention over the 64 seed points, before voting.
        # batch_first=True so input/output are (B, N, embed_dim), matching
        # seed_feat's layout directly - no transpose needed.
        self.seed_self_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=attn_num_heads, batch_first=True)
        self.seed_attn_norm = nn.LayerNorm(512)

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
        return torch.stack([cluster0, cluster1], dim=1)

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

        global_feat = features.squeeze(-1)

        # Self-attention: each seed point attends to all others, residual +
        # LayerNorm (standard transformer block pattern).
        attn_out, _ = self.seed_self_attn(seed_feat, seed_feat, seed_feat)
        seed_feat = self.seed_attn_norm(seed_feat + attn_out)

        offsets = self.vote_offset_mlp(seed_feat) * self.max_vote_offset / 3.0
        vote_xyz = seed_xyz + offsets
        vote_feat = self.vote_feat_mlp(seed_feat)

        cluster_feats = self._cluster_votes(vote_xyz, vote_feat)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(global_feat.shape[0], -1)
            main_outs.append(self.main_heads[r](torch.cat([cluster_feats[:, r], slot_emb], dim=1)))
            aux_outs.append(self.aux_heads[r](torch.cat([global_feat, slot_emb], dim=1)))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)

        return main_out, aux_out, vote_xyz, seed_xyz
