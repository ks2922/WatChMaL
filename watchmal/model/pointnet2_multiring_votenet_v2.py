import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNetV2(nn.Module):
    """
    Same architecture as PointNet2MultiRingVoteNet, with one fix:
    `_cluster_votes` now uses a SOFT, differentiable assignment instead of a
    hard boolean mask, so gradients actually flow back through `vote_xyz`
    into `vote_offset_mlp`. In the original version, the hard `<=` boolean
    comparison and `torch.where` cut off any gradient to the offset MLP,
    so it never received training signal and stayed near its random
    initialisation throughout training.

    Pooling is kept as max-pool (matching the original), implemented as a
    soft relaxation of the original -inf masking trick: instead of setting
    excluded points' features to -inf before max-pooling, excluded points
    get a soft, distance-weighted penalty subtracted from their features.
    As the soft assignment weight for a point approaches 1 (confidently
    assigned to that cluster), the penalty approaches 0 and the feature
    value is preserved exactly as in the original hard-masked version --
    so this recovers the original behaviour in the limit, while remaining
    differentiable everywhere in between.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, max_vote_offset=200.0, dropout=0.3,
                 cluster_temperature=20.0, cluster_penalty_scale=1e3, **kwargs):
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
        selection + SOFT (differentiable) assignment + max-pool per group.

        Anchor selection (furthest-point) is still non-differentiable
        (argmax) -- that's fine, since the anchors are just reference
        points for computing per-point assignment weights, not something
        we need gradient on directly. What matters is that vote_xyz's
        influence on the final pooled features now flows through a
        differentiable path (the softmax weights -> penalty -> max), which
        it did not in the original hard-masked version.
        """
        B, N, _ = vote_xyz.shape
        centroid = vote_xyz.mean(dim=1, keepdim=True)  # (B,1,3)
        d0 = torch.linalg.norm(vote_xyz - centroid, dim=-1)  # (B,N)
        anchor0_idx = d0.argmax(dim=1)
        anchor0 = vote_xyz[torch.arange(B), anchor0_idx].unsqueeze(1)  # (B,1,3)
        d1 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)
        anchor1_idx = d1.argmax(dim=1)
        anchor1 = vote_xyz[torch.arange(B), anchor1_idx].unsqueeze(1)

        dist_to_0 = torch.linalg.norm(vote_xyz - anchor0, dim=-1)  # (B,N)
        dist_to_1 = torch.linalg.norm(vote_xyz - anchor1, dim=-1)

        # Soft assignment weights, differentiable in vote_xyz (and hence in
        # vote_offset_mlp's parameters). weights[...,0] + weights[...,1] = 1.
        logits = torch.stack([-dist_to_0, -dist_to_1], dim=-1) / self.cluster_temperature  # (B,N,2)
        weights = torch.softmax(logits, dim=-1)  # (B,N,2)

        # Soft relaxation of the original hard mask: subtract a penalty
        # proportional to (1 - weight) before max-pooling. A confidently
        # assigned point (weight -> 1) gets ~0 penalty, exactly recovering
        # the original unmasked feature value; a confidently excluded
        # point (weight -> 0) gets a large penalty, pushing it out of the
        # max competition -- same effect as the original -inf masking, but
        # smooth and differentiable near the cluster boundary.
        penalty0 = (1.0 - weights[..., 0]) * self.cluster_penalty_scale  # (B,N)
        penalty1 = (1.0 - weights[..., 1]) * self.cluster_penalty_scale

        feat0 = vote_feat - penalty0.unsqueeze(-1)
        feat1 = vote_feat - penalty1.unsqueeze(-1)

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
