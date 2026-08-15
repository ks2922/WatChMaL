import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNetV4(nn.Module):
    """
    Ablation of PointNet2MultiRingVoteNetV3: identical architecture MINUS
    slot_embeddings. Tests whether the learned per-ring identity tag adds
    anything beyond what separately-weighted per-ring MLP heads already
    provide on their own -- each head has independent parameters, so ring
    identity can emerge purely from training under the permutation-matching
    loss, without an explicit embedding telling each head "which ring" it is.

    Returns the same (main_out, aux_out, vote_xyz, seed_xyz) tuple as V3,
    so it's a drop-in swap with MultiRingRegressionEngineV3 -- no engine
    changes needed for this ablation.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 max_vote_offset=200.0, dropout=0.3,
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

        # NOTE: no self.slot_embeddings here - this is the whole point of
        # this ablation. main_heads/aux_heads below take ONLY the
        # cluster/global feature as input, nothing concatenated.

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 128), nn.ReLU(),
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
                seed_xyz = xyz
                seed_feat = features.transpose(1, 2)

        global_feat = features.squeeze(-1)

        offsets = self.vote_offset_mlp(seed_feat) * self.max_vote_offset / 3.0
        vote_xyz = seed_xyz + offsets
        vote_feat = self.vote_feat_mlp(seed_feat)

        cluster_feats = self._cluster_votes(vote_xyz, vote_feat)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            # No slot_emb concatenation - each head sees only its own
            # cluster feature / the shared global feature, distinguished
            # purely by having independent weights.
            main_outs.append(self.main_heads[r](cluster_feats[:, r]))
            aux_outs.append(self.aux_heads[r](global_feat))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)

        return main_out, aux_out, vote_xyz, seed_xyz
