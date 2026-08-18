import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule


class PointNet2MultiRingVoteNetV10(nn.Module):
    """
    Combines V6 (k-NN restricted self-attention over the 64 seed points,
    applied before voting) with V8 (2-token cross-ring attention between
    the two pooled cluster features, applied after clustering).

    Rationale: these operate at different pipeline stages and address
    different problems -- V6 preserves fine spatial detail during the
    seed->vote->cluster step (fixing V5's smearing failure mode), while
    V8 lets the two already-separated ring summaries share context with
    each other afterward. No overlap, so combining them is a genuine test
    of additivity rather than redundancy.

    use_slot_embeddings=False reproduces the V4 ablation finding (slot
    identity may not be load-bearing once clustering does the separation)
    on top of this combined architecture, without a separate near-
    duplicate file.

    Use with n_points=2000.
    """
    def __init__(self, num_input_channels, num_rings=2, channels_per_ring=7,
                 slot_embed_dim=32, max_vote_offset=200.0, dropout=0.3,
                 cluster_temperature=20.0, cluster_penalty_scale=1e3,
                 attn_num_heads=8, knn_k=16, cross_ring_attn_heads=4,
                 use_slot_embeddings=True, **kwargs):
        super().__init__()
        assert num_rings == 2, "fixed-K anchor clustering below only implemented for num_rings=2"
        in_feat = num_input_channels - 3
        self.num_rings = num_rings
        self.channels_per_ring = channels_per_ring
        self.max_vote_offset = max_vote_offset
        self.cluster_temperature = cluster_temperature
        self.cluster_penalty_scale = cluster_penalty_scale
        self.attn_num_heads = attn_num_heads
        self.knn_k = knn_k
        self.use_slot_embeddings = use_slot_embeddings

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

        # V6: k-NN restricted self-attention over seed points, before voting
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

        # V8: 2-token cross-ring attention over [cluster0, cluster1], after clustering
        self.cross_ring_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=cross_ring_attn_heads, batch_first=True)
        self.cross_ring_norm = nn.LayerNorm(512)

        self.slot_embeddings = nn.Parameter(torch.randn(num_rings, slot_embed_dim) * 0.02)
        head_in_dim = 512 + (slot_embed_dim if use_slot_embeddings else 0)

        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

        aux_in_dim = 1024 + (slot_embed_dim if use_slot_embeddings else 0)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(aux_in_dim, 128), nn.ReLU(),
                nn.Linear(128, channels_per_ring),
            ) for _ in range(num_rings)
        ])

    def _build_knn_mask(self, seed_xyz):
        """Returns a boolean attn_mask of shape (B * num_heads, N, N), True = NOT
        allowed to attend. Each point attends to itself and its k-1 nearest
        neighbours (identical to V6's implementation)."""
        B, N, _ = seed_xyz.shape
        k = min(self.knn_k, N)

        dists = torch.cdist(seed_xyz, seed_xyz)
        knn_idx = dists.topk(k, largest=False).indices

        mask = torch.ones(B, N, N, dtype=torch.bool, device=seed_xyz.device)
        mask.scatter_(-1, knn_idx, False)

        mask = mask.unsqueeze(1).expand(B, self.attn_num_heads, N, N)
        mask = mask.reshape(B * self.attn_num_heads, N, N)
        return mask

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

        # V6: k-NN restricted self-attention on seeds, before voting
        knn_mask = self._build_knn_mask(seed_xyz)
        attn_out, _ = self.seed_self_attn(seed_feat, seed_feat, seed_feat, attn_mask=knn_mask)
        seed_feat = self.seed_attn_norm(seed_feat + attn_out)

        offsets = self.vote_offset_mlp(seed_feat) * self.max_vote_offset / 3.0
        vote_xyz = seed_xyz + offsets
        vote_feat = self.vote_feat_mlp(seed_feat)

        cluster_feats = self._cluster_votes(vote_xyz, vote_feat)

        # V8: cross-ring attention between the two cluster summaries
        cross_out, _ = self.cross_ring_attn(cluster_feats, cluster_feats, cluster_feats)
        cluster_feats = self.cross_ring_norm(cluster_feats + cross_out)

        main_outs, aux_outs = [], []
        for r in range(self.num_rings):
            if self.use_slot_embeddings:
                slot_emb = self.slot_embeddings[r].unsqueeze(0).expand(global_feat.shape[0], -1)
                main_in = torch.cat([cluster_feats[:, r], slot_emb], dim=1)
                aux_in = torch.cat([global_feat, slot_emb], dim=1)
            else:
                main_in = cluster_feats[:, r]
                aux_in = global_feat
            main_outs.append(self.main_heads[r](main_in))
            aux_outs.append(self.aux_heads[r](aux_in))

        main_out = torch.stack(main_outs, dim=1)
        aux_out = torch.stack(aux_outs, dim=1)

        return main_out, aux_out, vote_xyz, seed_xyz
