"""
Engine for two-ring baseline regression with permutation-invariant (matching)
loss. Handles the N=2 ring assignment problem directly: for each event, both
possible predicted-slot-to-ground-truth-ring assignments are scored, and the
lower-cost assignment is used for the loss and metrics.

Targets are requested as 'positions', 'angles', 'energies' (NOT 'directions')
because H5Dataset.load_target's direction_from_angles conversion is not shape-safe
for the (N, 2, 2) multi-ring angle array -- see project notes. This engine performs
the angle-to-direction conversion itself using torch ops that generalize correctly
over the ring dimension.
"""

import torch
import torch.nn.functional as F
from collections.abc import Mapping

from watchmal.engine.reconstruction import ReconstructionEngine


def direction_from_angles_torch(angles, zenith_axis=1):
    """Torch equivalent of watchmal.utils.math.direction_from_angles, safe for
    arbitrary leading dims, e.g. (B, num_rings, 2) -> (B, num_rings, 3)."""
    zenith = angles[..., 0]
    azimuth = angles[..., 1]
    dir_along = torch.cos(zenith)
    trans_x = torch.sin(zenith) * torch.cos(azimuth)
    trans_z = torch.sin(zenith) * torch.sin(azimuth)
    # matches np.insert(dir_trans, 1, dir_along, axis=1): [trans_x, dir_along, trans_z]
    return torch.stack([trans_x, dir_along, trans_z], dim=-1)


class MultiRingRegressionEngine(ReconstructionEngine):
    """Two-ring baseline regression engine with matching-based loss."""

    def __init__(self, target_key, model, rank, device, dump_path,
                 target_scale_offset=0, target_scale_factor=1,
                 num_rings=2, huber_delta=0.1):
        super().__init__(target_key, model, rank, device, dump_path)
        if isinstance(self.target_key, str):
            self.target_key = [self.target_key]
        # expect exactly ['positions', 'angles', 'energies'] (order doesn't matter here,
        # since we build the per-ring vector explicitly below rather than via column_stack)
        self.num_rings = num_rings
        self.huber_delta = huber_delta

        if isinstance(target_scale_factor, Mapping):
            self.scale = {t: torch.tensor(target_scale_factor.get(t, 1), dtype=torch.float32).to(self.device)
                          for t in ("positions", "directions", "energies")}
        else:
            self.scale = {t: torch.tensor(target_scale_factor, dtype=torch.float32).to(self.device)
                          for t in ("positions", "directions", "energies")}

        self.target_dict = None
        self.target_stack = None
        self.predictions = None
        self.assignment = None

    def process_target(self, data):
        """Build (B, num_rings, 7) target tensor: [position(3), direction(3), energy(1)] per ring."""
        positions = data["positions"].to(self.device).float()      # (B, R, 3)
        angles = data["angles"].to(self.device).float()            # (B, R, 2)
        energies = data["energies"].to(self.device).float()        # (B, R)
        directions = direction_from_angles_torch(angles)           # (B, R, 3)

        self.target_dict = {"positions": positions, "directions": directions, "energies": energies}

        scaled_pos = positions / self.scale["positions"]
        scaled_dir = directions / self.scale["directions"]
        scaled_energy = (energies / self.scale["energies"]).unsqueeze(-1)  # (B, R, 1)

        self.target_stack = torch.cat([scaled_pos, scaled_dir, scaled_energy], dim=-1)  # (B, R, 7)

    def forward_pass(self):
        """Predict (B, num_rings, 7), find best ring assignment, and unscale predictions."""
        self.model_out = self.model(self.data)  # (B, R, 7), already ring-shaped by model

        # --- N=2 matching: compute cost for both assignments, pick cheaper per event ---
        pred = self.model_out  # (B, R, 7)
        tgt = self.target_stack  # (B, R, 7)

        # elementwise huber cost per (predicted slot i, target ring j) pair, summed over the 7 channels
        # cost_matrix[b, i, j] = sum_c huber(pred[b,i,c], tgt[b,j,c])
        cost_ii = F.huber_loss(pred, tgt, delta=self.huber_delta, reduction="none").sum(dim=-1)  # identity assignment, (B, R)
        pred_swapped = pred.flip(dims=[1])  # swap ring slots 0 and 1
        cost_swap = F.huber_loss(pred_swapped, tgt, delta=self.huber_delta, reduction="none").sum(dim=-1)  # (B, R)

        total_identity = cost_ii.sum(dim=-1)   # (B,)
        total_swap = cost_swap.sum(dim=-1)     # (B,)

        # assignment: 0 = identity (pred[0]->tgt[0], pred[1]->tgt[1]), 1 = swap
        self.assignment = (total_swap < total_identity).long()  # (B,)

        matched_pred = torch.where(
            self.assignment.view(-1, 1, 1).bool(),
            pred_swapped,
            pred,
        )  # (B, R, 7), predictions reordered to match targets

        # unscale for reporting/metrics
        pred_pos = matched_pred[..., 0:3] * self.scale["positions"]
        pred_dir = matched_pred[..., 3:6] * self.scale["directions"]
        pred_energy = matched_pred[..., 6] * self.scale["energies"]

        self.predictions = {
            "predicted_positions": pred_pos,
            "predicted_directions": pred_dir,
            "predicted_energies": pred_energy,
            "matched_model_out": matched_pred,
        }

        if self.target_dict is None:
            return self.predictions
        return self.target_dict | self.predictions

    def compute_metrics(self):
        matched_pred = self.predictions["matched_model_out"]  # (B, R, 7), scaled space
        # final scalar loss: mean huber over all ring/channel elements using the matched assignment
        self.loss = F.huber_loss(matched_pred, self.target_stack, delta=self.huber_delta, reduction="mean")

        pred_pos = self.predictions["predicted_positions"]
        pred_dir = self.predictions["predicted_directions"]
        pred_energy = self.predictions["predicted_energies"]
        true_pos = self.target_dict["positions"]
        true_dir = self.target_dict["directions"]
        true_energy = self.target_dict["energies"]

        # per-ring metrics, averaged over rings
        position_error = torch.linalg.vector_norm(pred_pos - true_pos, dim=-1).mean()
        dir_cos = torch.sum(pred_dir * true_dir, dim=-1) / torch.linalg.vector_norm(pred_dir, dim=-1)
        direction_error = torch.arccos(torch.clamp(dir_cos, -1, 1)).mean()
        energy_bias = torch.mean((pred_energy - true_energy) / true_energy)
        energy_error = torch.mean(torch.abs(pred_energy - true_energy) / true_energy)

        return {
            "loss": self.loss,
            "position error": position_error,
            "direction error": direction_error,
            "energy bias": energy_bias,
            "energy error": energy_error,
        }

    def save_state(self, suffix="", name=None):
        super().save_state(suffix, name)

    def restore_state(self, weight_file):
        super().restore_state(weight_file)
