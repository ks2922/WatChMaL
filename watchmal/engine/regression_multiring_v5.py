import torch
import torch.nn.functional as F
from watchmal.engine.regression_multiring_v3 import MultiRingRegressionEngineV3
from watchmal.model.pointnet2_multiring_votenet_v3 import vote_supervision_loss


class MultiRingRegressionEngineV5(MultiRingRegressionEngineV3):
    """
    Same as V3, except the identity/swap MATCHING COST (which decides ring
    assignment per event) is computed in physical units and normalized
    per-quantity-group, instead of summing raw scaled-channel Huber loss.

    Rationale: target_scale_factor exists to keep network OUTPUTS in a
    stable numeric range for training, not to make position/direction/
    energy comparable to each other for the purpose of deciding which
    permutation is "cheaper". Summing scaled-channel Huber loss directly
    (as V1-V3 do) means whichever group happens to end up numerically
    largest after scaling dominates the assignment decision, regardless
    of whether that's physically meaningful. This engine fixes the cost
    computation only -- main_loss/aux_loss/vote_loss (the actual training
    signal) are UNCHANGED from V3, still computed in scaled space as before.

    Default normalization constants (position_norm_cm, direction_norm,
    energy_norm_mev) are rough physical scales, not derived from your
    data -- worth sanity-checking against your actual per-channel residual
    distributions (e.g. plot each group's raw Huber cost separately for a
    few batches) before trusting this blindly for a full run.
    """
    def __init__(self, *args, position_norm_cm=50.0, direction_norm=0.3,
                 energy_norm_mev=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_norm_cm = position_norm_cm
        self.direction_norm = direction_norm
        self.energy_norm_mev = energy_norm_mev

    def _unscaled(self, main_out):
        pos = main_out[..., 0:3] * self.scale["positions"]
        dir_ = main_out[..., 3:6] * self.scale["directions"]
        energy = main_out[..., 6] * self.scale["energies"]
        return pos, dir_, energy

    def _normalized_cost(self, main_out, target_dict):
        pred_pos, pred_dir, pred_energy = self._unscaled(main_out)
        true_pos = target_dict["positions"]
        true_dir = target_dict["directions"]
        true_energy = target_dict["energies"]

        # Huber cost per group, in physical units, each normalized by a
        # fixed constant so no group dominates purely due to raw scale.
        cost_pos = F.huber_loss(pred_pos / self.position_norm_cm,
                                 true_pos / self.position_norm_cm,
                                 delta=self.huber_delta, reduction="none").mean(dim=-1)
        cost_dir = F.huber_loss(pred_dir / self.direction_norm,
                                 true_dir / self.direction_norm,
                                 delta=self.huber_delta, reduction="none").mean(dim=-1)
        cost_energy = F.huber_loss(pred_energy / self.energy_norm_mev,
                                    true_energy / self.energy_norm_mev,
                                    delta=self.huber_delta, reduction="none")

        return cost_pos + cost_dir + cost_energy  # (B, R)

    def forward_pass(self):
        main_out, aux_out, vote_xyz, seed_xyz = self.model(self.data)
        self._vote_xyz = vote_xyz
        self.model_out = main_out

        main_swapped = main_out.flip(dims=[1])

        cost_ii = self._normalized_cost(main_out, self.target_dict)
        cost_swap = self._normalized_cost(main_swapped, self.target_dict)

        total_identity = cost_ii.sum(dim=-1)
        total_swap = cost_swap.sum(dim=-1)
        self.assignment = (total_swap < total_identity).long()

        matched_main = torch.where(self.assignment.view(-1, 1, 1).bool(), main_swapped, main_out)
        aux_swapped = aux_out.flip(dims=[1])
        matched_aux = torch.where(self.assignment.view(-1, 1, 1).bool(), aux_swapped, aux_out)

        self._matched_main = matched_main
        self._matched_aux = matched_aux

        pred_pos = matched_main[..., 0:3] * self.scale["positions"]
        pred_dir = matched_main[..., 3:6] * self.scale["directions"]
        pred_energy = matched_main[..., 6] * self.scale["energies"]

        self.predictions = {
            "predicted_positions": pred_pos,
            "predicted_directions": pred_dir,
            "predicted_energies": pred_energy,
            "matched_model_out": matched_main,
        }

        if self.target_dict is None:
            return self.predictions
        return self.target_dict | self.predictions
