import torch
import torch.nn.functional as F
from watchmal.engine.regression_multiring_v3 import MultiRingRegressionEngineV3
from watchmal.model.pointnet2_multiring_votenet_v3 import vote_supervision_loss


class MultiRingRegressionEngineV3DirWeight(MultiRingRegressionEngineV3):
    """
    Same as MultiRingRegressionEngineV3, except main_loss upweights the
    direction channels (indices 3:6) relative to position/energy, to test
    whether direction's consistent underperformance under clustering is
    fixable by giving it more gradient weight, rather than an architecture
    change. Cheapest possible intervention - one loss-weighting change,
    no new model needed, reuses PointNet2MultiRingVoteNetV3 unchanged.

    Caveat: this is a less principled fix than the architectural variants
    (V7/V8/V9) - it can trade the deficit onto position/energy rather than
    genuinely resolving the underlying information bottleneck. Worth
    comparing against V7/V8/V9 rather than treating in isolation.
    """
    def __init__(self, *args, direction_loss_weight=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_loss_weight = direction_loss_weight

    def compute_metrics(self):
        # Per-channel Huber, then apply extra weight to direction channels
        # (indices 3:6) before averaging - same total contribution from
        # position/energy as V3, direction weighted up by direction_loss_weight.
        per_channel_main = F.huber_loss(
            self._matched_main, self.target_stack, delta=self.huber_delta, reduction="none"
        )  # (B, R, 7)
        channel_weights = torch.ones(7, device=per_channel_main.device)
        channel_weights[3:6] = self.direction_loss_weight
        main_loss = (per_channel_main * channel_weights).mean()

        aux_loss = F.huber_loss(self._matched_aux, self.target_stack, delta=self.huber_delta, reduction="mean")
        vote_loss = vote_supervision_loss(self._vote_xyz, self.target_dict["positions"])
            self._vote_xyz, self.target_dict["positions"]
        )

        self.loss = main_loss + self.aux_loss_weight * aux_loss + self.vote_loss_weight * vote_loss

        pred_pos = self.predictions["predicted_positions"]
        pred_dir = self.predictions["predicted_directions"]
        pred_energy = self.predictions["predicted_energies"]
        true_pos = self.target_dict["positions"]
        true_dir = self.target_dict["directions"]
        true_energy = self.target_dict["energies"]

        position_error = torch.linalg.vector_norm(pred_pos - true_pos, dim=-1).mean()
        dir_cos = torch.sum(pred_dir * true_dir, dim=-1) / torch.linalg.vector_norm(pred_dir, dim=-1)
        direction_error = torch.arccos(torch.clamp(dir_cos, -1, 1)).mean()
        energy_bias = torch.mean((pred_energy - true_energy) / true_energy)
        energy_error = torch.mean(torch.abs(pred_energy - true_energy) / true_energy)

        return {
            "loss": self.loss,
            "main_loss": main_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "vote_loss": vote_loss.detach(),
            "position error": position_error,
            "direction error": direction_error,
            "energy bias": energy_bias,
            "energy error": energy_error,
        }
