import torch
import torch.nn.functional as F
from watchmal.engine.regression_multiring_v2 import MultiRingRegressionEngineV2
from watchmal.model.pointnet2_multiring_votenet_v3 import vote_supervision_loss


class MultiRingRegressionEngineV3(MultiRingRegressionEngineV2):
    def __init__(self, *args, vote_loss_weight=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.vote_loss_weight = vote_loss_weight

    def forward_pass(self):
        main_out, aux_out, vote_xyz, seed_xyz = self.model(self.data)
        self._vote_xyz = vote_xyz
        self.model_out = main_out

        tgt = self.target_stack

        cost_ii = F.huber_loss(main_out, tgt, delta=self.huber_delta, reduction="none").sum(dim=-1)
        main_swapped = main_out.flip(dims=[1])
        cost_swap = F.huber_loss(main_swapped, tgt, delta=self.huber_delta, reduction="none").sum(dim=-1)

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

    def compute_metrics(self):
        main_loss = F.huber_loss(self._matched_main, self.target_stack, delta=self.huber_delta, reduction="mean")
        aux_loss = F.huber_loss(self._matched_aux, self.target_stack, delta=self.huber_delta, reduction="mean")
        vote_loss = vote_supervision_loss(self._vote_xyz, self.target_dict["positions"])

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
