import torch
from watchmal.engine.regression_multiring_v5 import MultiRingRegressionEngineV5


class MultiRingRegressionEngineV5TDAMergeDist(MultiRingRegressionEngineV5):
    """
    Same as MultiRingRegressionEngineV5, except the model call passes the
    whole-event merge-distance TDA feature through.
    """
    def process_data(self, data):
        super().process_data(data)
        self.merge_distance = data["merge_distance"].to(self.device)

    def forward_pass(self):
        main_out, aux_out, vote_xyz, seed_xyz = self.model(
            self.data, merge_distance=self.merge_distance)
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
