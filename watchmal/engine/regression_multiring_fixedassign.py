import torch
from watchmal.engine.regression_multiring import MultiRingRegressionEngine


class MultiRingRegressionEngineFixedAssign(MultiRingRegressionEngine):
    """
    Same as MultiRingRegressionEngine, except self.assignment is always
    forced to identity (0), never computed from matching cost -- removes
    the matching-loss-induced non-stationarity in what each output slot
    represents across training iterations, isolating whether this is the
    cause of BatchNorm's running-statistic divergence observed in the
    ResNet multi-ring baseline. Diagnostic test only, not a real
    architecture improvement (fixed assignment is a strictly worse
    training signal than the real matched assignment).
    """
    def forward_pass(self):
        self.model_out = self.model(self.data)
        pred = self.model_out
        tgt = self.target_stack

        # always identity -- no cost computed, no swap ever applied
        self.assignment = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
        matched_pred = pred  # identity assignment means matched_pred == pred, unconditionally

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
