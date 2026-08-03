import torch
from watchmal.engine.regression import RegressionEngine as BaseRegressionEngine


class RegressionEngine(BaseRegressionEngine):
    """
    TDA-aware regression engine.
    Only overrides process_data and forward_pass to pass optional
    tda_features to the model. All other behavior (compute_metrics,
    step, train, evaluate) is inherited unchanged from base.
    """

    def process_data(self, data):
        super().process_data(data)
        self.tda_features = None
        if isinstance(data, dict) and "tda_features" in data:
            self.tda_features = data["tda_features"].to(self.device)

    def forward_pass(self):
        if self.tda_features is not None:
            self.model_out = self.model(self.data, self.tda_features)
        else:
            self.model_out = self.model(self.data)

        split_model_out = torch.split(self.model_out, self.target_sizes, dim=1)
        self.predictions = {"predicted_" + t: o * self.scale[t] + self.offset[t]
                            for t, o in zip(self.target_key, split_model_out)}
        if self.target_dict is None:
            return self.predictions
        return self.target_dict | self.predictions
