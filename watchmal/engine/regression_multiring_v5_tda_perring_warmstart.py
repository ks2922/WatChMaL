import torch
from watchmal.engine.regression_multiring_v5_tda_perring import MultiRingRegressionEngineV5TDAPerRing


class MultiRingRegressionEngineV5TDAPerRingWarmStart(MultiRingRegressionEngineV5TDAPerRing):
    """
    Same as MultiRingRegressionEngineV5TDAPerRing, adding one extra method
    to load pretrained weights from a DIFFERENT (shape-mismatched)
    architecture -- e.g. warm-starting from a plain V8 checkpoint into this
    TDA-fusion model, where the head's input dimension differs due to the
    extra TDA feature. Does not modify or subclass the base
    ReconstructionEngine at all; entirely additive.
    """
    def load_pretrained_weights(self, weight_file, strict=False):
        with open(weight_file, 'rb') as f:
            checkpoint = torch.load(f, map_location=self.device)
        source_state = checkpoint['state_dict']
        target_state = self.module.state_dict()

        filtered_state = {}
        skipped = []
        for key, source_tensor in source_state.items():
            if key in target_state and target_state[key].shape == source_tensor.shape:
                filtered_state[key] = source_tensor
            else:
                skipped.append(key)

        print(f"Loaded {len(filtered_state)} / {len(source_state)} pretrained parameters")
        print(f"Skipped (shape mismatch or not present): {skipped}")

        target_state.update(filtered_state)
        self.module.load_state_dict(target_state, strict=strict)
