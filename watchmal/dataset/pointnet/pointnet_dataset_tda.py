"""
PointNet dataset with optional precomputed TDA features.
"""

import numpy as np

from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du


class PointNetDataset(H5Dataset):
    def __init__(self, h5file, geometry_file, use_times=True, use_orientations=False,
                 n_points=4000, transforms=None, use_memmap=True,
                 tda_feature_file=None, tda_key="tda_features"):
        super().__init__(h5file, use_memmap)

        geo_file = np.load(geometry_file, "r")
        self.geo_positions = geo_file["position"].astype(np.float32)
        self.geo_orientations = geo_file["orientation"].astype(np.float32)

        self.use_orientations = use_orientations
        self.use_times = use_times
        self.n_points = n_points
        self.transforms = du.get_transformations(transformations, transforms) or []

        self.channels = 4
        if use_orientations:
            self.channels += 3
        if use_times:
            self.channels += 1

        self.tda_features = None
        if tda_feature_file is not None:
            tda_npz = np.load(tda_feature_file)
            self.tda_features = tda_npz[tda_key].astype(np.float32)

            self.tda_n_events = self.tda_features.shape[0]

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        n_hits = min(self.n_points, self.event_hit_pmts.shape[0])
        hit_positions = self.geo_positions[self.event_hit_pmts[:n_hits], :]

        data = np.zeros((self.channels, self.n_points), dtype=np.float32)
        data[:3, :n_hits] = hit_positions.T

        if self.use_orientations:
            hit_orientations = self.geo_orientations[self.event_hit_pmts[:n_hits], :]
            data[3:6, :n_hits] = hit_orientations.T

        if self.use_times:
            data[-2, :n_hits] = self.event_hit_times[:n_hits]

        data[-1, :n_hits] = self.event_hit_charges[:n_hits]

        for t in self.transforms:
            data = t(data)

        data_dict["data"] = data

        if self.tda_features is not None:
            if item < self.tda_n_events:
                data_dict["tda_features"] = self.tda_features[item]
            else:
                data_dict["tda_features"] = np.zeros(self.tda_features.shape[1], dtype=np.float32)

        return data_dict
