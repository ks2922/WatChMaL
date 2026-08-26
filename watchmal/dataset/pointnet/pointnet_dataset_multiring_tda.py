"""
Multi-ring PointNet datasets with precomputed TDA features attached.
Both classes build an idx_to_pos lookup (raw h5 event index -> row position
in the TDA feature file) lazily in initialize(). The TDA feature arrays
themselves are kept open as h5py Dataset objects (not materialized via
np.array) so each per-event read pulls only that row from disk, rather
than every DataLoader worker holding a full in-memory copy of the whole
feature array -- this caused an OOM kill in an earlier version that used
np.array(...) here.
"""

import numpy as np
import h5py
from watchmal.dataset.pointnet.pointnet_dataset import PointNetDataset


class PointNetDatasetMultiRingTDAPerRing(PointNetDataset):
    def __init__(self, h5file, geometry_file, tda_features_file, use_times=True,
                 use_orientations=False, n_points=2000, transforms=None, use_memmap=True):
        super().__init__(h5file, geometry_file, use_times, use_orientations,
                          n_points, transforms, use_memmap)
        self.tda_features_file = tda_features_file
        self.tda_h5 = None
        self.idx_to_pos = None

    def initialize(self):
        super().initialize()
        self.tda_h5 = h5py.File(self.tda_features_file, "r")
        event_idxs = self.tda_h5["event_idxs"][:]
        self.idx_to_pos = {int(idx): pos for pos, idx in enumerate(event_idxs)}

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        pos = self.idx_to_pos.get(int(item))
        if pos is None:
            data_dict["tda_features_ring0"] = np.zeros(self.tda_h5["tda_features_ring0"].shape[1], dtype=np.float32)
            data_dict["tda_features_ring1"] = np.zeros(self.tda_h5["tda_features_ring1"].shape[1], dtype=np.float32)
        else:
            data_dict["tda_features_ring0"] = self.tda_h5["tda_features_ring0"][pos].astype(np.float32)
            data_dict["tda_features_ring1"] = self.tda_h5["tda_features_ring1"][pos].astype(np.float32)
        return data_dict


class PointNetDatasetMultiRingTDAMergeDist(PointNetDataset):
    def __init__(self, h5file, geometry_file, merge_distance_file, use_times=True,
                 use_orientations=False, n_points=2000, transforms=None, use_memmap=True):
        super().__init__(h5file, geometry_file, use_times, use_orientations,
                          n_points, transforms, use_memmap)
        self.merge_distance_file = merge_distance_file
        self.merge_h5 = None
        self.idx_to_pos = None

    def initialize(self):
        super().initialize()
        self.merge_h5 = h5py.File(self.merge_distance_file, "r")
        event_idxs = self.merge_h5["event_idxs"][:]
        self.idx_to_pos = {int(idx): pos for pos, idx in enumerate(event_idxs)}

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        pos = self.idx_to_pos.get(int(item))
        if pos is None:
            data_dict["merge_distance"] = np.float32(0.0)
        else:
            data_dict["merge_distance"] = np.float32(self.merge_h5["merge_distance"][pos])
        return data_dict


class PointNetDatasetMultiRingTDAPerRingNpy(PointNetDataset):
    """
    Same as PointNetDatasetMultiRingTDAPerRing, but loads TDA features from
    pre-converted .npy files with mmap_mode='r' instead of h5py.Dataset
    indexing -- tests whether h5py's per-call read overhead is the actual
    bottleneck versus the gate mechanism itself.
    """
    def __init__(self, h5file, geometry_file, tda_features_prefix, use_times=True,
                 use_orientations=False, n_points=2000, transforms=None, use_memmap=True):
        super().__init__(h5file, geometry_file, use_times, use_orientations,
                          n_points, transforms, use_memmap)
        self.tda_features_prefix = tda_features_prefix
        self.tda_ring0 = None
        self.tda_ring1 = None
        self.idx_to_pos = None

    def initialize(self):
        super().initialize()
        self.tda_ring0 = np.load(f"{self.tda_features_prefix}_ring0.npy", mmap_mode='r')
        self.tda_ring1 = np.load(f"{self.tda_features_prefix}_ring1.npy", mmap_mode='r')
        event_idxs = np.load(f"{self.tda_features_prefix}_event_idxs.npy")
        self.idx_to_pos = {int(idx): pos for pos, idx in enumerate(event_idxs)}

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        pos = self.idx_to_pos.get(int(item))
        if pos is None:
            data_dict["tda_features_ring0"] = np.zeros(self.tda_ring0.shape[1], dtype=np.float32)
            data_dict["tda_features_ring1"] = np.zeros(self.tda_ring1.shape[1], dtype=np.float32)
        else:
            data_dict["tda_features_ring0"] = np.array(self.tda_ring0[pos], dtype=np.float32)
            data_dict["tda_features_ring1"] = np.array(self.tda_ring1[pos], dtype=np.float32)
        return data_dict
