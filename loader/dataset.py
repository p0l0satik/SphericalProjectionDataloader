import numpy as np

from torch.utils.data import Dataset


class SphericalProjectionReturnValue:
    def __init__(self, proj_points, proj_labels, points=None, labels=None, proj_idx=None):
        self.proj_points = proj_points
        self.proj_labels = proj_labels
        self.points = points
        self.labels = labels
        self.proj_idx = proj_idx


class SphericalProjectionKitti(Dataset):
    def __init__(self, main_directory, length, return_orig_points=False) -> None:
        super().__init__()
        self.length = length
        self.load_path = main_directory
        self.return_orig_points = return_orig_points

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cloud = np.load(self.load_path / "scan_{0:06d}.npz".format(idx))
        if self.return_orig_points:
            return cloud["proj_points"], cloud["proj_labels"],cloud["points"],cloud["labels"], cloud["proj_idx"]
        return  cloud["proj_points"],cloud["proj_labels"],

