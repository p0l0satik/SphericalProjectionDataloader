import numpy as np

from torch.utils.data import Dataset


class SphericalProjectionKitti(Dataset):
    def __init__(self, main_directory, length) -> None:
        super().__init__()
        self.length = length
        self.load_path = main_directory

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cloud = np.load(self.load_path / "scan_{0:06d}.npz".format(idx))
        return (
            cloud["proj_points"],
            cloud["proj_labels"],
        )
