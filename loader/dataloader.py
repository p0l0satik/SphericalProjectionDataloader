import numpy as np

from torch.utils.data import Dataset
from pathlib import Path

import loader.constants as constants
from loader.laserscan import LaserScan

class SphericalProjectionKitti(Dataset):
    def __init__(self, main_directory, length, visualise_ransac=False) -> None:
        super().__init__()
        self.ls = LaserScan(visualise_ransac=visualise_ransac)
        self.length = length
        self.label_folder = main_directory / constants.PLANES_LABELS
        self.kitti_label_folder = main_directory / constants.KITTI_LABELS
        self.pointclouds = main_directory / constants.POINT_CLOUDS

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        self.ls.open_scan(self.pointclouds / "{0:06d}.bin".format(idx))
        sem_label, proj_sem_label = self.ls.open_label(self.label_folder / "label-{0:06d}.npy".format(idx))
        sem_rd_label, proj_sem_rd_label = self.ls.open_label(self.kitti_label_folder / "{0:06d}.label".format(idx), ransac=False)

        sem_label[np.isin(sem_rd_label, constants.KITI_ROAD_LABELS)] = 1
        proj_sem_label[np.isin(proj_sem_rd_label, constants.KITI_ROAD_LABELS)] = 1

        return (
            self.ls.points,
            np.transpose(self.ls.proj_xyz, (2, 0, 1)),
            sem_label,
            proj_sem_label,
            self.ls.proj_idx,
        )
