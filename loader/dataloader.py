import numpy as np
from torch.utils.data import Dataset
from loader.laserscan import SemLaserScan

class KittiSimple(Dataset):
    def __init__(self, dir, len) -> None:
        super().__init__()
        self.ls = SemLaserScan()
        self.len = len
        self.label_folder = dir + "plane_labels/"
        self.road_label_folder = dir + "labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        KITI_ROAD_LABELS = (40, 48, 44, 49)
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        sem_label, proj_sem_label = self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        sem_rd_label, proj_sem_rd_label = self.ls.open_label(self.road_label_folder  + '{0:06d}.label'.format(idx))

        sem_label[(sem_label != 0) | (np.isin(sem_rd_label, KITI_ROAD_LABELS))] = 1
        proj_sem_label[(proj_sem_label != 0) | (np.isin(proj_sem_rd_label, KITI_ROAD_LABELS))] = 1

        return self.ls.points, np.transpose(self.ls.proj_xyz, (2, 0, 1)), sem_label, proj_sem_label, self.ls.proj_idx
