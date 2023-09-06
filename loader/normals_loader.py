import numpy as np
from torch.utils.data import Dataset
from loader.laserscan import SemLaserScan

def prep_long_2_right_up(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+2] - orig[:, i+2, j+2]
            v2 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat

def prep_long_2_half_right_up(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i+1, j+2] - orig[:, i+2, j+2]
            v2 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat

def prep_long_5_right_up(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=5)[5:8]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i+4, j+5] - orig[:, i+5, j+5]
            v2 = orig[:, i+5, j+10] - orig[:, i+5, j+5]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat

def prep_long_5(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=5)[5:8]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+5] - orig[:, i+5, j+5]
            v2 = orig[:, i+5, j+10] - orig[:, i+5, j+5]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat


def prep_stock(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            norm_mat[:, i, j] = np.cross(v1, v2)
    return norm_mat

def prep_cross_aver(orig):
    norm_mat = np.zeros_like(orig)
    orig = np.pad(orig, pad_width=1)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]
            norm_mat[:, i, j] = np.mean((np.cross(v1, v2), np.cross(v3, v4)), axis = 0)
    return norm_mat

def prep_cov(orig):
    norm_mat = np.zeros_like(orig)
    orig = np.pad(orig, pad_width=1)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            p = orig[:, i, j] + (orig[:, i, j+1] + orig[:, i, j+2] + orig[:, i+1, j] + orig[:, i+1, j+1] + orig[:, i+1, j+2] + orig[:, i+2, j] + orig[:, i+2, j+1] + orig[:, i+2, j+2]) / 9
            s = np.zeros((3, 3), dtype=np.float)
            for l in range(3):
                for m in range(3):
                    s += p @ p.T
            s /= 9
            val, vec = np.linalg.eig(s)       
            val, vec = np.real(val), np.real(vec)
            norm_mat[:, i, j] = vec[np.argmin(val)]
    return norm_mat


# def useless(orig):
#     norm_mat = np.zeros((3, 64, 1024))

#     orig = np.pad(orig, pad_width=1)[1:4]

#     for i in range(norm_mat.shape[1]):
#         for j in range(norm_mat.shape[2]):
#             p = orig[:, i, j] + (orig[:, i, j+1] + orig[:, i, j+2] + orig[:, i+1, j] + orig[:, i+1, j+1] + orig[:, i+1, j+2] + orig[:, i+2, j] + orig[:, i+2, j+1] + orig[:, i+2, j+2]) / 9
#             print(p)
#             s = 0
#             for l in range(3):
#                 for m in range(3):
#                     s += p @ p
#             s /= 9
#             val, vec = np.linalg.eig(s)       
#             val, vec = np.real(val), np.real(vec)
#             norm_mat[:, i, j] = np.hstack((cr1, cr2))
#     return norm_mat


def prep_cross_aver_8(orig):
    norm_mat = np.zeros_like(orig)
    orig = np.pad(orig, pad_width=1)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]

            v5 = orig[:, i, j] - orig[:, i+1, j+1]
            v6 = orig[:, i, j+2] - orig[:, i+1, j+1]

            v7 = orig[:, i+2, j] - orig[:, i+1, j+1]
            v8 = orig[:, i+2, j+2] - orig[:, i+1, j+1]
            norm_mat[:, i, j] = np.mean((np.cross(v1, v2), np.cross(v3, v4), np.cross(v5, v6), np.cross(v7, v8)), axis = 0)
    return norm_mat


def prep_8_stacked(orig):
    norm_mat = np.zeros((12, 64, 1024))

    orig = np.pad(orig, pad_width=1)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]

            v5 = orig[:, i, j] - orig[:, i+1, j+1]
            v6 = orig[:, i, j+2] - orig[:, i+1, j+1]

            v7 = orig[:, i+2, j] - orig[:, i+1, j+1]
            v8 = orig[:, i+2, j+2] - orig[:, i+1, j+1]

            cr1 = np.cross(v1, v2)
            cr2 = np.cross(v3, v4)
            cr3 = np.cross(v5, v6)
            cr4= np.cross(v7, v8)
            norm_mat[:, i, j] = np.hstack((cr1, cr2, cr3, cr4))
    return norm_mat

def prep_cross_stacked(orig):
    norm_mat = np.zeros((6, 64, 1024))

    orig = np.pad(orig, pad_width=1)[1:4]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]
            cr1 = np.cross(v1, v2)
            cr2 = np.cross(v3, v4)
            norm_mat[:, i, j] = np.hstack((cr1, cr2))
    return norm_mat

def prep_long_short_stacked(orig):
    norm_mat = np.zeros((6, 64, 1024))

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i, j+2] - orig[:, i+2, j+2]
            v4 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr1 = np.cross(v1, v2)
            cr2 = np.cross(v3, v4)
            norm_mat[:, i, j] = np.hstack((cr1, cr2))
    return norm_mat

class KittiSimple(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.len = 4541
        self.label_folder = dir + "plane_labels/"
        self.road_label_folder = dir + "labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.open_road_label(self.road_label_folder  + '{0:06d}.label'.format(idx))
        self.ls.sem_label[(self.ls.sem_label != 0) | (np.isin(self.ls.sem_rd_label, (40, 48, 44, 49)))] = 1
        np.isin
        # self.ls.sem_label[self.ls.sem_rd_label == 40] = 1
        # self.ls.sem_label[self.ls.sem_rd_label == 48] = 1
        # self.ls.sem_label[self.ls.sem_rd_label == 44] = 1
        # self.ls.sem_label[self.ls.sem_rd_label == 49] = 1


        print(np.unique(self.ls.proj_sem_label))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_sem_label[self.ls.proj_sem_rd_label == 40] = 1
        print(np.unique(self.ls.proj_sem_rd_label))


        return self.ls.points, self.ls.sem_label, self.ls.proj_sem_label

class CarlaSimple(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        self.ls = SemLaserScan(project=True, nclasses=100, W=2048*32, fov_up=10.0, fov_down=-30.0)
        self.len = 2403
        self.dir = dir
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        self.ls.open_scan(self.dir + '{0:06d}.pcd'.format(idx))
        self.ls.open_label(self.dir + '{0:06d}.npy'.format(idx))
        self.ls.sem_label[self.ls.sem_label != 0] = 1
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        return self.ls.points, self.ls.sem_label, self.ls.proj_sem_label
    
class CustomCarla(Dataset):
    def __init__(self, dir, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100, H=32, W=1024, fov_up=45.0, fov_down=-30.0)
        self.mode = mode
        self.len = 2403
        self.train_len = 1800
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.dir = dir
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.dir + '{0:06d}.pcd'.format(idx))
        self.ls.open_label(self.dir + '{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.ls.proj_xyz, self.ls.proj_sem_label

class CustomCarlaProcessing(Dataset):
    def __init__(self, dir, proc_func, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100, H=32, W=1024, fov_up=45.0, fov_down=-30.0)
        self.mode = mode
        self.len = 2403
        self.train_len = 1800
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.dir = dir
        self.proc_func = proc_func
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.dir + '{0:06d}.pcd'.format(idx))
        self.ls.open_label(self.dir + '{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.proc_func(self.ls.proj_xyz), self.ls.proj_sem_label
    
class CustomKittiWithOrig(Dataset):
    def __init__(self, dir, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.ls.proj_xyz, self.ls.proj_sem_label, self.ls.points, self.ls.sem_label, self.ls.proj_idx


class CustomKitti(Dataset):
    def __init__(self, dir, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.ls.proj_xyz, self.ls.proj_sem_label
    
class CustomKittiProcessing(Dataset):
    def __init__(self, dir, proc_func, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
        self.proc_func = proc_func
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.proc_func(self.ls.proj_xyz), self.ls.proj_sem_label
    
class CustomKittiProcessingVis(Dataset):
    def __init__(self, dir, proc_func, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
        self.proc_func = proc_func
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.proc_func(self.ls.proj_xyz), self.ls.proj_sem_label, self.ls.points, self.ls.sem_label, self.ls.proj_idx
    

class CustomCarlaProcessingVis(Dataset):
    def __init__(self, dir, proc_func, mode="train"):
        print("kk")
        self.ls = SemLaserScan(project=True, nclasses=100, W=2048*6, fov_up=10.0, fov_down=-30.0)
        self.mode = mode
        self.len = 2403
        self.train_len = 1800
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.dir = dir
        self.proc_func = proc_func
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.dir + '{0:06d}.pcd'.format(idx))
        self.ls.open_label(self.dir + '{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.proc_func(self.ls.proj_xyz), self.ls.proj_sem_label, self.ls.points, self.ls.sem_label, self.ls.proj_idx

class CustomKittiPrep(Dataset):
    def __init__(self, path_d, path_l):
        self.data = np.load(path_d, allow_pickle=True)
        self.labels = np.load(path_l,  allow_pickle=True)

        print(self.data.shape)
        print(self.data[0].shape)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]