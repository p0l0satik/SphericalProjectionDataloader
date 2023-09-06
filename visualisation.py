import open3d as o3d
import numpy as np
from torch.utils.data import  DataLoader
from loader.dataloader import *
import time
def visualise(loader, show_projected_labels = False):
    for data in loader:
        points, _, labels, proj_labels, proj_idx = data
        points = points.detach().numpy().squeeze()
        labels = labels.detach().numpy().squeeze()
        proj_labels = proj_labels.detach().numpy().squeeze()
        proj_idx = proj_idx.detach().numpy().squeeze()

        pcd = o3d.geometry.PointCloud()
        colors = np.zeros_like(points)
        colors[:,0] = 255*labels
        if show_projected_labels:
            colors[proj_idx,1] = 255*proj_labels

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
        break

data = KittiSimple("examples/data/kitti/00/", len=1)
loader = DataLoader(data, batch_size=1, shuffle=False)
visualise(loader=loader, show_projected_labels=False)