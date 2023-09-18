import open3d as o3d
import numpy as np
import argparse

from torch.utils.data import DataLoader
from pathlib import Path

from loader.dataloader import SphericalProjectionKitti


def visualise(loader, show_projected_labels=False):
    for data in loader:
        points, _, labels, proj_labels, proj_idx = data
        points = points.detach().numpy().squeeze()
        labels = labels.detach().numpy().squeeze()
        proj_labels = proj_labels.detach().numpy().squeeze()
        proj_idx = proj_idx.detach().numpy().squeeze()

        pcd = o3d.geometry.PointCloud()
        colors = np.zeros_like(points)
        colors[:, 0] = 255 * labels
        if show_projected_labels:
            colors[proj_idx, 1] = 255 * proj_labels

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="examples/data/kitti/00/",
                    help="path to the dataset main directory")
    parser.add_argument("--dataset_len", type=int, default=1, help="length of the dataset sequence")
    args = parser.parse_args()

    data = SphericalProjectionKitti(Path(args.dataset), length=args.dataset_len)
    loader = DataLoader(data, batch_size=1, shuffle=False)
    visualise(loader=loader, show_projected_labels=False)
