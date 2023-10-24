import open3d as o3d
import numpy as np
import argparse
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from loader.dataset import SphericalProjectionKitti
from network.common_blocks import get_model_and_optimizer
from network.config import Config


def enumerate_with_step(xs, start=0, step=5):
    for x in xs:
        yield start, x
        start += step


def visualize_predictions(loader, config, chpt_path, iterations=1, step=5):
    model, _, _ = get_model_and_optimizer(config)
    model.load_state_dict(torch.load(chpt_path, map_location=torch.device(config.device)))
    model.train(False)
    model.to(config.device)
    for i, sample in enumerate_with_step(loader, start=0, step=step):
        if i >= iterations:
            return
        inputs, labels, pcl, pcl_labels, idx = sample
        idx = idx.squeeze()
        points = pcl.detach().numpy().squeeze()
        labels = labels.squeeze()

        colors = np.zeros_like(points)
        pcl_labels[pcl_labels > 0] = 1
        pcl_labels = pcl_labels.detach().numpy().squeeze()
        colors[:, 0] = 255 * pcl_labels

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # pointcloud with GT labels
        o3d.visualization.draw_geometries([pcd])

        inputs = torch.tensor(inputs).to(device=config.device, dtype=torch.float)
        # Getting predictions
        outputs = model(inputs)

        pcd_with_predictions = o3d.geometry.PointCloud()
        mask = idx >= 0
        points_back_proj = points[idx[mask]]
        colors_back_proj = np.zeros_like(points_back_proj)
        colors_back_proj[:, 1] = 255 * labels[mask].detach().numpy().squeeze()

        outputs = (
            F.softmax(outputs, dim=1)[0].cpu().permute(1, 2, 0).detach().numpy().squeeze()[:, :, 1]
        )
        colors_back_proj[:, 2] = 255 * outputs[mask]

        pcd_with_predictions.points = o3d.utility.Vector3dVector(points_back_proj)
        pcd_with_predictions.colors = o3d.utility.Vector3dVector(colors_back_proj)
        o3d.visualization.draw_geometries([pcd_with_predictions])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chpt",
        type=str,
        help="path to the checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="config file for the run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to the dataset",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="how much samples you wish to visualise",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="step for loader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="the device to run network, cpu is default",
    )

    args = parser.parse_args()
    config = Config(Path(args.config))

    config.device = args.device
    data = SphericalProjectionKitti(
        Path(args.dataset), length=config.length, return_orig_points=True
    )
    generator = torch.Generator().manual_seed(config.random_seed)
    _, _, test_data = random_split(
        data, [config.train_len, config.validation_len, config.test_len], generator=generator
    )

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    visualize_predictions(
        loader=test_loader,
        config=config,
        chpt_path=args.chpt,
        iterations=args.iterations,
        step=args.step,
    )
