import numpy as np
import argparse

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from preparation.preprocessor import SphericalProjectionKittiPreprocessor


def preprocess(loader, save_path, compress=False):
    for n, scan in enumerate(tqdm(loader)):
        # prepare data
        points, proj_points, labels, proj_labels, proj_idx = scan
        points = points.detach().numpy().squeeze()
        labels = labels.detach().numpy().squeeze()
        proj_labels = proj_labels.detach().numpy().squeeze()
        proj_idx = proj_idx.detach().numpy().squeeze()
        proj_points = proj_points.detach().numpy().squeeze()

        # saving data
        if compress:
            np.savez_compressed(
                save_path / "scan_{0:06d}.npz".format(n),
                points=points,
                labels=labels,
                proj_points=proj_points,
                proj_labels=proj_labels,
                proj_idx=proj_idx,
            )
        else:
            np.savez(
                save_path / "scan_{0:06d}.npz".format(n),
                points=points,
                labels=labels,
                proj_points=proj_points,
                proj_labels=proj_labels,
                proj_idx=proj_idx,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="examples/data/dataset_ready/",
        help="path where to set preprocessed dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="examples/data/kitti/00/",
        help="path to the dataset main directory",
    )
    parser.add_argument(
        "--dataset_len", type=int, default=1, help="length of the dataset sequence"
    )

    parser.add_argument(
        "--use_ransac", type=bool, default=True, help="enables ransac preprocessing"
    )

    args = parser.parse_args()

    dataset = SphericalProjectionKittiPreprocessor(
        Path(args.dataset),
        length=args.dataset_len,
        use_ransac=args.use_ransac,
        visualise_ransac=False,
    )
    prep_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preprocess(loader=prep_loader, save_path=Path(args.save_path), compress=False)
