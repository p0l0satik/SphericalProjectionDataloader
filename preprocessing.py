import numpy as np
import argparse

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from preparation.preprocessor import SphericalProjectionKittiPreprocessor


def preprocess(loader, save_path, compress=False):
    save_path.mkdir(parents=True, exist_ok=True)
    for n, scan in enumerate(tqdm(loader)):
        # prepare data
        points, proj_points, labels, proj_labels, proj_idx = scan
        points = points.detach().numpy().squeeze()
        labels = labels.detach().numpy().squeeze()
        proj_labels = proj_labels.detach().numpy().squeeze()
        proj_idx = proj_idx.detach().numpy().squeeze()
        proj_points = proj_points.detach().numpy().squeeze()

        # saving data
        save_func = np.savez_compressed if compress else np.savez
        save_func(
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
        "--no_ransac",
        dest="no_ransac",
        action="store_true",
        help="enables ransac filtering of outliers for plane labels; ransac is enabled by default",
    )

    parser.add_argument(
        "--visualise_ransac",
        dest="visualise_ransac",
        action="store_true",
        help="visualises ransac preprocessing; disabled by default",
    )

    parser.add_argument(
        "--no_road",
        dest="no_road",
        action="store_true",
        help="diables adding road labels; road is enabled by default",
    )

    parser.set_defaults(use_ransac=False, add_road=False, visualise_ransac=False)

    args = parser.parse_args()

    dataset = SphericalProjectionKittiPreprocessor(
        Path(args.dataset),
        length=args.dataset_len,
        use_ransac=not args.no_ransac,
        add_road=not args.no_road,
        visualise_ransac=args.visualise_ransac,
    )
    prep_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    preprocess(loader=prep_loader, save_path=Path(args.save_path), compress=False)
