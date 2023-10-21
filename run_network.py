import argparse
import torch

from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import random_split

from loader.dataset import SphericalProjectionKitti
from network.config import Config
from network.common_blocks import get_model_and_optimizer, train, validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="runs_configs/exp1_basic.yaml",
        help="config file for the run",
    )
    args = parser.parse_args()

    config = Config(Path(args.config))
    config.prepare()

    data = SphericalProjectionKitti(config.dataset_dir, length=config.length)
    generator = torch.Generator().manual_seed(config.random_seed)
    train_loader, validation_loader, test_loader = random_split(
        data, [config.train_len, config.validation_len, config.test_len], generator=generator
    )

    train_loader = DataLoader(
        train_loader, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )
    validation_loader = DataLoader(
        validation_loader, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )
    test = DataLoader(
        test_loader, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )

    model, optimizer, scheduler = get_model_and_optimizer(
        config.device, in_ch=config.inp_channels, num_encoding_blocks=config.num_enc_blocks
    )

    train(
        config.device,
        train_loader,
        validation_loader,
        optimizer=optimizer,
        model=model,
        config=config,
    )

    validate(device=config.device, test_loader=test, model=model, config=config)
