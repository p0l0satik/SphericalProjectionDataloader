import torch

from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import random_split

from loader.dataset import SphericalProjectionKitti
from network.config import Config
from network.common_blocks import get_model_and_optimizer, train, validate


class RunConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.run_name = "exp_1_basic_unet"
        self.description = "Basic unet"
        self.n_epochs = 15


if __name__ == "__main__":
    config = RunConfig()
    config.prepare()

    device = "cuda:0"

    data = SphericalProjectionKitti(Path("/home/polosatik/mnt/kitti/prep2"), length=4541)
    generator = torch.Generator().manual_seed(42)
    train_loader, validation_loader, test_loader = random_split(
        data, [3700, 541, 300], generator=generator
    )

    train_loader = DataLoader(train_loader, batch_size=4, shuffle=True, num_workers=12)
    validation_loader = DataLoader(validation_loader, batch_size=4, shuffle=True, num_workers=12)
    test = DataLoader(test_loader, batch_size=4, shuffle=True, num_workers=12)

    model, optimizer, scheduler = get_model_and_optimizer(
        device, in_ch=config.inp_channels, num_encoding_blocks=config.num_enc_blocks
    )

    train(device, train_loader, validation_loader, optimizer=optimizer, model=model, config=config)

    validate(device=device, test_loader=test, model=model, config=config)
