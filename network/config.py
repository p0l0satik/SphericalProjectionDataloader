import wandb
import yaml
import torch

from datetime import datetime
from torch import nn
from pathlib import Path


class Config:
    def __init__(self, config_path) -> None:
        with open(config_path, "r") as file:
            config_file = yaml.safe_load(file)

        self.run_name = config_file["run_name"]
        self.description = config_file["description"]
        self.dataset_dir = Path(config_file["dataset_dir"])
        self.checkpoint_dir = Path(config_file["checkpoint_save_path"])
        self.use_wandb = config_file["use_wandb"]
        self.wandb_proj = config_file["wandb_project"]

        # network parameters
        self.n_epochs = config_file["parameters"]["epochs"]
        self.num_enc_blocks = config_file["parameters"]["num_encoder_blocks"]
        self.inp_channels = config_file["parameters"]["input_channels"]
        self.device = config_file["parameters"]["device"]
        self.max_classes = config_file["parameters"]["max_classes"]

        # loss
        if config_file["parameters"]["criterion_type"] == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise RuntimeError("No match for criterion type")

        # optimizer parameters
        if config_file["parameters"]["optimizer_type"] == "AdamW":
            self.optimizer = torch.optim.AdamW
        else:
            raise RuntimeError("No match for optimizer type")

        # scheduler parameters
        self.factor = config_file["parameters"]["scheduler"]["factor"]
        self.patience = config_file["parameters"]["scheduler"]["patience"]
        self.threshold = config_file["parameters"]["scheduler"]["threshold"]

        # loader parameters
        self.batch_size = config_file["loader"]["batch_size"]
        self.workers = config_file["loader"]["workers"]
        self.dataset = config_file["loader"]["dataset"]
        self.length = config_file["loader"]["length"]
        self.train_len = config_file["loader"]["train_len"]
        self.test_len = config_file["loader"]["test_len"]
        self.validation_len = config_file["loader"]["validation_len"]

        self.random_seed = config_file["other"]["random_seed"]
        self.curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.use_wandb:
            self.wandb_config = dict(
                # network params
                n_epochs=self.n_epochs,
                num_enc_blocks=self.num_enc_blocks,
                inp_channels=self.inp_channels,
                max_classes=self.max_classes,
                device=self.device,
                criterion=config_file["parameters"]["criterion_type"],
                optimiser=config_file["parameters"]["optimizer_type"],
                # scheduler parameters
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                # dataset params
                workers=self.workers,
                batch_size=self.batch_size,
                dataset=self.dataset,
                train_len=self.train_len,
                test_len=self.test_len,
                validation_len=self.validation_len,
                length=self.length,
            )

    def prepare(self):
        self.checkpoint_dir = self.checkpoint_dir / f"{self.run_name}_{self.curr_time}"
        # create a directory for checkpoints if not exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            wandb.init(
                project=self.wandb_proj,
                notes=self.description,
                config=self.wandb_config,
                mode="online",
            )

            name = self.run_name + "_" + self.curr_time
            wandb.run.name = name
