import wandb
import os 

from datetime import datetime
from torch import nn

class Config:
    def __init__(self) -> None:
        self.criterion = nn.CrossEntropyLoss()
        self.n_epochs = 15
        self.run_name = ""
        self.description = ""
        self.batch_size = 4
        self.num_enc_blocks = 5
        self.inp_channels = 3
        self.dataset = "kitti"
        self.curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def prepare(self):
        self.path = f"new_2_chpt/{self.run_name}_{self.curr_time}/"

        isExist = os.path.exists(self.path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.path)

        wandb_config = dict(
            batch_size=self.batch_size,
            num_blocks=self.num_enc_blocks,
            inp_channels=self.inp_channels,
            dataset=self.dataset,
            n_epochs=self.n_epochs,
        )

        wandb.init(
            project="PlaneSegmentationImproved",
            notes=self.description,
            config=wandb_config,
            mode="online",
        )

        name = self.run_name + "_" + self.curr_time
        wandb.run.name = name
