import torch
import numpy as np
import torch.nn.functional as F

from torchmetrics import JaccardIndex
from tqdm import tqdm
from unet import UNet

from network.logger import ValidationLogger, TrainLogger
from network.metrics import SegmentationMetrics
from network.loss_functions import dice_loss
from network.config import Config


class Network:
    def __init__(self, config: Config):
        self.config = config

        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model = UNet(
            in_channels=config.inp_channels,
            out_classes=config.max_classes,
            dimensions=2,
            num_encoding_blocks=config.num_enc_blocks,
            normalization="batch",
            upsampling_type="linear",
            padding=True,
            activation="ReLU",
        ).to(config.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
        )

    def run_one_epoch(self, loader, train_mode=True):
        running_loss = 0.0
        self.model.train(train_mode)
        dataset_len = len(loader)
        for data in tqdm(loader):
            inputs, labels = data
            inputs = torch.tensor(inputs).to(device=self.config.device, dtype=torch.float)
            labels = torch.tensor(labels).to(device=self.config.device, dtype=torch.long)

            if train_mode:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.config.criterion(outputs, labels) + dice_loss(
                F.softmax(outputs, dim=1).float(),
                F.one_hot(labels, self.config.max_classes).permute(0, 3, 1, 2).float(),
                multiclass=True,
            )
            if train_mode:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        # calculating mean loss for the epoch
        loss = running_loss / dataset_len
        return loss

    def validate(self, test_loader, chpt=""):
        if chpt != "":
            self.model.load_state_dict(torch.load(chpt))

        self.model.train(False)
        self.model.to(self.config.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        jaccard = JaccardIndex(num_classes=self.config.max_classes)
        validation_logger = ValidationLogger(self.config)

        metric_calculator = SegmentationMetrics(
            average=True, ignore_background=True, activation="softmax"
        )
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                val_inputs, val_labels = data
                val_inputs = torch.tensor(val_inputs).to(
                    device=self.config.device, dtype=torch.float
                )
                val_labels = torch.tensor(val_labels).to(
                    device=self.config.device, dtype=torch.long
                )

                # Make predictions for this batch
                starter.record()
                val_outputs = self.model(val_inputs)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)

                # calculating metrics
                batch_iou = 0
                for j in range(len(val_outputs)):
                    prediction = F.softmax(val_outputs, dim=1)[j].permute(1, 2, 0)[:, :, 1]
                    batch_iou += jaccard(prediction.cpu(), val_labels[j].cpu().int())

                pixel_accuracy, dice, precision, recall = metric_calculator(
                    val_labels.int(), val_outputs
                )

                validation_logger.log_one_step(
                    curr_time, batch_iou, precision, recall, dice, pixel_accuracy
                )
        validation_logger.log_final()

    def train(self, train_loader, validation_loader):
        (
            best_val,
            best_train,
        ) = (
            1,
            1,
        )

        train_logger = TrainLogger(self.config)
        for epoch in range(self.config.n_epochs):
            train_logger.log_epoch_start()
            # running train
            self.model.train(True)
            train_loss = self.run_one_epoch(
                train_loader,
                train_mode=True,
            )
            # running validation
            with torch.no_grad():
                val_loss = self.run_one_epoch(
                    validation_loader,
                    train_mode=False,
                )

            # comparing results with absolute values
            if val_loss < best_val:
                best_val = val_loss
                model_path = self.config.checkpoint_dir / f"{self.config.run_name}_{epoch}"
                torch.save(self.model.state_dict(), model_path)
            if train_loss < best_train:
                best_train = train_loss

            # Calculating inference time and getting sample prediction
            sample_input, sample_labels = next(iter(validation_loader))
            sample_input = torch.tensor(sample_input).to(
                device=self.config.device, dtype=torch.float
            )

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            starter.record()
            sample_predictions = self.model(sample_input)
            ender.record()
            torch.cuda.synchronize()

            #  calculating inference time
            inference_time = starter.elapsed_time(ender)

            train_logger.log_epoch_finish(
                train_loss,
                val_loss,
                best_train,
                best_val,
                inference_time,
                sample_labels,
                sample_predictions,
            )
        train_logger.log_final()
