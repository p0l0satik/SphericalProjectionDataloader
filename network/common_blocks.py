import torch
import numpy as np
import torch.nn.functional as F

from torchmetrics import JaccardIndex
from tqdm import tqdm
from unet import UNet

from network.logger import ValidationLogger, TrainLogger
from network.metrics import SegmentationMetrics
from network.loss_functions import dice_loss


def get_model_and_optimizer(device, in_ch=3, num_encoding_blocks=5, patience=3):
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    unet = UNet(
        in_channels=in_ch,
        out_classes=2,
        dimensions=2,
        num_encoding_blocks=num_encoding_blocks,
        normalization="batch",
        upsampling_type="linear",
        padding=True,
        activation="ReLU",
    ).to(device)
    model = unet

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=patience, threshold=0.01
    )
    return model, optimizer, scheduler


def one_epoch(device, loader, optimizer, model, criterion=None, train_mode=True):
    running_loss = 0.0
    max_classes = 2  # TODO move to config
    model.train(train_mode)
    dataset_len = len(loader)
    for data in tqdm(loader):
        inputs, labels = data
        inputs = torch.tensor(inputs).to(device=device, dtype=torch.float)
        labels = torch.tensor(labels).to(device=device, dtype=torch.long)

        if train_mode:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels) + dice_loss(
            F.softmax(outputs, dim=1).float(),
            F.one_hot(labels, max_classes).permute(0, 3, 1, 2).float(),
            multiclass=True,
        )
        if train_mode:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    # calculating mean loss for the epoch
    loss = running_loss / dataset_len
    return loss


def validate(device, test_loader, model, config, chpt=""):
    if chpt != "":
        model.load_state_dict(torch.load(chpt))

    model.train(False)
    model.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    jaccard = JaccardIndex(num_classes=2)
    validation_logger = ValidationLogger(config)

    metric_calculator = SegmentationMetrics(
        average=True, ignore_background=True, activation="softmax"
    )
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            val_inputs, val_labels = data
            val_inputs = torch.tensor(val_inputs).to(device=device, dtype=torch.float)
            val_labels = torch.tensor(val_labels).to(device=device, dtype=torch.long)

            # Make predictions for this batch
            starter.record()
            val_outputs = model(val_inputs)
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


def train(
    device,
    tl,
    vl,
    optimizer,
    model,
    config,
):
    (
        best_val,
        best_train,
    ) = (
        1,
        1,
    )

    train_logger = TrainLogger(config)
    for epoch in range(config.n_epochs):
        train_logger.log_epoch_start()
        # running train
        model.train(True)
        train_loss = one_epoch(
            device,
            tl,
            model=model,
            criterion=config.criterion,
            optimizer=optimizer,
            train_mode=True,
        )
        # running validation
        with torch.no_grad():
            val_loss = one_epoch(
                device,
                vl,
                model=model,
                criterion=config.criterion,
                optimizer=optimizer,
                train_mode=False,
            )

        # comparing results with absolute values
        if val_loss < best_val:
            best_val = val_loss
            model_path = "{}/{}_{}".format(config.path, config.run_name, epoch)
            torch.save(model.state_dict(), model_path)
        if train_loss < best_train:
            best_train = train_loss

        # Calculating inference time and getting sample prediction
        sample_input, sample_labels = next(iter(vl))
        sample_input = torch.tensor(sample_input).to(device=device, dtype=torch.float)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        sample_predictions = model(sample_input)
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
