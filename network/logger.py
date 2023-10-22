import wandb
import numpy as np
import torch.nn.functional as F
import torch


class ValidationLogger:
    def __init__(self, config):
        self.mean_IoU = 0
        self.mean_precision = 0
        self.mean_recall = 0
        self.mean_dice = 0
        self.mean_pixel_accuracy = 0
        self.mean_time = 0
        self.full_time = 0
        self.iterations = 0
        self.config = config

    def log_one_step(self, curr_time, batch_iou, precision, recall, dice, pixel_accuracy):
        if self.config.use_wandb:
            wandb.log({"test inference": curr_time})
            wandb.log({"batch IoU": batch_iou})
            wandb.log({"precision": precision})
            wandb.log({"recall": recall})
            wandb.log({"DICE": dice})
            wandb.log({"pixel accuracy": pixel_accuracy})
        else:
            print(
                f"""
                test inference: {curr_time}, batch IoU: {batch_iou}, precision: {precision} 
                recall: {recall}, DICE: {dice}, pixel accuracy: {pixel_accuracy}
                """
            )

        self.mean_IoU += batch_iou / self.config.batch_size
        self.full_time += curr_time
        self.mean_pixel_accuracy += pixel_accuracy
        self.mean_dice += dice
        self.mean_recall += recall
        self.mean_precision += precision
        self.iterations += 1

    def log_final(self):
        self.mean_IoU /= self.iterations
        self.mean_time = self.full_time / self.iterations
        self.mean_precision /= self.iterations
        self.mean_dice /= self.iterations
        self.mean_recall /= self.iterations
        self.mean_pixel_accuracy /= self.iterations
        if self.config.use_wandb:
            wandb.log({"mean inference": self.mean_time})
            wandb.log({"mean IoU": self.mean_IoU})
            wandb.log({"mean precision": self.mean_IoU})
            wandb.log({"mean recall": self.mean_recall})
            wandb.log({"mean DICE": self.mean_dice})
            wandb.log({"mean pixel accuracy": self.mean_pixel_accuracy})
            wandb.log({"total validation time": self.full_time})

        else:
            print(
                f"""
                mean inference: {self.mean_time}, mean IoU: {self.mean_IoU}  
                mean precision: {self.mean_recall}, mean recall: {self.mean_recall} 
                mean DICE: {self.mean_dice},  mean pixel accuracy: {self.mean_pixel_accuracy}
                total validation time: {self.full_time}
                """
            )


class TrainLogger:
    def __init__(self, config):
        self.inference_times = []
        self.config = config
        self.epoch = 0
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.starter.record()

    def log_epoch_start(self):
        self.epoch += 1
        print(f"EPOCH {self.epoch+1}:")

    def log_epoch_finish(
        self,
        train_loss,
        val_loss,
        best_train,
        best_val,
        inference_time,
        labels,
        prediction,
    ):
        if self.config.use_wandb:
            print(f"LOSS train: {train_loss}, validation: {val_loss}")
            wandb.log({"best train accuracy": best_train})
            wandb.log({"best test accuracy": best_val})

            wandb.log({"current train accuracy": train_loss})
            wandb.log({"current test accuracy": val_loss})

            wandb.log({"current inference": inference_time})
        else:
            print(
                f"""
                CURRENT ACCURACY train: {train_loss}, validation: {val_loss}  
                BEST ACCURACY train: {best_loss}, validation: {best_val}  
                inference time: {inference_time}
                """
            )
        self.inference_times.append(inference_time)

        gt = labels[0].cpu().detach().numpy().astype("int")
        prediction_prepared = (
            F.softmax(prediction, dim=1)[0]
            .cpu()
            .permute(1, 2, 0)
            .detach()
            .numpy()
            .squeeze()[:, :, 1]
            .round()
            .astype("int")
        )
        if self.config.use_wandb:
            wandb.log(
                {"val view": wandb.Image(prediction_prepared, caption="Validation prediction")}
            )
            wandb.log({"val gt": wandb.Image(gt, caption="Validation gt")})

    def log_final(self):
        self.ender.record()
        torch.cuda.synchronize()
        total_time = self.starter.elapsed_time(self.ender)
        if self.config.use_wandb:
            wandb.log({"mean inference": np.mean(np.asarray(self.inference_times))})
            wandb.log({"total train time": np.mean(np.asarray(total_time))})
        else:
            print(
                f"""
               mean inference time: {np.mean(np.asarray(self.inference_times))}
               total time: {total_time}
               """
            )
