from dataclasses import asdict
import subprocess
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sn
import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

import cv2
import numpy as np
import pandas as pd


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"])
                .strip()
                .decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(
                        str(git_dir_path)
                    )  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(
                            ["git", "check-ignore", "-q", str(path)]
                        ).returncode
                        == 1
                    )
                ):
                    code.add_file(
                        str(path), name=str(path.relative_to(self.code_dir))
                    )

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(
                    str(path), name=str(path.relative_to(self.code_dir))
                )

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(
        self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False
    ):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class LogConfusionMatrix(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, log_epoch: int = 1):
        self.preds = []
        self.targets = []
        self.ready = True
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            confusion_matrix = metrics.confusion_matrix(
                y_true=targets, y_pred=preds
            )

            # set figure size
            plt.figure(figsize=(14, 8))

            # set labels size
            sn.set(font_scale=1.4)

            # set font size
            sn.heatmap(
                confusion_matrix, annot=True, annot_kws={"size": 8}, fmt="g"
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"confusion_matrix/{experiment.name}": wandb.Image(plt)},
                commit=False,
            )

            # according to wandb docs this should also work but it crashes
            # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogConfusionMatrix_PML(Callback):
    """Generate confusion matrix every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, log_epoch: int = 1):
        self.probs = []
        self.preds = []
        self.targets = []
        self.ready = True
        self.log_epoch = log_epoch

        self.class_names = [
            "normal",
            "active tb",
            "inactive tb",
            "pneumonia",
            "sick",
        ]

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.probs.append(outputs["probs"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        """Generate confusion matrix."""
        if self.ready:
            logger = get_wandb_logger(trainer)
            experiment = logger.experiment

            probs = torch.cat(self.probs).cpu().numpy()
            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()

            assert preds.shape == targets.shape
            idx = np.where(targets == -1)
            targets[idx] = preds[idx]

            confusion_matrix = metrics.multilabel_confusion_matrix(
                y_true=targets, y_pred=preds
            )

            for name, cm in zip(self.class_names, confusion_matrix):
                # set figure size
                plt.figure(figsize=(14, 8))
                # set labels size
                sn.set(font_scale=1.4)
                # set font size
                sn.heatmap(cm, annot=True, annot_kws={"size": 8}, fmt="g")

                # names should be uniqe or else charts from different experiments in wandb will overlap
                experiment.log(
                    {
                        f"confusion_matrix_{name}/{experiment.name}": wandb.Image(
                            plt
                        )
                    },
                    commit=False,
                )

                # according to wandb docs this should also work but it crashes
                # experiment.log(f{"confusion_matrix/{experiment.name}": plt})

                # reset plot
                plt.clf()

            for i, name in zip(np.arange(5), self.class_names):
                cls_probs = probs[:, i]
                df = pd.DataFrame(cls_probs, columns=[f"val_prob"])
                df["targets"] = targets[:, i]
                idx = df[df.targets == -1].index
                numbers = [float(x) / 10 for x in range(10)]

                cutoff_df = pd.DataFrame(
                    columns=[
                        "Probability",
                        "Accuracy",
                        "Sensitivity",
                        "Specificity",
                    ]
                )
                for j in numbers:
                    df[j] = df.val_prob.map(lambda x: 1 if x > j else 0)

                    df.loc[idx, "targets"] = df.loc[idx, j]
                    cm1 = metrics.confusion_matrix(df.targets, df[j])
                    total1 = sum(sum(cm1))
                    Accuracy = (cm1[0, 0] + cm1[1, 1]) / total1
                    Specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
                    Sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                    cutoff_df.loc[j] = [j, Accuracy, Sensitivity, Specificity]
                cutoff_df.to_csv(
                    f"{pl_module.hparams.save_dir}/val_{name}_cutoff.csv",
                    index=False,
                )
                df.loc[idx, "targets"] = -1
                df.to_csv(
                    f"{pl_module.hparams.save_dir}/val_{name}_probs.csv",
                    index=False,
                )

            self.preds.clear()
            self.targets.clear()
            self.probs.clear()


class LogF1PrecRecHeatmap(Callback):
    """Generate f1, precision, recall heatmap every epoch and send it to wandb.
    Expects validation step to return predictions and targets.
    """

    def __init__(self, class_names: List[str] = None, log_epoch: int = 1):
        self.preds = []
        self.targets = []
        self.ready = True
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Gather data from single batch."""
        if self.ready:
            self.preds.append(outputs["preds"])
            self.targets.append(outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        """Generate f1, precision and recall heatmap."""
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            preds = torch.cat(self.preds).cpu().numpy()
            targets = torch.cat(self.targets).cpu().numpy()
            f1 = f1_score(preds, targets, average=None)
            r = recall_score(preds, targets, average=None)
            p = precision_score(preds, targets, average=None)
            data = [f1, p, r]

            # set figure size
            plt.figure(figsize=(14, 3))

            # set labels size
            sn.set(font_scale=1.2)

            # set font size
            sn.heatmap(
                data,
                annot=True,
                annot_kws={"size": 10},
                fmt=".3f",
                yticklabels=["F1", "Precision", "Recall"],
            )

            # names should be uniqe or else charts from different experiments in wandb will overlap
            experiment.log(
                {f"f1_p_r_heatmap/{experiment.name}": wandb.Image(plt)},
                commit=False,
            )

            # reset plot
            plt.clf()

            self.preds.clear()
            self.targets.clear()


class LogImagePredictions(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, log_epoch: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a train batch from the train dat loader
            train_samples = next(iter(trainer.datamodule.train_dataloader()))
            train_imgs, train_labels, path, labels = train_samples

            # run the batch through the network
            train_imgs = train_imgs.to(device=pl_module.device)
            logits = pl_module(train_imgs)
            train_imgs = train_imgs.detach().cpu().numpy()
            train_imgs = np.transpose(train_imgs, (0, 2, 3, 1))
            train_labels = train_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(logits).squeeze().detach().cpu().numpy()
            )

            log_img_arr = []
            for x, pred, y in zip(
                train_imgs[: self.num_samples],
                preds[: self.num_samples],
                train_labels[: self.num_samples],
            ):
                y = (y * 255).astype(np.uint8)
                pred = (pred * 255).astype(np.uint8)
                pred = cv2.resize(
                    pred, dsize=y.shape, interpolation=cv2.INTER_LANCZOS4
                )
                log_img_arr.append(
                    wandb.Image(x, caption="Lung Segmentation Image")
                )
                log_img_arr.append(
                    wandb.Image(pred, caption="Lung Segmentation Image")
                )
                log_img_arr.append(
                    wandb.Image(y, caption="Lung Segmentation Image")
                )

            # log the images as wandb Image
            experiment.log({f"Train_Images/{experiment.name}": log_img_arr})

            # # log the images as wandb Image
            # experiment.log(
            #     {
            #         f"Train_Images/{experiment.name}": [
            #             wandb.Image(x, caption=f"Lung Segmentation Results")
            #             for x, pred, y in zip(
            #                 train_imgs[: self.num_samples],
            #                 preds[: self.num_samples],
            #                 train_labels[: self.num_samples],
            #             )
            #         ]
            #     }
            # )

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels, path, label = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits = pl_module(val_imgs)
            val_imgs = val_imgs.detach().cpu().numpy()
            val_imgs = np.transpose(val_imgs, (0, 2, 3, 1))
            val_labels = val_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(logits).squeeze().detach().cpu().numpy()
            )

            log_img_arr = []
            for x, pred, y in zip(
                val_imgs[: self.num_samples],
                preds[: self.num_samples],
                val_labels[: self.num_samples],
            ):
                y = (y * 255).astype(np.uint8)
                pred = (pred * 255).astype(np.uint8)
                pred = cv2.resize(
                    pred, dsize=y.shape, interpolation=cv2.INTER_LANCZOS4
                )
                log_img_arr.append(
                    wandb.Image(x, caption="Lung Segmentation Image")
                )
                log_img_arr.append(
                    wandb.Image(pred, caption="Lung Segmentation Image")
                )
                log_img_arr.append(
                    wandb.Image(y, caption="Lung Segmentation Image")
                )

            # log the images as wandb Image
            experiment.log({f"Val_Images/{experiment.name}": log_img_arr})
            # experiment.log(
            #     {
            #         f"Val_Images/{experiment.name}": [
            #             wandb.Image(x, caption="Lung Segmentation Image"),
            #             for x, pred, y in zip(
            #                 val_imgs[: self.num_samples],
            #                 preds[: self.num_samples],
            #                 val_labels[: self.num_samples],
            #             )
            #         ]
            #     }
            # )


class LogImagePredictions_Multiview(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, log_epoch: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.view_index_to_string = [
            "ap",
            "lat",
            "ap_latcon",
            "ap_trochlea",
            "ap_proximal",
            "lat_olecranon",
            "lat_proximal",
        ]
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            train_samples = next(iter(trainer.datamodule.train_dataloader()))
            train_imgs, train_labels, view_index = train_samples

            # run the batch through the network
            train_imgs = train_imgs.to(device=pl_module.device)
            view_index = view_index.to(device=pl_module.device)

            logits = pl_module(train_imgs, view_index)
            preds = (
                pl_module.get_predict(logits)
                .view(-1, train_labels.shape[1])
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            view_index = view_index.detach().cpu().numpy()
            # preds = torch.argmax(logits, axis=-1)

            log_img_arr = []
            for x, pred, y, vidx in zip(
                train_imgs[: self.num_samples],
                preds[: self.num_samples],
                train_labels[: self.num_samples],
                view_index[: self.num_samples],
            ):
                for i in range(len(vidx)):
                    log_img_arr.append(
                        wandb.Image(
                            x[i],
                            caption=f"Pred:{pred[i]}, Label:{y[i]}, View: {self.view_index_to_string[vidx[i]]}",
                        )
                    )

            # log the images as wandb Image
            experiment.log({f"Train_Images/{experiment.name}": log_img_arr})

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels, view_index = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            view_index = view_index.to(device=pl_module.device)

            logits = pl_module(val_imgs, view_index)
            preds = (
                pl_module.get_predict(logits)
                .view(-1, val_labels.shape[1])
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            view_index = view_index.detach().cpu().numpy()
            # preds = torch.argmax(logits, axis=-1)

            log_img_arr = []
            for x, pred, y, vidx in zip(
                val_imgs[: self.num_samples],
                preds[: self.num_samples],
                val_labels[: self.num_samples],
                view_index[: self.num_samples],
            ):
                for i in range(len(vidx)):
                    log_img_arr.append(
                        wandb.Image(
                            x[i],
                            caption=f"Pred:{pred[i]}, Label:{y[i]}, View: {self.view_index_to_string[vidx[i]]}",
                        )
                    )
            # log the images as wandb Image
            experiment.log({f"Val_Images/{experiment.name}": log_img_arr})


class LogImagePredictions_TB_PNA(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, log_epoch: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        classname = ["Others", "Active TB", "Pneumonia"]
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a train batch from the train dat loader
            train_samples = next(iter(trainer.datamodule.train_dataloader()))
            train_imgs, train_labels, centers = train_samples

            # run the batch through the network
            train_imgs = train_imgs.to(device=pl_module.device)
            logits, features = pl_module(train_imgs)
            train_imgs = train_imgs.squeeze().detach().cpu().numpy()
            train_imgs = np.transpose(train_imgs, (0, 2, 3, 1))
            train_labels = train_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(logits).squeeze().detach().cpu().numpy()
            )

            # # log the images as wandb Image
            experiment.log(
                {
                    f"Train_Images/{experiment.name}": [
                        wandb.Image(
                            x,
                            caption=f"Pred:{classname[pred]}, Label:{classname[y]}",
                        )
                        for x, pred, y in zip(
                            train_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            train_labels[: self.num_samples],
                        )
                    ]
                }
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        classname = ["Others", "Active TB", "Pneumonia"]
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels, centers = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits, features = pl_module(val_imgs)
            val_imgs = val_imgs.squeeze().detach().cpu().numpy()
            val_imgs = np.transpose(val_imgs, (0, 2, 3, 1))
            val_labels = val_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(logits).squeeze().detach().cpu().numpy()
            )

            # # log the images as wandb Image
            experiment.log(
                {
                    f"Val_Images/{experiment.name}": [
                        wandb.Image(
                            x,
                            caption=f"Pred:{classname[pred]}, Label:{classname[y]}",
                        )
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )


class LogImagePredictions_TB_PNA_PML(Callback):
    """Logs a validation batch and their predictions to wandb.
    Example adapted from:
        https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    """

    def __init__(self, num_samples: int = 8, log_epoch: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.ready = True
        self.log_epoch = log_epoch

    def on_sanity_check_start(self, trainer, pl_module):
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        classname = pl_module.class_names
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a train batch from the train dat loader
            train_samples = next(iter(trainer.datamodule.train_dataloader()))
            train_imgs, train_labels, centers = train_samples

            # run the batch through the network
            train_imgs = train_imgs.to(device=pl_module.device)
            logits, features = pl_module(train_imgs)
            probs = torch.sigmoid(logits)
            train_imgs = train_imgs.squeeze().detach().cpu().numpy()
            train_imgs = np.transpose(train_imgs, (0, 2, 3, 1))
            train_labels = train_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(probs).squeeze().detach().cpu().numpy()
            )

            # # log the images as wandb Image
            experiment.log(
                {
                    f"Train_Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            train_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            train_labels[: self.num_samples],
                        )
                    ]
                }
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            pl_module.current_epoch != 0
            and self.log_epoch % pl_module.current_epoch == 0
        ):
            return

        classname = pl_module.class_names
        if self.ready:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            # get a validation batch from the validation dat loader
            val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_imgs, val_labels, centers = val_samples

            # run the batch through the network
            val_imgs = val_imgs.to(device=pl_module.device)
            logits, features = pl_module(val_imgs)
            probs = torch.sigmoid(logits)
            val_imgs = val_imgs.squeeze().detach().cpu().numpy()
            val_imgs = np.transpose(val_imgs, (0, 2, 3, 1))
            val_labels = val_labels.squeeze().detach().cpu().numpy()
            preds = (
                pl_module.get_predict(probs).squeeze().detach().cpu().numpy()
            )

            # # log the images as wandb Image
            experiment.log(
                {
                    f"Val_Images/{experiment.name}": [
                        wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                        for x, pred, y in zip(
                            val_imgs[: self.num_samples],
                            preds[: self.num_samples],
                            val_labels[: self.num_samples],
                        )
                    ]
                }
            )
