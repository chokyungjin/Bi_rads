from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..metric import get_metric_ACC
from .utils import add_prefix_to_dict, key_to_abbreviation


class CXR_Image_resnet(LightningModule):
    
    def __init__(
        self,
        model: nn.Module,
        num_class: int,
        criterion: list,
        disease_lambda: float = 1,
        matching_lambda: float = 1, 
        optimizer=None,
        scheduler=None,
        threshold_cls: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.threshold_cls = threshold_cls
        self.model = model
        self.num_class = num_class
        self.disease_lambda = disease_lambda
        self.matching_lambda = matching_lambda
        if criterion[0] == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        if criterion[1] == 'BCE':
            self.disease_criterion = nn.BCEWithLogitsLoss()
        else:
            self.disease_criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer  # optimizer
        self.scheduler = scheduler  # scheduler
        

        self.acc_train = get_metric_ACC(
                self.num_class, "train/", self.threshold_cls
            )
        
        self.acc_valid = get_metric_ACC(
                self.num_class, "valid/", self.threshold_cls
            )
        
        self.acc_test = get_metric_ACC(
                self.num_class, "test/", self.threshold_cls
            )
            
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
                
    def _step(self, batch: Any, stage: str = "train"):
        base_img, pair_img, reports, change_labels, disease_labels, patient_id = batch
        
        change_labels = change_labels
        outputs = self.model(base_img, pair_img)
        preds = F.softmax(outputs, dim=0)

        change_loss = self.criterion(outputs, change_labels)
        overall_loss = change_loss 

        return change_loss, overall_loss, preds, change_labels

    def training_step(self, batch: Any, batch_idx: int):
        change_loss, overall_loss, pred, change_labels = self._step(batch, stage="train")
        self.acc_train.update(pred, change_labels)
        return {"loss": overall_loss,
                "change_loss": change_loss,
                }

    def validation_step(self, batch: Any, batch_idx: int):
        change_loss, overall_loss, pred, change_labels = self._step(batch, stage="valid")
        self.acc_valid.update(pred, change_labels)
        return {"loss": overall_loss,
                "change_loss": change_loss,
                }
    
    def test_step(self, batch: Any, batch_idx: int):
        change_loss, overall_loss, pred, change_labels = self._step(batch, stage="test")
        self.acc_test.update(pred, change_labels)
        return {"loss": overall_loss,
                "change_loss": change_loss,
                }
    
    def training_step_end(self, step_output):
        self.log_dict(
            add_prefix_to_dict(step_output, "train/"),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )  # logging loss

        self.log_dict(
            key_to_abbreviation(self.acc_train.compute()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )  # logging train classification accuracy
        return step_output

    def on_train_epoch_start(self) -> None:
        self.configure_optimizers()

    def training_epoch_end(self, outputs):
        overall_loss = torch.stack([x['loss'] for x in outputs]).mean()
        change_loss = torch.stack([x['change_loss'] for x in outputs]).mean()
        self.log("train/loss", overall_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
               
        self.log("train/change_loss", change_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        self.log_dict(
            key_to_abbreviation(self.acc_train.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.acc_train.reset()

    def validation_epoch_end(self, outputs):
        overall_loss = torch.stack([x['loss'] for x in outputs]).mean()
        change_loss = torch.stack([x['change_loss'] for x in outputs]).mean()
        self.log("valid/loss", overall_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
               
        self.log("valid/change_loss", change_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        self.log_dict(
            key_to_abbreviation(self.acc_valid.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.acc_valid.reset()

    def test_epoch_end(self, outputs):
        
        # logging
        self.log_dict(
            key_to_abbreviation(self.acc_test.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.acc_test.reset()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict