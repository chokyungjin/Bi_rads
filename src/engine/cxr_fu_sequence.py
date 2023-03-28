import warnings
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..metric import get_metrics_PRFA
from .utils import add_prefix_to_dict, key_to_abbreviation


class CXR_Text(LightningModule):
    
    def __init__(
        self,
        model: nn.Module,
        criterion=None,
        optimizer=None,
        scheduler=None,
        threshold_cls: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.threshold_cls = threshold_cls
        self.text_model = model
        self.num_class = self.text_model.num_labels
        
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer  # optimizer
        self.scheduler = scheduler  # scheduler
                
         # metrics
        self.meter_pcls_train = get_metrics_PRFA(
                self.num_class, "train/", self.threshold_cls
            )
        self.meter_pcls_valid = get_metrics_PRFA(
                self.num_class, "valid/", self.threshold_cls
            )
        self.meter_pcls_test = get_metrics_PRFA(
                self.num_class, "test/", self.threshold_cls
            )
            
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
                
    def _step(self, batch: Any, stage: str = "train"):
        label = batch['change_labels']
        loss, _, preds = self.text_model(batch['caption_ids'],
                                         None,
                                        batch['attention_mask'],
                                        batch['change_labels'])
        return loss, preds, label, batch

    def training_step(self, batch: Any, batch_idx: int):
        losses, pred, label, _ = self._step(batch, stage="train")
        self.meter_pcls_train.update(pred, label)
        return {"loss": losses}

    def validation_step(self, batch: Any, batch_idx: int):
        losses, pred, label, _ = self._step(batch, stage="valid")
        self.meter_pcls_valid.update(pred, label)
        return {"loss": losses}
    
    def test_step(self, batch: Any, batch_idx: int):
        losses, pred, label, _ = self._step(batch, stage="test")
        self.meter_pcls_test.update(pred, label)
        return {"loss": losses}
    
    def training_step_end(self, step_output):
        self.log_dict(
            add_prefix_to_dict(step_output, "train/"),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )  # logging loss

        self.log_dict(
            key_to_abbreviation(self.meter_pcls_train.compute()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )  # logging train classification accuracy
        return step_output

    def on_train_epoch_start(self) -> None:
        self.configure_optimizers()

    def training_epoch_end(self, outputs):
        self.meter_pcls_train.reset()

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("valid/loss", mean_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        self.log_dict(
            key_to_abbreviation(self.meter_pcls_valid.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.meter_pcls_valid.reset()

    def test_epoch_end(self, outputs):
        
        # logging
        self.log_dict(
            key_to_abbreviation(self.meter_pcls_test.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.meter_pcls_test.reset()

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict