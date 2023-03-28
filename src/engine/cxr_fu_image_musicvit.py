import warnings
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from src.models.cxr_vision_musicvit import ACM_loss

from ..metric import get_metrics_PRFA
from .utils import add_prefix_to_dict, key_to_abbreviation


class CXR_Image(LightningModule):
    
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
        #self.save_hyperparameters(logger=False, ignore=["model"])
        
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
        
        base_img, pair_img, reports, change_labels, disease_labels, patient_id = batch
        change_labels = change_labels
        disease_labels = disease_labels
        base_embed, fu_embed, outputs, matching = self.model(base_img, pair_img)
        preds = F.softmax(outputs, dim=0)

        change_loss = self.criterion(outputs, change_labels)
        
        disease_loss = self.disease_criterion(base_embed.type(torch.cuda.DoubleTensor), 
                                      disease_labels[:][0]) + \
                        self.disease_criterion(fu_embed.type(torch.cuda.DoubleTensor), 
                                       disease_labels[:][1])
                        
        matching_loss = ACM_loss(matching).mean()
        
        overall_loss = change_loss + \
            self.disease_lambda * disease_loss + \
                self.matching_lambda * matching_loss
        return change_loss, disease_loss, matching_loss, overall_loss, preds, change_labels

    def training_step(self, batch: Any, batch_idx: int):
        change_loss, disease_loss, matching_loss, overall_loss, pred, change_labels = self._step(batch, stage="train")
        self.meter_pcls_train.update(pred, change_labels)
        return {"loss": overall_loss,
                "disease_loss": disease_loss,
                "matching_loss": matching_loss,
                "change_loss": change_loss,
                }

    def validation_step(self, batch: Any, batch_idx: int):
        change_loss, disease_loss, matching_loss, overall_loss, pred, change_labels = self._step(batch, stage="valid")
        self.meter_pcls_valid.update(pred, change_labels)
        return {"loss": overall_loss,
                "disease_loss": disease_loss,
                "matching_loss": matching_loss,
                "change_loss": change_loss,
                }
    
    def test_step(self, batch: Any, batch_idx: int):
        change_loss, disease_loss, matching_loss, overall_loss, pred, change_labels = self._step(batch, stage="test")
        self.meter_pcls_test.update(pred, change_labels)
        return {"loss": overall_loss,
                "disease_loss": disease_loss,
                "matching_loss": matching_loss,
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
        overall_loss = torch.stack([x['loss'] for x in outputs]).mean()
        disease_loss = torch.stack([x['disease_loss'] for x in outputs]).mean()
        matching_loss = torch.stack([x['matching_loss'] for x in outputs]).mean()
        change_loss = torch.stack([x['change_loss'] for x in outputs]).mean()
        self.log("train/loss", overall_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        self.log("train/disease_loss", disease_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)        
        self.log("train/matching_loss", matching_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)        
        self.log("train/change_loss", change_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        
        self.log_dict(
            key_to_abbreviation(self.meter_pcls_train.compute()),
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )
        self.meter_pcls_train.reset()

    def validation_epoch_end(self, outputs):
        overall_loss = torch.stack([x['loss'] for x in outputs]).mean()
        disease_loss = torch.stack([x['disease_loss'] for x in outputs]).mean()
        matching_loss = torch.stack([x['matching_loss'] for x in outputs]).mean()
        change_loss = torch.stack([x['change_loss'] for x in outputs]).mean()
        self.log("valid/loss", overall_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)
        self.log("valid/disease_loss", disease_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)        
        self.log("valid/matching_loss", matching_loss, 
                 prog_bar=True, 
                 on_epoch=True,
                 sync_dist=True)        
        self.log("valid/change_loss", change_loss, 
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