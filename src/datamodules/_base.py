from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .components._base import BaseDataset


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        mean: float,
        std: float,
        transforms: Optional[Dict[str, Dict]] = None,
        dataset: Optional[BaseDataset] = None,
        dataset_config: Dict[str, Any] = None,

    ):
        super().__init__()
        # self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.presistent_worksers = True if self.num_workers > 0 else False

        self.transforms = dict()
        self.transforms["train"] = transforms["train"]
        self.transforms["valid"] = transforms["valid"]
        self.transforms["test"] = transforms["test"]

        self.dataset = dataset
        self.dataset_config = dataset_config
        self.data_train: Optional[BaseDataset] = None
        self.data_val: Optional[BaseDataset] = None
        self.data_test: Optional[BaseDataset] = None

        self.setup()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.

        called on every process in DDP

        """
        datasets = dict()
        for split in ["train", "valid", "test"]:
            datasets[split] = self.dataset(
                data_dir=self.data_dir,
                split=split,
                transforms=self.transforms[split],
                mean=self.mean,
                std=self.std,
                **self.dataset_config
            )

        self.data_train = datasets["train"]
        self.data_val = datasets["valid"]
        self.data_test = datasets["test"]

    @property
    def num_classes(self) -> int:
        return self.data_train.num_class

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=None,
            collate_fn=self.data_train.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.presistent_worksers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=self.data_test.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.presistent_worksers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=self.data_val.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.presistent_worksers,
        )
