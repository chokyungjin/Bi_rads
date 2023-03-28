from typing import Any, Dict

from ._base import BaseDataModule
from .components import CXRMultimodal, CXRMultimodalv2_text


class DataModuleMMCXR_text(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        mean: float,
        std: float,
        transforms: Dict[str, Dict] = None,
        dataset_config: Dict[str, Any] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transforms=transforms,
            mean=mean,
            std=std,
            dataset=CXRMultimodalv2_text,
            dataset_config = dataset_config,
        )
        self.save_hyperparameters(logger=False)