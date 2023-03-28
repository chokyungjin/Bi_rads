import os
import warnings
from typing import Dict, Optional

import albumentations as A
from torch.utils import data

from ..transforms import build_transforms


class BaseDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        transforms: Optional[Dict] = None,
    ):
        super().__init__()

        if not os.path.isdir(data_dir):
            raise ValueError(f"{data_dir} is invaild path")
        self.data_dir = data_dir

        if split not in ["train", "valid", "test"]:
            msg = f"{split} is invaild split. split should be in 'train', 'val' and 'test'"
            raise ValueError(msg)
        self.split = split

        if transforms is None:
            warnings.warn("transforms is None")
        self.transforms = transforms

        self.num_classes = None
        self.collate_fn = None

    def _get_transform(self, transforms: dict, mean: float, std: float,  **kwargs):
        if transforms is not None:
            return A.Compose(
                transforms=build_transforms(transforms, mean, std),
                additional_targets={'image0': 'image', 
                                    'image1': 'image', 
                                    'image2': 'image'},
                **kwargs,
            )
        return None

    def _add_normalset(self, split):
        pass
