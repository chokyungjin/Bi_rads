# albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, ListConfig

# pixel level transforms
from .randaug_pixel import (RandomBlur, RandomGamma,
                            random_pixel_augment_music,
                            random_pixel_augment_v1, random_pixel_augment_v2)
# spatial level transforms
from .randaug_spatial import (random_resize_crop_v1, random_resize_crop_v2,
                              random_rotate_v1, random_rotate_v2,
                              random_shear_v1, random_shear_v2,
                              random_spatial_augment_v1, random_translate_v1,
                              random_translate_v2)

available_transforms__ = {
    "resize": A.Resize,
    "random_flip": A.HorizontalFlip,
    #
    # pixel augmentations
    #
    "random_gamma": RandomGamma,
    "random_blur": RandomBlur,
    "random_bright_contrast": A.RandomBrightnessContrast,
    "random_pixel_augment_v1": random_pixel_augment_v1,
    "random_pixel_augment_v2": random_pixel_augment_v2,
    "random_pixel_augment_music": random_pixel_augment_music,
    #
    # spatial augmentations
    #
    "random_resize_crop_v1": random_resize_crop_v1,
    "random_resize_crop_v2": random_resize_crop_v2,
    "random_rotate_v1": random_rotate_v1,
    "random_rotate_v2": random_rotate_v2,
    "random_shear_v1": random_shear_v1,
    "random_shear_v2": random_shear_v2,
    "random_translate_v1": random_translate_v1,
    "random_translate_v2": random_translate_v2,
    "random_spatial_augment_v1": random_spatial_augment_v1,
}

def check(transform: dict):
    tmp = dict()
    for key, val in transform.items():
        # print(key, val)
        if isinstance(val, ListConfig):
            tmp[key] = tuple(val)
        elif isinstance(val, DictConfig):
            tmp[key] = check(val)
        else:
            tmp[key] = val
    return tmp


def build_transforms(transforms: dict, mean: float, std: float):
    keys = list(transforms.keys())

    transform_list = []

    for key, val in transforms.items():

        if val is None:
            val = {}

        new_key = str(key)
        new_val = check(dict(val))

        transform_list.append(available_transforms__[new_key](**new_val))
        keys.remove(key)
    
    transform_list.append(A.Normalize(mean=(mean), std=(std,)),)
    transform_list.append(ToTensorV2())
    
    assert not keys
    return transform_list