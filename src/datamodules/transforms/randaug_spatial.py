import albumentations as A
import cv2
from albumentations import Affine, Rotate

from .random_resize_crop import RandomResizeCrop

"""
cv2.BORDER_CONSTANT     : 0
cv2.BORDER_REPLICATE    : 1
cv2.BORDER_REFLECT      : 2
cv2.BORDER_WRAP         : 3
cv2.BORDER_REFLECT_101  : 4
cv2.BORDER_TRANSPARENT  : 5
"""


def random_resize_crop_v1(
    height=768, width=768, scale=0.1, border_mode=cv2.BORDER_CONSTANT, p=0.5
):
    if border_mode == cv2.BORDER_CONSTANT:
        return RandomResizeCrop(height, width, scale, border_mode, p=p)
    return RandomResizeCrop(height, width, scale, border_mode, p=p)


def random_resize_crop_v2(height=768, width=768, scale=0.1, p=0.5):
    rrc1 = random_resize_crop_v1(height, width, scale, cv2.BORDER_CONSTANT)
    rrc2 = random_resize_crop_v1(height, width, scale, cv2.BORDER_REFLECT_101)
    return A.OneOf([rrc1, rrc2], p=p)


def random_translate_v1(translate_x, translate_y, p=0.5):
    trs = Affine(
        translate_percent={
            "x": (-translate_x / 100, translate_x / 100),
            "y": (-translate_y / 100, translate_y / 100),
        },
        p=p,
    )
    return trs


def random_translate_v2(translate_x, translate_y, p=0.5):
    trs1 = Affine(translate_percent={"x": (-translate_x / 100, translate_x / 100), "y": 0.0})
    trs2 = Affine(translate_percent={"x": 0.0, "y": (-translate_y / 100, translate_y / 100)})
    return A.OneOf([trs1, trs2], p=p)


def random_rotate_v1(rotate, border_mode=cv2.BORDER_CONSTANT, p=0.5):
    if border_mode == cv2.BORDER_CONSTANT:
        return Rotate(
            limit=rotate,
            interpolation=cv2.INTER_LINEAR,
            border_mode=border_mode,
            value=0,
            mask_value=0,
            p=p,
        )
    return Rotate(
        limit=rotate,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        p=p,
    )


def random_rotate_v2(rotate, p=0.5):
    rot1 = random_rotate_v1(rotate, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    rot2 = random_rotate_v1(rotate, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    return A.OneOf([rot1, rot2], p=p)


def random_shear_v1(x: int, y: int, border_mode=cv2.BORDER_CONSTANT, p=0.5):
    if border_mode == cv2.BORDER_CONSTANT:
        return Affine(
            shear={"x": (-x, x), "y": (-y, y)},
            mode=border_mode,
            cval=0,
            cval_mask=0,
            p=p,
        )
    return Affine(shear={"x": (-x, x), "y": (-y, y)}, mode=cv2.BORDER_REFLECT_101, p=p)


def random_shear_v2(x, y, p=0.5):
    shear1 = random_shear_v1(x=x, y=y, border_mode=cv2.BORDER_CONSTANT, p=1.0)
    shear2 = random_shear_v1(x=x, y=y, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    return A.OneOf([shear1, shear2], p=p)


def random_spatial_augment_v1(
    height=768,
    width=768,
    scale=0.1,
    rotate=45,
    shear_x=20,
    shear_y=20,
    translate_x=15,
    translate_y=15,
    p=0.5,
):
    return A.OneOf(
        [
            random_resize_crop_v1(height, width, scale, cv2.BORDER_CONSTANT),
            random_rotate_v2(rotate),
            random_shear_v2(shear_x, shear_y),
            random_translate_v1(translate_x, translate_y),
        ],
        p=p,
    )
