import random

import cv2
from albumentations.augmentations import functional as f
from albumentations.augmentations.bbox_utils import (denormalize_bbox,
                                                     normalize_bbox)
from albumentations.augmentations.crops import functional as crop_f
from albumentations.augmentations.geometric import functional as geo_f
from albumentations.core.transforms_interface import DualTransform


class RandomResizeCrop(DualTransform):
    """Crop a random part of the input.
    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        height,
        width,
        scale=0.2,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=False,
        p=1.0,
    ):

        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.scale_limit = (1.0 - scale, 1.0 + scale)
        self.interpolation = interpolation

        self.border_mode = border_mode
        self.value = 0
        self.mask_value = 0

    def get_params(self):
        return {
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "h_start": random.random(),
            "w_start": random.random(),
        }

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)

        height = params["rows"]
        width = params["cols"]
        scale = params["scale"]
        rows, cols = int(height * scale), int(width * scale)
        if rows < self.height:
            h_pad_top = int((self.height - rows) / 2.0)
            h_pad_bottom = self.height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.width:
            w_pad_left = int((self.width - cols) / 2.0)
            w_pad_right = self.width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        # now support only center
        # h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
        #    h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        # )

        params.update(
            {
                "rows": rows,
                "cols": cols,
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params

    def apply(
        self,
        img,
        scale,
        h_start=0,
        w_start=0,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        interpolation=cv2.INTER_LINEAR,
        **params
    ):
        img = geo_f.scale(img, scale, interpolation)
        if scale >= 1.0:
            return crop_f.random_crop(img, self.height, self.width, h_start, w_start)
        else:
            return f.pad_with_params(
                img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                border_mode=self.border_mode,
                value=self.value,
            )

    def apply_to_mask(
        self,
        img,
        scale,
        h_start=0,
        w_start=0,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        interpolation=cv2.INTER_LINEAR,
        **params
    ):
        img = geo_f.scale(img, scale, cv2.INTER_NEAREST)

        if scale >= 1.0:
            return crop_f.random_crop(img, self.height, self.width, h_start, w_start)
        else:
            return f.pad_with_params(
                img,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                border_mode=self.border_mode,
                value=self.mask_value,
            )

    def apply_to_bbox(
        self,
        bbox,
        scale,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        rows=0,
        cols=0,
        **params
    ):
        # Bounding box coordinates are scale invariant
        if scale >= 1.0:

            params.pop("interpolation")
            return crop_f.bbox_random_crop(
                bbox,
                self.height,
                self.width,
                rows=rows,
                cols=cols,
                **params,
            )
        else:
            x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
            bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
            return normalize_bbox(
                bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right
            )

    def apply_to_keypoint(
        self,
        keypoint,
        scale,
        pad_top=0,
        pad_bottom=0,
        pad_left=0,
        pad_right=0,
        **params
    ):
        keypiont = geo_f.keypoint_scale(keypoint, scale, scale)
        if scale >= 1.0:
            return crop_f.keypoint_random_crop(
                keypoint, self.height, self.width, **params
            )
        else:
            x, y, angle, scale = keypoint
            return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return ("height", "width", "scale_limit", "interpolation")