import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class RandomBlur(ImageOnlyTransform):
    def __init__(self, blur_limit=15, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        # random kernel size (3, blur_limit)
        basic = A.Blur(blur_limit=blur_limit, p=1.0)
        gaussian = A.GaussianBlur(blur_limit=(3, blur_limit), p=1.0)
        median = A.MedianBlur(blur_limit=blur_limit, p=1.0)
        motion = A.MotionBlur(blur_limit=blur_limit, p=1.0)
        self.blur = A.OneOf([basic, gaussian, median, motion], p=p)

    def apply(self, image, **params):
        return self.blur(image=image)["image"]

    def get_transform_init_args_names(self):
        return "blur_limit"

class RandomBlur_music(ImageOnlyTransform):
    def __init__(self, blur_limit=3, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        # random kernel size (3, blur_limit)
        median = A.MedianBlur(blur_limit=blur_limit, p=0.1)
        motion = A.MotionBlur(p=0.2)
        sharp = A.Sharpen(p=0.2)
        self.blur = A.OneOf([median, motion, sharp], p=p)

    def apply(self, image, **params):
        return self.blur(image=image)["image"]

    def get_transform_init_args_names(self):
        return "blur_limit"


class RandomGamma(A.RandomGamma):
    def __init__(self, gamma_limit: int = 20, eps=None, always_apply=False, p=0.5):
        self.gamma_limit = (int(100 - gamma_limit), int(100 + gamma_limit))
        super().__init__(self.gamma_limit, eps, always_apply, p)


def random_pixel_augment_v1(
    limit_blur=15,
    limit_gamma=5,
    limit_brightness=0.2,
    limit_contrast=0.2,
    p=0.5,
):
    return A.OneOf(
        [
            RandomBlur(blur_limit=limit_blur, p=1.0),
            RandomGamma(gamma_limit=limit_gamma, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=limit_brightness, contrast_limit=0.0, p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=limit_contrast, p=1.0
            ),
        ],
        p=p,
    )


def random_pixel_augment_v2(
    limit_blur=15,
    limit_gamma=5,
    limit_brightness=0.2,
    limit_contrast=0.2,
    n=2,
    p=0.5,
):
    return A.SomeOf(
        [
            RandomBlur(blur_limit=limit_blur, p=1.0),
            RandomGamma(gamma_limit=limit_gamma, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=limit_brightness, contrast_limit=0.0, p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.0, contrast_limit=limit_contrast, p=1.0
            ),
        ],
        n=n,
        p=p,
    )

def random_pixel_augment_music(
    p = 1
):
    return A.Compose([
                    A.OneOf([A.MedianBlur(blur_limit=3, p=0.1),
                    A.MotionBlur(p=0.2),
                    A.Sharpen(p=0.2),
                    ], p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.2),
                    A.OneOf([
                        A.OpticalDistortion(p=0.3),
                    ], p=0.2),
                    A.OneOf([
                        A.CLAHE(clip_limit=4.0),
                        A.Equalize(),
                    ], p=0.2),
                    A.OneOf([
                        A.GaussNoise(p=0.2),
                        A.MultiplicativeNoise(p=0.2),
                    ], p=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0.1, p=0.3)
    ])