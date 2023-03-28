import copy

import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.ops as box_ops
from ..datamodules.components._utils import load_cxr_image, load_mask


def check_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach()
        if image.shape[0] == 1 or image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def inverse_image(image):
    image = copy.deepcopy(image)
    image = np.max(image) - image
    return image


def overlay_mask(image, mask, color, w_image=1.0, w_mask=0.5):
    image = copy.deepcopy(image)
    mask = copy.deepcopy(mask)

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach()
        if image.shape[0] == 1 or image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()

    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().squeeze().numpy()

    mask = mask.astype(np.uint8)

    if len(image.squeeze().shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color = np.where(mask_color > 0, color, 0).astype(np.uint8)

    overlay = cv2.addWeighted(image, w_image, mask_color, w_mask, 0)
    return overlay, image


def check_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach()
        if image.shape[0] == 1 or image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        image = image.numpy()
    if image.dtype == np.float32:
        image *= 255
        image = image.astype(np.uint8)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def get_contours(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().squeeze().numpy()
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    mask = mask.astype(np.uint8)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def overlay_contours(
    image, mask, color=(255, 255, 255), thickness=1, return_contour=False
):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()

    image_drawed = copy.deepcopy(image)
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_drawed = cv2.cvtColor(image_drawed, cv2.COLOR_GRAY2BGR)
    contours = get_contours(mask)
    image_drawed = cv2.drawContours(
        image_drawed, contours[0], -1, color=color, thickness=thickness
    )
    if return_contour:
        return image_drawed, contours
    return image_drawed


# def plot_bboxes(image,bboxes, format="cxcywh"):
#     # for bbox in bboxes:
#     #     image = plot_bbox(image, bbox, format)
#     # return image


def plot_bbox(image: np.ndarray, x1, x2, y1, y2, color, thickness: int = 2):
    #  xyxy format
    # print(x1, y1, x2, y2)
    draw = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return draw


def plot_bboxes(
    image, boxes, color=(255, 255, 255), thickness=2, format="cxcywh", labels=None
):
    draw = check_image(copy.deepcopy(image))
    image = check_image(image)
    # boxes = box_ops.box_convert(boxes, format, "xyxy")
    for i, box in enumerate(boxes):
        cx, cy, w, h = box[:4]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1 = int(x1 * image.shape[1])
        x2 = int(x2 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        y2 = int(y2 * image.shape[0])
        draw = plot_bbox(draw, x1, x2, y1, y2, color, thickness)
        
        if labels is not None:
            cv2.putText(
            draw,
            f"{labels[i]}",
            (x1 - 2, y1 - 5),
            1,
            2,
            color=color,
            thickness=thickness,
        )
    return draw


def plot_bboxes_with_scores(image, bboxes, scores, color, thickness=2, format="cxcywh"):
    draw = check_image(copy.deepcopy(image))
    for box, score in zip(bboxes, scores):
        cx, cy, w, h = box[:4]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        x1 = int(x1 * image.shape[1])
        x2 = int(x2 * image.shape[1])
        y1 = int(y1 * image.shape[0])
        y2 = int(y2 * image.shape[0])
        draw = plot_bbox(draw, x1, x2, y1, y2, color, thickness)

        cv2.putText(
            draw,
            f"{score:.2f}",
            (x1 - 2, y1 - 5),
            1,
            2,
            color=color,
            thickness=thickness,
        )
    return draw
