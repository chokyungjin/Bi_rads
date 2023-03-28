import cv2
import numpy as np
import pydicom
from PIL import Image


def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img

def label_process(json_list, idx, d_num):
        
    if d_num == 14:
        for num in range(d_num):
            if isNaN(json_list['disease_labels'][idx][0][num]):
                json_list['disease_labels'][idx][0][num] = 0
            elif json_list['disease_labels'][idx][0][num] == -1:
                json_list['disease_labels'][idx][0][num] = 0
                
            if isNaN(json_list['disease_labels'][idx][1][num]):
                json_list['disease_labels'][idx][1][num] = 0
            elif json_list['disease_labels'][idx][1][num] == -1:
                json_list['disease_labels'][idx][1][num] = 0  
    else:
        ## normal 0 abnormal 1
        if json_list['disease_labels'][idx][0] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:  # No Finding
            json_list['disease_labels'][idx][0] = 0
        elif json_list['disease_labels'][idx][0] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            json_list['disease_labels'][idx][0] = 0
        else:
            json_list['disease_labels'][idx][0] = 1
            
        if json_list['disease_labels'][idx][1] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:  # No Finding
            json_list['disease_labels'][idx][1] = 0
        elif json_list['disease_labels'][idx][1] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            json_list['disease_labels'][idx][1] = 0
        else:
            json_list['disease_labels'][idx][1] = 1
    
    return json_list['disease_labels'][idx]

def check_crop_label_margin(crop_label, margin=True, ratio=0.08):
        x_start, x_end, y_start, y_end = crop_label
        x_margin = int(511-x_start)
        y_margin = int(y_end-y_start)

        if x_margin < 300:
            x_start, x_end = 47, 445
        if y_margin < 300:
            y_start, y_end = 55, 453

        if margin is True:
            margin = int((x_end - x_start) * ratio // 2)
            x_start -= margin
            x_end += margin

            margin = int((y_end - y_start) * ratio // 2)
            y_start -= margin
            y_end += margin

        if x_start < 0:
            x_start = 0
        if y_start < 0:
            y_start = 0
        if x_end > 511:
            x_end = 511
        if y_end > 511:
            y_end = 511

        return [x_start, x_end, y_start, y_end]
    
def get_imgs(img_path, fov, transforms=None, multiscale=False):
    base_path = img_path[0]
    base_path = base_path.replace('/2.0.0/files/', '/2.0.0/png8/files/')
    base_path = base_path.replace('.dcm', '.png')
    
    pair_path = img_path[1]
    pair_path = pair_path.replace('/2.0.0/files/', '/2.0.0/png8/files/')
    pair_path = pair_path.replace('.dcm', '.png')
    
    base_img = cv2.imread(str(base_path), 0)
    pair_img = cv2.imread(str(pair_path), 0)
    
    # tranform images
    x_min, _, y_min, y_max = check_crop_label_margin(fov[0])
    base_img = base_img[x_min:, y_min:y_max]
    
    # tranform images
    x_min, _, y_min, y_max = check_crop_label_margin(fov[1])
    pair_img = pair_img[x_min:, y_min:y_max]
    
    if transforms is not None:
        transformed = transforms(image=base_img, image1=pair_img)
        base_img = transformed['image']
        pair_img = transformed['image1']

    return base_img, pair_img

def isNaN(num):
    return num != num