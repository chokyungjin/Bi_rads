import pydicom as pyd
import copy
import cv2
import numpy as np
from PIL import Image
from pydicom.pixel_data_handlers import apply_rescale, apply_voi_lut


# def read_cxr_image(path):


def get_bits_stored(dcm: pyd.FileDataset):
    return dcm[0x0028, 0x0101].value


def get_bits_allocted(dcm: pyd.FileDataset):
    return dcm[0x0028, 0x0100].value


def invert_pixel_array(
    pixel_array: np.ndarray, dcm: pyd.FileDataset, use_dcm_header=False
):
    """
    invert dicom pixel array
    return : inverted pixel array
    """

    if use_dcm_header:
        bits_stored = get_bits_stored(dcm)
        max_value = 2 ** int(bits_stored) - 1
    else:
        max_value = np.max(pixel_array)

    return max_value - pixel_array


def save_image_uint16(image, path):
    image = image.astype(np.uint16)
    array_buffer = image.tobytes()
    output = Image.new("I", image.T.shape)
    output.frombytes(array_buffer, "raw", "I;16")
    output.save(path)


def get_image_from_dcm(
    dicom,
    output_path=None,
    use_dcm_header=False,
):
    """
    dcm : str or pyd.FileDataset
    return : np.array
    """
    if isinstance(dicom, str):
        dcm = pyd.dcmread(dicom)
    elif isinstance(dicom, pyd.FileDataset):
        dcm = copy.deepcopy(dicom)

    pixel_array = dcm.pixel_array
    pixel_array = apply_rescale(pixel_array, dcm)

    # check
    # print(dcm[0x0028, 0x0004].value)
    # origin = dcm[0x0028, 0x0004].value
    # converted = str(origin).upper().strip()
    # if origin != converted:
    #     print(origin, converted, dicom)
    #     dcm[0x0028, 0x0004].value = "MONOCHROME1"
    # dcm[0x0028, 0x0004].value = str(origin).upper().strip()

    if use_dcm_header:
        pixel_array = apply_voi_lut(pixel_array, dcm)

    if get_photometric_interpretation(dcm) == "MONOCHROME1":
        pixel_array = invert_pixel_array(pixel_array, dcm, use_dcm_header)

    # min max normalization
    max_value = np.max(pixel_array)
    min_value = np.min(pixel_array)
    pixel_array = (pixel_array - min_value) / (max_value - min_value)
    pixel_array *= 2**16 - 1

    assert np.max(pixel_array) == 65535 and np.min(pixel_array) == 0

    if output_path is not None:
        save_image_uint16(pixel_array, output_path)
    return pixel_array

def get_bitsstored_bitsallocated(dcm: pyd.FileDataset):
    bits_stored = get_bits_stored(dcm)
    bits_allocated = get_bits_allocted(dcm)
    return bits_stored, bits_allocated


def get_photometric_interpretation(dcm: pyd.FileDataset):
    """
    get photometric interpretation value from dicom

    return : 'MONOCHROME1' or 'MONOCHROME2'
    """
    photometric = str(dcm[0x0028, 0x0004].value)
    photometric = "MONOCHROME2" if photometric is None else photometric
    photometric = photometric.upper().strip()

    return photometric


def from_uint16_to_float32(pixel_array: np.ndarray, dcm: pyd.FileDataset):
    """ """
    bits_stored = get_bits_stored(dcm)
    max_value = (
        2 ** int(bits_stored) - 1 if bits_stored is not None else np.max(pixel_array)
    )
    return pixel_array.astype(np.float32) / max_value


def get_max_contour(contours):
    max_size = -1
    max_contour = None
    for contour in contours[0]:
        if cv2.contourArea(contour) > max_size:
            max_contour = contour
            max_size = cv2.contourArea(contour)
    return max_contour


def get_box_from_mask(path_mask, output_format="cxcywh"):
    # from torchvision.ops.boxes import box_convert

    mask: np.array = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    height, width = mask.shape[:2]
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours[0]) != 1:
        raise ValueError(f"{path_mask} has multiple contours")

    max_contour = get_max_contour(contours)
    x, y, w, h = cv2.boundingRect(max_contour)

    x = x / width
    w = w / width

    y = y / height
    h = h / height

    cx = x + w / 2
    cy = y + h / 2
    # output = box_convert(box, in_fmt="xywh", out_fmt=output_format)[0]
    return np.array([cx, cy, w, h])


# output_arr = utils_cxr.from_uint16_to_float32(pixel_array,dcm)
# output_arr = Image.fromarray(output_arr)
# output_path = path_dcm.replace(".dcm", ".tiff")
# output_path = output_path.replace("dicom", "png")
# output_arr.save(output_path)
