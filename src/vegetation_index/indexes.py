import numpy as np
from PIL import Image

def calculate_ndvi(red_img_path: str, nir_img_path: str):

    red_image = np.asarray(Image.open(red_img_path)).astype(np.float64)
    nir_image = np.asarray(Image.open(nir_img_path)).astype(np.float64)

    numerator = nir_image - red_image
    denominator = nir_image + red_image

    ndvi = np.divide(numerator, denominator, out=np.zeros_like(
        numerator), where=denominator != 0)

    ndvi_norm = (ndvi + 1) / 2
    img_uint8 = (ndvi_norm * 255).astype(np.uint8)

    return img_uint8

def calculate_ndre(re_img_path: str, nir_img_path: str):

    re_image = np.asarray(Image.open(re_img_path)).astype(np.float64)
    nir_image = np.asarray(Image.open(nir_img_path)).astype(np.float64)

    numerator = nir_image - re_image
    denominator = nir_image + re_image

    ndre = np.divide(numerator, denominator, out=np.zeros_like(
        numerator), where=denominator != 0)

    ndre_norm = (ndre + 1) / 2
    img_uint8 = (ndre_norm * 255).astype(np.uint8)

    return img_uint8