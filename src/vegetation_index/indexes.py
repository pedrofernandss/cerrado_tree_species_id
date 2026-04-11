import numpy as np
from PIL import Image

def calculate_ndvi(red_img_path: str, nir_img_path: str):

    red_image = np.asarray(Image.open(red_img_path)).astype(np.float64)
    nir_image = np.asarray(Image.open(nir_img_path)).astype(np.float64)

    numerator = nir_image - red_image
    denominator = nir_image + red_image

    ndvi = np.divide(numerator, denominator, out=np.zeros_like(
        numerator), where=denominator != 0)

    return ndvi.astype(np.float32)

def calculate_ndre(re_img_path: str, nir_img_path: str):

    re_image = np.asarray(Image.open(re_img_path)).astype(np.float64)
    nir_image = np.asarray(Image.open(nir_img_path)).astype(np.float64)

    numerator = nir_image - re_image
    denominator = nir_image + re_image

    ndre = np.divide(numerator, denominator, out=np.zeros_like(
        numerator), where=denominator != 0)

    return ndre.astype(np.float32)