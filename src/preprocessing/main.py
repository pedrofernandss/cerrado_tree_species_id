import math
import os
import cv2

from glob import glob
from itertools import cycle, islice

from metadata import get_xml_metadata
from corrections import align_images_using_ecc, vig_correct, undistort, align_phase_rotation
from transforms import zoom_center, crop_center

def correct_image(imgPath):
    """
        Process the image by applying vignette correction, undistortion, and alignment.
        Parameters:
            - imgPath: Path to the image file.
        Returns:
            - new_img: The processed image as a NumPy array.
    """
    
    IMG_REF_SHAPE = (2570, 1925)
    infoDict = get_xml_metadata(imgPath)

    # custom pipeline for jpg images, because they have a different resolution 
    if imgPath[-3:] == 'JPG':
            new_img = cv2.imread(imgPath)
            new_img = zoom_center(new_img, 1.3)
            new_img = cv2.resize(new_img, IMG_REF_SHAPE)
            new_img = crop_center(new_img, 1500)
            return new_img

    new_img = vig_correct(imgPath, infoDict)
    new_img = undistort(new_img, infoDict)
    new_img = align_phase_rotation(new_img, infoDict)
    new_img = crop_center(new_img, 1500)

    return new_img

def batch_image_preprocessing():

    # DRIVER
    IMG_REF_SHAPE = (2570, 1925)

    # Take relative paths, because chenged os current dir
    os.chdir("./raw-imgs")
    types = ('*.JPG', '*.TIF') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    
    input_imgs = sorted(files_grabbed)

    i = cycle(input_imgs)
    slc = 5
    for _ in range(math.ceil(len(input_imgs)/slc)):
        cur_imgs = list(islice(i,slc))
    
        bands = {}
        for img_path in cur_imgs:
            up = img_path.upper()
            if '_D.JPG' in up: bands['D'] = img_path
            elif '_MS_G.TIF' in up: bands['G'] = img_path
            elif '_MS_NIR.TIF' in up: bands['NIR'] = img_path
            elif '_MS_R.TIF' in up: bands['R'] = img_path
            elif '_MS_RE.TIF' in up: bands['RE'] = img_path

        ref_img = correct_image(bands['G']) # G BAND -> REFERENCE
        target_jpg = correct_image(bands['D'])
        target_nir = correct_image(bands['NIR'])
        target_r = correct_image(bands['R'])
        target_re = correct_image(bands['RE'])
    
        # align image to G BAND
        aligned_jpg_image = align_images_using_ecc(ref_img, target_jpg, True)
        aligned_nir_image = align_images_using_ecc(ref_img, target_nir)
        aligned_r_image = align_images_using_ecc(ref_img, target_r)
        aligned_re_image = align_images_using_ecc(ref_img, target_re)

        # remember to crop and normalize reference/target img here:
        # normalize the image to range 0 to 255 and convert to uint8
        aligned_jpg_image = cv2.normalize(aligned_jpg_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aligned_nir_image = cv2.normalize(aligned_nir_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aligned_r_image = cv2.normalize(aligned_r_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aligned_re_image = cv2.normalize(aligned_re_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        aligned_g_image = cv2.normalize(ref_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # crop again
        aligned_jpg_image = crop_center(aligned_jpg_image, 1000)
        aligned_nir_image = crop_center(aligned_nir_image, 1000)
        aligned_r_image = crop_center(aligned_r_image, 1000)
        aligned_re_image = crop_center(aligned_re_image, 1000)
        aligned_g_image = crop_center(aligned_g_image, 1000)

        cv2.imwrite(f"{os.path.dirname(os.getcwd())}/preprocessed-imgs/processed-{bands['D']}", aligned_jpg_image)
        cv2.imwrite(f"{os.path.dirname(os.getcwd())}/preprocessed-imgs/processed-{bands['G']}", aligned_g_image)
        cv2.imwrite(f"{os.path.dirname(os.getcwd())}/preprocessed-imgs/processed-{bands['NIR']}", aligned_nir_image)
        cv2.imwrite(f"{os.path.dirname(os.getcwd())}/preprocessed-imgs/processed-{bands['R']}", aligned_r_image)
        cv2.imwrite(f"{os.path.dirname(os.getcwd())}/preprocessed-imgs/processed-{bands['RE']}", aligned_re_image)

if __name__ == "__main__":
    batch_image_preprocessing()