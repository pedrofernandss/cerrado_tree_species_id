import cv2

from metadata import get_xml_metadata
from corrections import vig_correct, undistort, align_phase_rotation
from tranforms import zoom_center, crop_center

def process_image(imgPath):
    """
        Process the image by applying vignette correction, undistortion, and alignment.
        Parameters:
            - imgPath: Path to the image file.
        Returns:
            - new_img: The processed image as a NumPy array.
    """
    
    # constant for the reference image size
    IMG_REF_SHAPE = (2570, 1925)

    # get xml metadata for camera corrections
    infoDict = get_xml_metadata(imgPath)

    # custom pipeline for jpg images, because they have a different resolution 
    if imgPath[-3:] == 'JPG':
            new_img = cv2.imread(imgPath)
            new_img = zoom_center(new_img, 1.3)
            new_img = cv2.resize(new_img, IMG_REF_SHAPE)
            new_img = crop_center(new_img, 1500)
            return new_img


    # apply vignette correction
    new_img = vig_correct(imgPath, infoDict)

    # undistort image
    new_img = undistort(new_img, infoDict)

    # align phase and rotation
    new_img = align_phase_rotation(new_img, infoDict)
    
    # crop center
    new_img = crop_center(new_img, 1500)

    return new_img