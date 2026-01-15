import glob
import math
import os
import subprocess
from itertools import cycle, islice

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def vig_correct(image_path, infoDict):
  """
    Vignette Correction
    This function applies a vignette correction to an image based on the provided calibration data.
    Parameters:
    - image_path: Path to the image file.
    - infoDict: Dictionary containing calibration data, including:
        - 'Calibrated Optical Center X': X coordinate of the optical center.
        - 'Calibrated Optical Center Y': Y coordinate of the optical center.
        - 'Vignetting Data': Vignetting coefficients.
    Returns:
    - corrected_img: The vignette-corrected image as a NumPy array.
  """

  centerX = int(float(infoDict['Calibrated Optical Center X']))
  centerY = int(float(infoDict['Calibrated Optical Center Y']))
  k = [float(elem.strip(",")) for elem in infoDict['Vignetting Data'].split()]

  # Load the image
  image = Image.open(image_path)
  np_img = np.array(image, dtype=np.uint16)
  rows, cols = np_img.shape[:2]

  # Create meshgrid of coordinates
  y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

  # Calculate the distance r for each pixel
  r = np.sqrt((x_coords - centerX)**2 + (y_coords - centerY)**2)

  # Calculate the correction factor
  correction_factor = (
      k[5] * r**6 +
      k[4] * r**5 +
      k[3] * r**4 +
      k[2] * r**3 +
      k[1] * r**2 +
      k[0] * r +
      1.0
  )

  # Apply the correction factor to the image
  corrected_img = np_img * correction_factor[..., np.newaxis] if np_img.ndim == 3 else np_img * correction_factor
  return corrected_img.astype('uint16')

def undistort(new_img, infoDict):
  """
    Distortion correction
    This function undistorts an image using camera calibration data.
    Parameters:
    - new_img: The input image to be undistorted.
    - infoDict: Dictionary containing calibration data, including:
        - 'Calibrated Optical Center X': X coordinate of the optical center.
        - 'Calibrated Optical Center Y': Y coordinate of the optical center.
        - 'Dewarp Data': Distortion coefficients.
    Returns:
    - dst: The undistorted image as a NumPy array.
  """

  centerX = int(float(infoDict['Calibrated Optical Center X']))
  centerY = int(float(infoDict['Calibrated Optical Center Y']))

  dewarp_args = [float(elem) for elem in infoDict['Dewarp Data'].split(";")[1].split(",")]
  dist_coeffs = np.asarray(dewarp_args[4:])
  tuple0 = (dewarp_args[0], 0, centerX + dewarp_args[2])
  tuple1 = (0, dewarp_args[1], centerY + dewarp_args[3])
  tuple2 = (0, 0, 1)
  camera_matrix = np.asarray([tuple0, tuple1, tuple2])

  h,  w = new_img.shape[:2]
  _, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w,h),1,(w,h))

  dst = cv2.undistort(new_img, camera_matrix, dist_coeffs, None, camera_matrix)
  # crop the image
  x,y,w,h = roi
  dst = dst[y:y+h, x:x+w]
  return dst

def align_phase_rotation(new_img, infoDict):
  """
    Alignment of the phase and rotation differences caused by different camera locations and
    optical accuracy

    This function aligns the phase and rotation differences in an image based on calibration data.
    Parameters:
    - new_img: The input image to be aligned.
    - infoDict: Dictionary containing calibration data, including:
        - 'Calibrated Optical Center X': X coordinate of the optical center.
        - 'Calibrated Optical Center Y': Y coordinate of the optical center.
        - 'Calibrated H Matrix': Homography matrix for perspective transformation.
    Returns:
    - new_img: The aligned (to an ideal plane) image as a NumPy array.
  """
  
  rows, cols = new_img.shape
  H = np.asarray([float(elem) for elem in infoDict['Calibrated H Matrix'].split(",")]).reshape(3,3)
  new_img = cv2.warpPerspective(new_img, H, (cols, rows))
  return new_img


def _smooth_image(image, sigma=1):
    """
        Apply Gaussian smoothing to the image.
        Parameters:
            - image: The input image to be smoothed.
            - sigma: Standard deviation for Gaussian kernel.
        Returns:
            - smoothed_image: The smoothed image as a NumPy array.
    """
    return gaussian_filter(image, sigma=sigma)

def _edge_detection(image):
    """
        Apply Sobel filter to detect edges in the image.
        Parameters:
            - image: The input image to be processed.
        Returns:
            - magnitude: The magnitude of the gradient, representing edge strength.
    """

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return magnitude

def align_images_using_ecc(reference_image, target_image, jpg=False):
    """
        Align two images using the Enhanced Correlation Coefficient (ECC) algorithm.
        Parameters:
            - reference_image: The reference image to align to.
            - target_image: The target image to be aligned.
            - jpg: Boolean indicating if the target image is a JPG file.
        Returns:
            - aligned_image: The aligned target image as a NumPy array.
    """
    # JPG images do not have the same exif data as the reference TIFF images,
    # therefore they need a different treatment
    if jpg:
        # save colored jpg
        colored_jpg = target_image
        # create temporary greyscale img
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) 
    
    # smooth the images
    reference_image_smoothed = _smooth_image(reference_image)
    target_image_smoothed = _smooth_image(target_image)

    # apply edge detection
    base_edges = _edge_detection(reference_image_smoothed)
    target_edges = _edge_detection(target_image_smoothed)

    # convert images to float32 for ECC algorithm
    base_edges = base_edges.astype(np.float32)
    target_edges = target_edges.astype(np.float32)
    
    # define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    # set the number of iterations and termination criteria
    if jpg:
        # increase iterations for jpg images, beacause they went through lees processing
        number_of_iterations = 500 
    else:
        number_of_iterations = 25
    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)


    # apply the ECC algorithm to find the warp matrix
    cc, warp_matrix = cv2.findTransformECC(base_edges, target_edges, warp_matrix, warp_mode, criteria)
    
    if jpg:
        target_image = colored_jpg

    # warp the target image to align with the base image
    aligned_image = cv2.warpPerspective(target_image, warp_matrix, (reference_image.shape[1], reference_image.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return aligned_image