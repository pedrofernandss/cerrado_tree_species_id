import numpy as np
import pandas as pd
from PIL import Image
import os
import math
from itertools import cycle , islice
import glob

from gram_schmidt import gram_schmidt_fusion_rgb


current_imgs_dir = os.path.dirname(os.getcwd()) + "/2025-06-16"
out_dir = os.path.dirname(os.getcwd()) + "/2025-06-16"
os.chdir(current_imgs_dir)
types = ('*.JPG', '*.TIF')
files_grabbed = []
for pattern in types:
    files_grabbed.extend(glob.glob(pattern))
    
input_imgs = sorted(files_grabbed)  
i = cycle(input_imgs)
slc = 5  # number of images per group

# process images in groups of 5
for _ in range(math.ceil(len(input_imgs) / slc)):
    cur_imgs = list(islice(i, slc))
    if len(cur_imgs) < slc:
        break
        
    # determine destination folder based on current image's basename.
    base_name = os.path.basename(cur_imgs[0][:-4]) + '_FUSED.jpg'

    bands = {}
    for img_path in cur_imgs:
        up = img_path.upper()
        if '_D.JPG' in up: bands['D'] = img_path
        elif '_MS_G.TIF' in up: bands['G'] = img_path
        elif '_MS_NIR.TIF' in up: bands['NIR'] = img_path
        elif '_MS_R.TIF' in up: bands['R'] = img_path
        elif '_MS_RE.TIF' in up: bands['RE'] = img_path
    
    # load images from current group
    ref_img = np.asarray(Image.open(bands['G']))
    target_jpg = np.asarray(Image.open(bands['D'])) * 0.125
    target_nir = np.asarray(Image.open(bands['NIR']))
    target_re = np.asarray(Image.open(bands['RE']))  
    target_r = np.asarray(Image.open(bands['R'])) 
    
    
    # stack images to form multispectral and pseudo-RGB inputs
    multispectral = np.dstack((ref_img, target_jpg, target_nir, target_r, target_re))
    pseudo_rgb = np.dstack((target_r, ref_img, target_nir))
    
    fused_image = gram_schmidt_fusion_rgb(multispectral=multispectral, pseudo_rgb=pseudo_rgb)

    dest_folder_fused = os.path.dirname(os.getcwd()) + "/2025-06-16/fused"

    output_filename = os.path.join(dest_folder_fused, f"{os.path.splitext(base_name)[0]}.JPG")

    # save the fused image
    Image.fromarray(fused_image).save(output_filename)

    print(f"Saved fused image: {output_filename}")


print("All images from selected date was transformed, fused and saved.")
