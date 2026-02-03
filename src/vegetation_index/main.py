import glob
import math

import numpy as np
import tifffile as tiff
from PIL import Image
from pathlib import Path
from itertools import cycle, islice

from src.vegetation_index.indexes import calculate_ndre, calculate_ndvi

def calculate_indexes():
    types = ('*.JPG', '*.TIF')  # the tuple of file types
    files_grabbed = []
    for files in types:
        search_path = str(batch_path / files)
        files_grabbed.extend(glob.glob(search_path))

    input_imgs = sorted(files_grabbed)

    i = cycle(input_imgs)
    slc = 6

    for _ in range(math.ceil(len(input_imgs)/slc)):
        cur_imgs = list(islice(i, slc))

        rgb_path = None
        fused_path = None
        nir_path = None
        red_path = None
        re_path = None

        for path in cur_imgs:
            if path.endswith('_D.JPG'):
                rgb_path = path     
            elif '_D_FUSED.JPG' in path:
                fused_path = path
            elif '_MS_NIR.TIF' in path:
                nir_path = path
            elif '_MS_R.TIF' in path:
                red_path = path
            elif '_MS_RE.TIF' in path:
                re_path = path
        
        rgb_obj = Path(rgb_path)
        stem_name = rgb_obj.stem

        rgb_image = np.asarray(Image.open(str(rgb_path)))
        fused_image = np.asarray(Image.open(str(fused_path)))

        ndvi_image = calculate_ndvi(str(red_path), str(nir_path))
        ndre_image = calculate_ndre(str(re_path), str(nir_path))

        # Save pure NDVI e NDRE
        new_stem_ndvi = stem_name.replace('_D', '_NDVI')
        new_stem_ndre = stem_name.replace('_D', '_NDRE')

        ndvi_filename = f"{new_stem_ndvi}.TIF"
        ndre_filename = f"{new_stem_ndre}.TIF"

        tiff.imwrite(ndvi_output_dir / ndvi_filename, ndvi_image)
        tiff.imwrite(ndre_output_dir / ndre_filename, ndre_image)
        
        # Stack and save indexes with RGB image
        stack_rgb_ndvi = np.dstack((rgb_image, ndvi_image))
        stack_rgb_ndre = np.dstack((rgb_image, ndre_image))

        new_stem_rgb_ndvi = stem_name.replace('_D', '_RGB_NDVI')
        new_stem_rgb_ndre = stem_name.replace('_D', '_RGB_NDRE')

        rgb_ndvi_filename = f"{new_stem_rgb_ndvi}.TIF"
        rgb_ndre_filename = f"{new_stem_rgb_ndre}.TIF"

        tiff.imwrite(rgb_ndvi_output_dir / rgb_ndvi_filename, stack_rgb_ndvi)
        tiff.imwrite(rgb_ndre_output_dir / rgb_ndre_filename, stack_rgb_ndre)

        # Stack and save indexes with proposed Fusion
        stack_fused_ndvi = np.dstack((fused_image, ndvi_image))
        stack_fused_ndre = np.dstack((fused_image, ndre_image))
        
        new_stem_fused_ndvi = stem_name.replace('_D', '_FUSED_NDVI')
        new_stem_fused_ndre = stem_name.replace('_D', '_FUSED_NDRE')

        fused_ndvi_filename = f"{new_stem_fused_ndvi}.TIF"
        fused_ndre_filename = f"{new_stem_fused_ndre}.TIF"

        tiff.imwrite(fused_ndvi_output_dir / fused_ndvi_filename, stack_fused_ndvi)
        tiff.imwrite(fused_ndre_output_dir / fused_ndre_filename, stack_fused_ndre)

if __name__ == "__main__":
    batch_path = Path("/datasets/projeto_cerrado/2025-06-16")

    ndvi_output_dir = batch_path / "ndvi-imgs"
    ndre_output_dir = batch_path / "ndre-imgs"

    fused_ndvi_output_dir = batch_path / "fused-ndvi-imgs"
    fused_ndre_output_dir = batch_path / "fused-ndre-imgs"

    rgb_ndvi_output_dir = batch_path / "rgb-ndvi-imgs"
    rgb_ndre_output_dir = batch_path / "rgb-ndre-imgs"

    ndvi_output_dir.mkdir(parents=True, exist_ok=True)
    ndre_output_dir.mkdir(parents=True, exist_ok=True)

    rgb_ndvi_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_ndre_output_dir.mkdir(parents=True, exist_ok=True)

    fused_ndvi_output_dir.mkdir(parents=True, exist_ok=True)
    fused_ndre_output_dir.mkdir(parents=True, exist_ok=True)

    calculate_indexes()
    