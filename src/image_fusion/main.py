import numpy as np
import os
import math
import glob
import tifffile as tiff 
from PIL import Image
from itertools import cycle, islice

from gram_schmidt import gram_schmidt_fusion_rgb


current_imgs_dir = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado" + "/2024-08-28"
out_dir = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado" + "/2024-08-28"
os.chdir(current_imgs_dir)
types = ('*.JPG', '*.TIF')
files_grabbed = []
for pattern in types:
    files_grabbed.extend(glob.glob(pattern))
    
input_imgs = sorted(files_grabbed)  
i = cycle(input_imgs)
slc = 5  

for _ in range(math.ceil(len(input_imgs) / slc)):
    cur_imgs = list(islice(i, slc))
    if len(cur_imgs) < slc:
        break
        
    bands = {}
    for img_path in cur_imgs:
        up = img_path.upper()
        if '_D.JPG' in up: bands['D'] = img_path
        elif '_MS_G.TIF' in up: bands['G'] = img_path
        elif '_MS_NIR.TIF' in up: bands['NIR'] = img_path
        elif '_MS_R.TIF' in up: bands['R'] = img_path
        elif '_MS_RE.TIF' in up: bands['RE'] = img_path
    
    if len(bands) < 5:
        print(f"Aviso: Grupo incompleto ignorado. Bandas encontradas: {list(bands.keys())}")
        continue

    ref_img    = np.asarray(Image.open(bands['G'])).astype(np.float32)
    target_jpg = np.asarray(Image.open(bands['D'])).astype(np.float32) * 0.125
    target_nir = np.asarray(Image.open(bands['NIR'])).astype(np.float32)
    target_re  = np.asarray(Image.open(bands['RE'])).astype(np.float32)
    target_r   = np.asarray(Image.open(bands['R'])).astype(np.float32)
    
    multispectral = np.dstack((ref_img, target_jpg, target_nir, target_r, target_re))
    pseudo_rgb = np.dstack((target_r, ref_img, target_nir))
    
    fused_image = gram_schmidt_fusion_rgb(multispectral=multispectral, pseudo_rgb=pseudo_rgb)

    # 4. Geração do nome de saída e Salvamento em TIF
    # Pegamos o nome da imagem original e trocamos a extensão para .TIF
    base_name_clean = os.path.basename(bands['D']).replace('_D.JPG', '_D_FUSED.TIF')
    output_filename = os.path.join(out_dir, base_name_clean)

    # Salvamento usando tifffile (essencial para Float32)
    tiff.imwrite(output_filename, fused_image, photometric='rgb')

    print(f"✅ Salvo com Alta Fidelidade: {output_filename}")

print("\n--- Processamento de Fusão Concluído ---")