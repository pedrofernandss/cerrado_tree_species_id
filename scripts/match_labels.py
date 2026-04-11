import re
import shutil
from pathlib import Path

def extract_id(nome_arquivo):
    # Procura o padrão DJI + 14 dígitos + 4 dígitos
    match = re.search(r"(DJI_\d{14}_\d{4})", nome_arquivo)
    if match:
        return match.group(1) 
    return None

base_dataset = Path("/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado")

dates_list = ["2024-08-28", "2025-01-27-voo1", "2025-01-27-voo2", "2025-01-28", "2025-06-16"] 

images_type = ["fused-imgs", "fused-ndre-imgs", "fused-ndvi-imgs", "ndre-imgs", "ndvi-imgs", "rgb-ndre-imgs", "rgb-ndvi-imgs"]

path_labels_source = base_dataset / "fotos-rotuladas/rgb/labels"

labels_list = list(path_labels_source.glob("*.txt"))
dict_labels = {}
for lbl in labels_list:
    id_lbl = extract_id(lbl.name)
    if id_lbl:
        dict_labels[id_lbl] = lbl

for tipo in images_type:
    
    name_output_folder = tipo.replace("-imgs", "")
    path_output = base_dataset / "fotos-rotuladas" / name_output_folder
    
    img_dest_dir = path_output / "images"
    lbl_dest_dir = path_output / "labels"
    
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    lbl_dest_dir.mkdir(parents=True, exist_ok=True)

    matches_do_tipo = 0
    
    for data in dates_list:
        path_images_source = base_dataset / data / tipo
        
        if not path_images_source.exists():
            continue
            
        lista_imgs = list(path_images_source.glob("*.[tT][iI][fF]"))
        
        for img_path in lista_imgs:
            id_img = extract_id(img_path.name)
            
            if id_img and id_img in dict_labels:
                try:
                    shutil.copy(img_path, img_dest_dir / img_path.name)

                    lbl_origem = dict_labels[id_img]
                    novo_nome_label = img_path.stem + ".txt"
                    shutil.copy(lbl_origem, lbl_dest_dir / novo_nome_label)
                    
                    matches_do_tipo += 1
                except Exception as e:
                    print(f"   Error {id_img} ({data}): {e}")


print("\nMatches realized!")