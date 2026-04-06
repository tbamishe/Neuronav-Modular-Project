import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

def convert_masks_to_yolo(input_root, output_root, tool_map):
    images_out = os.path.join(output_root, "images")
    labels_out = os.path.join(output_root, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    mask_files = list(Path(input_root).rglob("*_endo_watershed_mask.png"))
    
    for mask_path in tqdm(mask_files, desc="Converting Masks"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[0] == 0: continue
        
        h, w = mask.shape
        img_path = str(mask_path).replace("_endo_watershed_mask.png", "_endo.png")
        file_id = mask_path.stem.replace("_endo_watershed_mask", "")
        label_file = os.path.join(labels_out, f"{file_id}.txt")
        
        has_tools = False
        with open(label_file, 'w') as f:
            for target_val, yolo_id in tool_map.items():
                binary_mask = np.where(mask == target_val, 255, 0).astype(np.uint8)
                if np.max(binary_mask) == 0: continue

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 100: continue
                    
                    points = cnt.reshape(-1, 2)
                    norm_pts = [f"{p[0]/max(1,w):.6f} {p[1]/max(1,h):.6f}" for p in points]
                    if len(norm_pts) > 2:
                        f.write(f"{yolo_id} {' '.join(norm_pts)}\n")
                        has_tools = True
        
        if has_tools:
            shutil.copy(img_path, os.path.join(images_out, f"{file_id}.png"))
        elif os.path.exists(label_file):
            os.remove(label_file)

if __name__ == "__main__":
    import yaml
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    convert_masks_to_yolo(cfg['raw_data_dir'], cfg['yolo_data_dir'], cfg['tool_map'])
