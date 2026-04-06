from ultralytics import YOLO
import yaml
import os

def run_training():
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. Create data.yaml for YOLO
    yolo_yaml = {
        'path': cfg['yolo_data_dir'],
        'train': 'images',
        'val': 'images',
        'names': {
            0: 'Surgical_Tool_Primary',
            1: 'Surgical_Tool_Secondary',
            2: 'Anatomical_Structure'
        }
    }
    
    with open('dataset_config.yaml', 'w') as f:
        yaml.dump(yolo_yaml, f)

    # 2. Train
    model = YOLO(cfg['model_type'])
    model.train(
        data='dataset_config.yaml',
        epochs=cfg['epochs'],
        imgsz=cfg['imgsz'],
        device=0,
        project=cfg['weights_output_dir'],
        name='Surgical_Tool_Segmentation'
    )

if __name__ == "__main__":
    run_training()
