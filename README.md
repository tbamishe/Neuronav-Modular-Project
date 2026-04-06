# Neuronav: Real-Time Surgical Tool Segmentation

**Neuronav** is an AI-driven surgical assistance prototype designed to enhance intraoperative awareness. By leveraging deep learning, the system provides pixel-level segmentation of laparoscopic instruments, enabling real-time tracking and proximity analysis to critical anatomical structures.

## 🚀 Key Achievements
* **Precision Segmentation:** Achieved a **Mean Average Precision (mAP) of 0.887**, ensuring high-fidelity tool boundary detection.
* **Optimized Inference:** Powered by **YOLO11-seg (Nano)**, achieving a benchmark latency of **~12ms per frame**, making it viable for live surgical feedback.
* **Robust Data Pipeline:** Engineered a modular preprocessing system to handle multi-class watershed masks from the **CholecSeg8k** dataset.

## 🧠 Methodology
### Data Engineering
The project utilizes the CholecSeg8k dataset. A custom Python pipeline was developed to:
1. Parse high-resolution watershed masks.
2. Filter noisy segmentation artifacts (contours < 100px) to prevent "invalid value" errors during normalization.
3. Convert pixel-wise masks into standardized YOLO polygon formats ($x_n, y_n$).

### Model Architecture
The system uses **YOLO11-seg**, selected for its superior balance between parameter efficiency and spatial accuracy. The model was trained for 25 epochs with a focus on mask-loss reduction to ensure precise fitment around metallic instruments.



## 🛠️ Project Structure
```text
Neuronav_Project/
├── data/
│   ├── raw/            # Original CholecSeg8k images and masks
│   └── processed/      # YOLO-formatted images and labels
├── config.py           # Centralized configuration and hardware settings
├── preprocess.py       # Mask-to-YOLO conversion logic
├── train.py            # Model training & hardware optimization
├── inference.py        # Real-time video/image prediction script
└── requirements.txt    # Environment dependencies
