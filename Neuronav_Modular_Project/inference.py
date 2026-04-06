from ultralytics import YOLO
import os
import sys

def run_demo(weights_path, source_video):
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    model = YOLO(weights_path)
    results = model.predict(
        source=source_video,
        save=True,
        conf=0.25,
        project="Neuronav_Showcase",
        name="Final_Demo"
    )
    print(f"✅ Demo saved to: {results[0].save_dir}")

if __name__ == "__main__":
    # Update this path based on your folder structure in Colab
    BEST_WEIGHTS = "/content/runs/segment/Neuronav_Project/Surgical_Tool_Segmentation3/weights/best.pt"
    TEST_VIDEO = "/content/test_surgery.mp4"
    run_demo(BEST_WEIGHTS, TEST_VIDEO)
