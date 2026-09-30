from pathlib import Path
from ultralytics import YOLO
def main():
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "yolo_dataset_manual" / "data.yaml"
    model = YOLO(str(project_dir / "yolo11n.pt"))
    model.train(
        data=str(data_path),
        epochs=100,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        project=str(project_dir / "train" / "runs"),
        name="yolo11n_test",
    )
if __name__ == "__main__":
    main()
