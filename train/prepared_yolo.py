"""准备单类别算式检测数据集；运行本文件只准备数据，不训练模型。

读取原始图片和 annotations.json，转换旋转框，以固定种子划分 80/20，
生成 yolo_dataset。原始数据和 Yolo_demo 练习目录不参与写入。
"""

import hashlib
import io
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image


def box_to_yolo(box, image_width, image_height):
    """百分比旋转框 → 四个角 → 水平外接框 → YOLO 归一化坐标。"""
    values = [float(value) for value in (
        image_width, image_height, box["x"], box["y"],
        box["width"], box["height"], box.get("rotation", 0),
    )]
    image_width, image_height, x, y, w, h, rotation = values
    x, y = x / 100 * image_width, y / 100 * image_height
    w, h = w / 100 * image_width, h / 100 * image_height
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [
        (x, y),
        (x + w * cos_t, y + w * sin_t),
        (x + w * cos_t - h * sin_t, y + w * sin_t + h * cos_t),
        (x - h * sin_t, y + h * cos_t),
    ]
    x_min = max(0, min(point[0] for point in corners))
    y_min = max(0, min(point[1] for point in corners))
    x_max = min(image_width, max(point[0] for point in corners))
    y_max = min(image_height, max(point[1] for point in corners))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("框裁剪到图片范围后为空，请检查原始标注")
    return (
        (x_min + x_max) / 2 / image_width,
        (y_min + y_max) / 2 / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    )
def load_annotations(annotations_path):
    with annotations_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return records
def main():
    project_dir = Path(__file__).resolve().parents[1]
    source_dir = project_dir.parent / "实验2数据" / "train"
    annotation_path = source_dir / "annotations.json"
    records = load_annotations(annotation_path)
    records.sort(key=lambda record: record["image"].casefold())
    random.Random(42).shuffle(records)
    # 按整张图片划分
    train_records = records[:80]
    val_records = records[80:]
    output_dir = project_dir / "yolo_dataset_manual"

    export_split(train_records, "train", source_dir, output_dir)
    export_split(val_records, "val", source_dir, output_dir)
    yaml_text = (
        f'path: "{output_dir.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: formula\n"
    )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print("数据配置已保存到：", yaml_path)
    print("全部图片：", len(records))
    print("训练图片：", len(train_records))
    print("验证图片：", len(val_records))
def export_split(records, split_name, source_dir, output_dir):
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        image_name = record["image"]
        # 复制原图，保留文件名
        shutil.copy2(
            source_dir / "images" / image_name,
            image_dir / image_name,
        )
        # 例如 44.JPG 对应 44.txt
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for box in record["boxes"]:
                cx, cy, bw, bh = box_to_yolo(
                    box,
                    record["original_width"],
                    record["original_height"],
                )
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"{split_name}：已导出 {len(records)} 张图片及对应标签")
if __name__ == "__main__":
    main()
