from pathlib import Path
from PIL import Image, ImageDraw
root = Path(__file__).resolve().parents[1] / "train" / "Yolo_demo"

image_path = root / "images" / "train" / "1.jpg"
label_path = root / "labels" / "train" / "1.txt"
with Image.open(image_path) as source:
    image = source.convert("RGB")
image_width, image_height = image.size
draw = ImageDraw.Draw(image)
with label_path.open("r") as f:
    for line in f:
        class_id,cx,cy,w,h = map(float, line.split())
        x1=(cx-w/2)*image_width
        y1=(cy-h/2)*image_height
        x2=(cx+w/2)*image_width
        y2=(cy+h/2)*image_height
        draw.rectangle([x1,y1,x2,y2], outline="red",width=2)
output_path=root / "1_check.jpg"
image.save(output_path)
print("已经保存到：", output_path)
