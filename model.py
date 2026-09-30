import math
import os
from pathlib import Path
from threading import Lock

import cv2

try:
    import torch
except (ImportError, OSError) as exc:
    # torch 是 ultralytics 的传递依赖，很多环境里会被自动装成 CUDA 版。
    # 在没有 NVIDIA 驱动 / 完整 CUDA 运行时的机器（例如 CPU 规格实例）上，
    # 导入会以 undefined symbol（常见为 ncclCommResume）失败，而原始报错
    # 完全看不出该怎么修，因此这里补一条可操作的提示。
    #
    # 同时捕获 OSError：CPython 通常把扩展模块加载失败包装成 ImportError
    # （见 importlib._bootstrap_external），但该行为属实现细节，
    # 多捕获一种异常没有代价，可避免包装在最坏情况下失效。
    raise ImportError(
        f"导入 torch 失败：{exc}\n"
        "若报错含 undefined symbol（例如 ncclCommResume），说明装的是 CUDA 版 "
        "torch，但它依赖的 CUDA/NCCL 库缺失或版本不匹配。\n"
        "在没有 NVIDIA 显卡的机器上，请改用 CPU 版：\n"
        "  pip install --force-reinstall --index-url "
        "https://download.pytorch.org/whl/cpu torch torchvision\n"
        "详见 README 的「部署到 Cloud Studio」章节。"
    ) from exc

from ultralytics import YOLO
from paddleocr import PaddleOCR

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse


def resolve_device():
    """决定 YOLO 推理使用 GPU 还是 CPU。

    返回 ultralytics 接受的 device 取值：GPU 时为显卡序号 0，否则为 "cpu"。

    原实现写死 device=0，在没有 NVIDIA 显卡的机器上（例如 Cloud Studio 的
    CPU 规格工作空间）会直接抛错。这里改为按实际硬件探测，使同一份代码
    既能在本地 GPU 环境运行，也能部署到 CPU 实例。
    """
    if not torch.cuda.is_available():
        print("未检测到可用 CUDA，YOLO 将使用 CPU 推理。", flush=True)
        return "cpu"

    device_index = 0
    print(
        f"检测到 CUDA 设备 {device_index}："
        f"{torch.cuda.get_device_name(device_index)}，YOLO 将使用 GPU 推理。",
        flush=True,
    )
    return device_index


class NewModel(LabelStudioMLBase):

    def setup(self):
        """初始化当前 backend 实例使用的模型。"""
        self.set("model_version", "yolo-paddleocr-v1")
        project_dir = Path(__file__).resolve().parent
        weight_path = project_dir / "formula_detector.pt"

        if not weight_path.is_file():
            raise FileNotFoundError(f"找不到 YOLO 权重：{weight_path}")

        self.yolo = YOLO(str(weight_path))
        self.device = resolve_device()

        self.ocr = PaddleOCR(
            lang=os.getenv("OCR_LANG", "en"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        # 同一实例收到多个请求时，依次使用模型。
        self.inference_lock = Lock()

        print("YOLO 和 PaddleOCR 初始化完成", flush=True)

    def recognize_formula(self, crop):
        """输入裁剪图，返回文字和 OCR 置信度。"""
        results = self.ocr.predict(crop)

        if not results:
            return "", 0.0

        item = results[0]
        texts = item["rec_texts"]
        scores = item["rec_scores"]

        if not texts:
            return "", 0.0

        # 目前假设每个算式框中只有一段完整文字。
        # 若出现多段文字，先打印出来，避免静默丢弃。
        if len(texts) > 1:
            print(
                f"注意：一个算式框识别出多段文字：{texts}，"
                "当前仅使用第一段，请检查裁剪图。",
                flush=True,
            )

        return texts[0], float(scores[0])

    def predict(self, tasks, context=None, **kwargs):
        """处理任务图片，返回矩形框及对应文字。"""
        predictions = []

        print("收到任务数量：", len(tasks), flush=True)

        for task in tasks:
            # 1. 获取当前任务的本地图片路径
            image_url = task["data"]["image"]
            image_path = self.get_local_path(
                image_url,
                task_id=task.get("id"),
            )

            # 2. 读取一次图片，后续直接传递图像数组
            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(f"无法读取图片：{image_path}")

            image_height, image_width = image.shape[:2]
            label_studio_results = []

            with self.inference_lock:
                # 3. YOLO 检测算式框
                result = self.yolo.predict(
                    source=image,
                    imgsz=640,
                    conf=0.5,
                    device=self.device,
                    save=False,
                )[0]

                print(
                    f"任务 {task.get('id')}："
                    f"检测到 {len(result.boxes)} 个算式框",
                    flush=True,
                )

                # 4. 逐框裁剪并执行 OCR
                for index, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().tolist()
                    )

                    # 将坐标限制在原图范围内
                    x1 = max(0.0, min(float(image_width), x1))
                    y1 = max(0.0, min(float(image_height), y1))
                    x2 = max(0.0, min(float(image_width), x2))
                    y2 = max(0.0, min(float(image_height), y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    left = math.floor(x1)
                    top = math.floor(y1)
                    right = math.ceil(x2)
                    bottom = math.ceil(y2)

                    crop = image[top:bottom, left:right].copy()

                    if crop.size == 0:
                        continue

                    text, ocr_score = self.recognize_formula(crop)
                    region_id = f"region_{index + 1}"

                    # 5. 生成矩形结果
                    rectangle_result = {
                        "id": region_id,
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "original_width": image_width,
                        "original_height": image_height,
                        "image_rotation": 0,
                        "value": {
                            "x": x1 / image_width * 100,
                            "y": y1 / image_height * 100,
                            "width": (
                                (x2 - x1) / image_width * 100
                            ),
                            "height": (
                                (y2 - y1) / image_height * 100
                            ),
                            "rotation": 0,
                            "rectanglelabels": ["算式"],
                        },
                    }

                    label_studio_results.append(rectangle_result)

                    # OCR 为空时保留框，留给人工填写文字
                    if text:
                        text_result = {
                            "id": region_id,
                            "from_name": "text",
                            "to_name": "image",
                            "type": "textarea",
                            "value": {
                                "text": [text],
                            },
                        }
                        label_studio_results.append(text_result)

                    print(
                        f"{region_id}：{text!r}，"
                        f"OCR 置信度：{ocr_score:.4f}",
                        flush=True,
                    )

            # 6. 每张任务图片对应一个预测对象
            # 没有检测框时，也返回该任务的空结果
            predictions.append({
                "model_version": self.get("model_version"),
                "result": label_studio_results,
            })

            print(
                f"任务 {task.get('id')}："
                f"返回 {len(label_studio_results)} 个结果对象",
                flush=True,
            )

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        return
