# YOLO 训练目录

从项目根目录运行（使用已安装依赖的 Conda 环境）：

```powershell
python train/train_yolo.py
```

- 训练数据：根目录 `yolo_dataset_manual/data.yaml`，保留原有训练/验证划分。
- 初始权重：根目录 `yolo11n.pt`。
- 新训练输出：`train/runs/`。参数保持为 50 epochs、batch 8、imgsz 640、GPU 0。
- backend 权重：根目录 `formula_detector.pt`，来自原 `runs/yolo11n_test-2/weights/best.pt`。训练不会自动覆盖它。
- `prepared_yolo.py`：数据转换脚本；输入仍是项目上一级的 `实验2数据/train`。运行会重写 `yolo_dataset_manual` 中同名图片和标签，请先备份人工修改的标签。正常训练现有数据不需要运行此脚本。
- `check_yolo.py` 和 `Yolo_demo/`：保留的小型标注检查练习，执行后生成 `train/Yolo_demo/1_check.jpg`。
- `checkpoints/`：保留的旧训练权重；不是 backend 加载路径。
- `reference/`：最终训练的原始参数和指标，历史路径仅用于溯源。

两套数据集 `yolo_dataset` 与 `yolo_dataset_manual` 均保留在根目录，不假设其标签内容完全相同。需要再次训练、比较模型或者修正识别问题时，应保留图片和标签；仅部署推理服务时不需要携带数据集。

> **注意（与 Git 仓库的关系）**：`.gitignore` 只忽略 `images/` 与 `labels/`，因此
> clone 仓库后**拿不到图片和标签**，只能拿到 `data.yaml` 与 `split_manifest.json`。
> 需要重训时请从原始数据重新生成，或单独备份数据集。
> `train/checkpoints/`、`train/runs/`、`train/weight/` 与 `yolo11n.pt` 同样不入库，
> 均可由训练流程重新产生。

清理日期：2026-09-30。已删除旧 runs、outputs、临时裁剪图、测试图片、预测 JSON 及旧实验脚本/测试目录。保留 backend、依赖声明、Docker 配置、IDE 配置与 cache.db。
