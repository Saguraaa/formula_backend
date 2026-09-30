# 算式检测与文字识别 Backend

当前入口为 `model.py`，串联 YOLO 与 PaddleOCR。生产权重为根目录 `formula_detector.pt`；根目录 `yolo11n.pt` 仅供训练初始化使用。

在项目根目录、已配置好的 Conda 环境中启动：

```powershell
label-studio-ml start . --port 9090
```

训练代码与说明见 `train/README.md`；训练数据保留在 `yolo_dataset_manual/` 和 `yolo_dataset/`。清理及权重路径修改后，重启 backend 使代码生效。

## 依赖文件说明

推理依赖按「通用 / 后端」拆成三层，安装时按顺序全部装上：

| 文件 | 内容 | 说明 |
| --- | --- | --- |
| `requirements-base.txt` | `label-studio-ml` 框架 + `gunicorn` | 已钉到具体 commit，保证构建可复现 |
| `requirements.txt` | `ultralytics`、`paddleocr`、`paddlex`、`opencv-python-headless` 等 | 通用第三方库，CPU/GPU 通用 |
| `requirements-cpu.txt` 或 `requirements-gpu.txt` | `torch`、`torchvision`、`paddlepaddle` | **二选一，不可同时安装** |

本地开发环境（`D:\conda\envs\labelstudioml`，Python 3.12.14）用的是
GPU 组合：`torch 2.14.0+cu130` + `paddlepaddle-gpu 3.0.0.dev20250717`。

```powershell
# CPU 环境（无 NVIDIA 显卡）
pip install -r requirements-base.txt -r requirements.txt -r requirements-cpu.txt

# GPU 环境（需驱动支持 CUDA 13.0）
pip install -r requirements-base.txt -r requirements.txt -r requirements-gpu.txt
```

`model.py` 会按实际硬件自动选择推理设备：检测到 CUDA 就用 GPU，否则回落到 CPU，
因此同一份代码可直接部署到 CPU 实例，无需改代码。

## 部署到 Cloud Studio

### 方式一：直接在 Python 环境运行（推荐用于 GPU 规格）

1. 在 Cloud Studio 用「从 Git 仓库导入」创建工作空间，选择所需算力规格
   （GPU 规格见官方计费页；纯演示用 CPU 规格也可，只是推理更慢）。
2. 在终端安装依赖并启动服务：

```bash
pip install -r requirements-base.txt -r requirements.txt -r requirements-gpu.txt
python _wsgi.py --host 0.0.0.0 --port 9090
```

3. 打开左侧「端口插件」，找到 9090 端口卡片，点「查看预览」或「复制分享链接」，
   得到形如 `https://<工作空间ID>--9090.<区域>.<域名>` 的公网地址。
4. 把该 HTTPS 地址填进 Label Studio 的 `Settings -> Machine Learning -> Add Model`。
   **不要填 `localhost`**，它指向的是你自己的浏览器而非服务所在容器。

关键约束：服务必须监听 `0.0.0.0`，端口必须与预览地址中的端口一致，
否则会出现「服务启动了但页面打不开」。`_wsgi.py` 默认已是 `0.0.0.0`。

### 方式二：构建 Docker 镜像

```bash
docker compose build
docker compose up
```

镜像默认安装 **CPU 版**深度学习后端。原因是容器内默认拿不到宿主 GPU
（需要 nvidia-container-toolkit 做 GPU 透传），而 CPU 版在 GPU 实例上也能正常运行。
需要 GPU 加速时建议用方式一。

镜像构建阶段会执行 `warmup.py`，把 PaddleOCR 的检测/识别模型（约 59 MB）
预先下载并固化进镜像。这一步不可省略：PaddleOCR 默认在**首次推理时**才联网下载模型，
留到运行期会导致第一个请求阻塞数分钟（Cloud Studio 预览必然超时）。

### 部署注意事项

- **环境变量**：`LABEL_STUDIO_URL` 与 `LABEL_STUDIO_API_KEY` 需在平台侧配置，
  不要提交进仓库。若 Label Studio 与 backend 不在同一网络，缺失它们会导致
  `get_local_path` 无法拉取图片。
- **鉴权**：端口链接默认为「公开」，而 ML backend 的 `/predict` 无鉴权。
  建议设置 `BASIC_AUTH_USER` / `BASIC_AUTH_PASS`，或把端口权限改为「仅自己可见」。
- **运行时长**：Cloud Studio 工作空间会定时关机且按机时计费，适合课程演示与联调，
  不适合当作 7×24 生产服务。
- **健康检查**：服务提供 `GET /health`，返回 `{"status":"UP"}`，可用于探活。

以下为原 backend 模板的部署说明。

This guide describes the simplest way to start using ML backend with Label Studio.

## Running with Docker (Recommended)

1. Start Machine Learning backend on `http://localhost:9090` with prebuilt image:

```bash
docker-compose up
```

2. Validate that backend is running

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

3. Connect to the backend from Label Studio running on the same host: go to your project `Settings -> Machine Learning -> Add Model` and specify `http://localhost:9090` as a URL.


## Building from source (Advanced)

To build the ML backend from source, you have to clone the repository and build the Docker image:

```bash
docker-compose build
```

## Running without Docker (Advanced)

To run the ML backend without Docker, you have to clone the repository and install all dependencies using pip:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Then you can start the ML backend:

```bash
label-studio-ml start ./dir_with_your_model
```

# Configuration
Parameters can be set in `docker-compose.yml` before running the container.


The following common parameters are available:
- `BASIC_AUTH_USER` - specify the basic auth user for the model server
- `BASIC_AUTH_PASS` - specify the basic auth password for the model server
- `LOG_LEVEL` - set the log level for the model server
- `WORKERS` - specify the number of workers for the model server
- `THREADS` - specify the number of threads for the model server

# Customization

The ML backend can be customized by adding your own models and logic inside the `./dir_with_your_model` directory. 
