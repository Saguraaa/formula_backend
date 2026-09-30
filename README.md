# 算式检测与文字识别 Backend

当前入口为 `model.py`，串联 YOLO 与 PaddleOCR。生产权重为根目录 `formula_detector.pt`；根目录 `yolo11n.pt` 仅供训练初始化使用。

在项目根目录、已配置好的 Conda 环境中启动：

```powershell
label-studio-ml start . --port 9090
```

训练代码与说明见 `train/README.md`；训练数据保留在 `yolo_dataset_manual/` 和 `yolo_dataset/`。清理及权重路径修改后，重启 backend 使代码生效。

## 依赖文件说明

推理依赖按「通用 / 后端」拆成三层：

| 文件 | 内容 | 说明 |
| --- | --- | --- |
| `requirements-base.txt` | `label-studio-ml` 框架 + `gunicorn` | 已钉到具体 commit，保证构建可复现 |
| `requirements.txt` | `ultralytics`、`paddleocr`、`paddlex`、`opencv-python-headless` 等 | 通用第三方库，CPU/GPU 通用 |
| `requirements-cpu.txt` / `requirements-gpu.txt` | `torch`、`torchvision`、`paddlepaddle` | **二选一，不可同时安装** |

> **⚠️ 不要用一条 `pip install -r requirements-base.txt -r requirements.txt -r requirements-*.txt` 装完。**
>
> `requirements.txt` 里的 `ultralytics` 会把 `torch` 当**传递依赖**自动装上，
> 而 **PyPI 上的 torch 默认就是 CUDA 版**。在 CPU 规格实例上这么装，
> 启动时会得到：
>
> ```
> ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume
> ```
>
> 必须**先把 torch 钉死成目标版本，再装其余依赖**，顺序不能反。

### CPU 环境（Cloud Studio 无 GPU 工作空间，默认走这条）

```bash
# 1) 先装 CPU 版 torch 与 torchvision。
#    --index-url（不是 --extra-index-url）确保只从 CPU 索引取包。
#    不钉小版本号：2.14.0 的 PyPI 构建是 CUDA 版，CPU 索引上不一定有同号 wheel，
#    钉死会导致「找不到匹配版本」。torch 与 torchvision 交给 pip 配对最稳。
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 2) 再装框架与通用依赖
pip install -r requirements-base.txt -r requirements.txt

# 3) 最后装 CPU 版 paddlepaddle
pip install paddlepaddle==3.0.0
```

**如果之前已经误装了 CUDA 版 torch**，第 1 步必须加 `--force-reinstall`，
否则 pip 认为 `2.14.0+cpu` 不高于已装的 `2.14.0+cu130` 而跳过替换：

```bash
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision
```

安装后务必确认装的是 CPU 版（版本号应带 `+cpu` 后缀，且 `cuda.is_available()` 为 False）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### GPU 环境（需 `nvidia-smi` 可用且驱动支持 CUDA 13.0）

本项目当前部署目标无 GPU，此节仅在换到 GPU 实例时参考。

```bash
# 1) torch 与 paddlepaddle-gpu 来自不同索引，必须分成两条命令
pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.14.0+cu130 torchvision==0.29.0+cu130

# 2) 框架与通用依赖
pip install -r requirements-base.txt -r requirements.txt

# 3) PaddlePaddle GPU 版（仅飞桨官方索引有，不在 PyPI）
pip install --index-url \
    https://www.paddlepaddle.org.cn/packages/nightly/cu126/ \
    paddlepaddle-gpu==3.0.0.dev20250717
```

本地开发环境（`D:\conda\envs\labelstudioml`，Windows + RTX 5080）用的是 GPU 组合。
注意**本地能跑不代表 Linux 能跑**：Windows 版 torch 把 CUDA 运行库直接打包在
`torch/lib` 内，而 Linux 版 `libtorch_cuda.so` 依赖外部 `nvidia-*-cu1x` 包，
两者对环境的假设完全不同。

`model.py` 会按实际硬件自动选择推理设备：检测到 CUDA 就用 GPU，否则回落到 CPU，
因此同一份代码在两种实例上都能直接运行，无需改代码。

### 环境自检

装完或启动报错时，先跑只读诊断脚本，它会一次性打印解释器、包版本、
CUDA 库来源，并复现真实报错：

```bash
python check_env.py
```

## 部署到 Cloud Studio

### 方式一：直接在 Python 环境运行

1. 在 Cloud Studio 用「从 Git 仓库导入」创建工作空间，选择算力规格。
   **先执行 `nvidia-smi` 确认有无 GPU**（当前部署目标无 GPU，走 CPU 那套命令）。
2. 按上节命令装依赖（注意顺序：先 torch，再 requirements，最后 paddle），
   然后启动服务：

```bash
python _wsgi.py --host 0.0.0.0 --port 9090
```

   若报 `ncclCommResume`，说明装的是 CUDA 版 torch 而机器没有 GPU，
   按 CPU 流程重装，并注意必须加 `--force-reinstall`。
   启动日志应出现「未检测到可用 CUDA，YOLO 将使用 CPU 推理」。

3. 打开左侧「端口插件」，找到 9090 端口卡片，点「查看预览」或「复制分享链接」，
   得到形如 `https://<工作空间ID>--9090.<区域>.<域名>` 的公网地址。
4. 把该 HTTPS 地址填进 Label Studio 的 `Settings -> Machine Learning -> Add Model`。
   **不要填 `localhost`**，它指向的是你自己的浏览器而非服务所在容器。

关键约束：服务必须监听 `0.0.0.0`，端口必须与预览地址中的端口一致，
否则会出现「服务启动了但页面打不开」。`_wsgi.py` 默认已是 `0.0.0.0`。

### 方式二：构建 Docker 镜像（无 GPU 环境下更省事）

```bash
docker compose build
docker compose up
```

镜像固定安装 **CPU 版**深度学习后端，不需要手工区分 `requirements-*` 文件，
也不会出现误装 CUDA 版 torch 的问题，因此在无 GPU 的机器上比方式一更省事。
构建阶段会执行 `warmup.py`，把 PaddleOCR 的检测/识别模型（约 59 MB）
预先下载并固化进镜像 —— 这一步不可省略：PaddleOCR 默认在**首次推理时**才联网下载模型，
留到运行期会导致第一个请求阻塞数分钟（Cloud Studio 预览必然超时）。

前提：工作空间需支持 Docker。若不支持，用方式一并严格按 CPU 顺序安装。

### 部署注意事项

- **环境变量**：`LABEL_STUDIO_URL` 与 `LABEL_STUDIO_API_KEY` 需在平台侧配置，
  不要提交进仓库。若 Label Studio 与 backend 不在同一网络，缺失它们会导致
  `get_local_path` 无法拉取图片。
- **鉴权**：端口链接默认为「公开」，而 ML backend 的 `/predict` 无鉴权。
  建议设置 `BASIC_AUTH_USER` / `BASIC_AUTH_PASS`，或把端口权限改为「仅自己可见」。
- **运行时长**：Cloud Studio 工作空间会定时关机且按机时计费，适合课程演示与联调，
  不适合当作 7×24 生产服务。
- **健康检查**：服务提供 `GET /health`，返回 `{"status":"UP"}`，可用于探活。
- **CPU 推理耗时**：无 GPU 时 YOLO 与 PaddleOCR 都跑在 CPU 上，单张图片明显更慢。
  `model.py` 已用 `inference_lock` 串行化推理，单进程不会并发争抢。
  若日志出现 `NUMEXPR_MAX_THREADS` 提示或多进程互相抢核，可显式限制线程数
  （数值按实际核数调整，不是越大越快）：
  `export OMP_NUM_THREADS=4`

### 常见报错排查

先跑一次诊断脚本，它会打印解释器/包版本/CUDA 库来源并复现真实报错：

```bash
python check_env.py
```

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `libtorch_cuda.so: undefined symbol: ncclCommResume` | 装的是 CUDA 版 torch，但机器没有可用 GPU（PyPI 上的 torch 默认即 CUDA 版，`ultralytics` 会把它作为传递依赖装上） | 改装 CPU 版：`pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch torchvision`。`--force-reinstall` 不可省，否则 pip 认为 `+cpu` 不高于已装的 `+cu130` 而跳过 |
| `ImportError: libGL.so.1: cannot open shared object file` | 镜像内最终生效的 cv2 是非 headless 版（`paddlex[ocr-core]` 依赖 `opencv-contrib-python`），而 `python:*-slim` 不含 `libGL.so.1` | Dockerfile 已安装 `libgl1` 与 `libglib2.0-0`；若是自行搭建环境，手工补装这两个系统库即可 |
| `No module named torch` | 依赖没装完，或装到了别的解释器 | 用 `python -c "import sys;print(sys.executable)"` 确认解释器，再按上文顺序装 |
| 端口预览打不开 | 服务没监听 `0.0.0.0`，或端口与预览地址不一致 | 确认启动参数为 `--host 0.0.0.0 --port 9090`，且预览地址里也是 9090 |
| 首个请求超时 | PaddleOCR 正在联网下载模型（约 59 MB） | 先跑 `python warmup.py` 预热，或改用 Docker 方式（构建期已预热） |

安装是否正确，可用这两条快速验证（不必等启动）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# CPU 版应输出类似 2.x.x+cpu False；若输出 ...+cu130 True 则装错了
```

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
