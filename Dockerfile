# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9090 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    OCR_LANG=en \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex-cache

# 更新基础系统。
# git 是必需的：requirements-base.txt 通过 git+https 拉取 label-studio-ml。
# curl 用于容器内的健康检查。
#
# libgl1 / libglib2.0-0 是必需的：最终生效的 cv2 是非 headless 版，
# 它动态链接 libGL.so.1 与 libgthread-2.0.so.0，而 python:*-slim 精简镜像
# 不含这些库，缺少时导入 cv2 会直接报：
#   ImportError: libGL.so.1: cannot open shared object file
#
# 为什么无法只靠 Python 层规避：
#   paddleocr -> paddlex[ocr-core] -> opencv-contrib-python==4.10.0.84（非 headless）
#   label-studio-ml -> label-studio-sdk -> opencv-python-headless
#   两个包都写入 site-packages/cv2/，后装者覆盖先装者，无法保证留下哪个。
#   因此这里补齐系统库，使无论最终是哪个版本都能正常导入。
#
# 注意：Debian bookworm 起不再提供 libgl1-mesa-glx，包名是 libgl1。
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y \
        git \
        curl \
        libgl1 \
        libglib2.0-0; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

# 安装框架依赖（label-studio-ml + gunicorn）
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# 安装通用第三方依赖
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# 安装深度学习后端。
# 默认走 CPU 版：容器内拿不到宿主 GPU（除非用 nvidia-container-toolkit 做
# GPU 透传），而 CPU 版同时适用于 Cloud Studio 的 CPU 与 GPU 规格实例。
# 需要 GPU 加速时，建议不用容器，直接在 Cloud Studio 的 GPU 工作空间里按
# README 的「不使用 Docker」方式运行。
COPY requirements-cpu.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-cpu.txt

# 可选测试依赖，仅在 TEST_ENV=true 时安装。
# 注意原实现用 ARG TEST_ENV 放在 FROM 之前，该变量在构建阶段不可见，
# 条件永远为假；这里改为 FROM 之后的 ARG 才真正生效。
ARG TEST_ENV=false
COPY requirements-test.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
        pip install -r requirements-test.txt; \
    fi

# 先复制预热脚本并执行，把 PaddleOCR 模型固化进镜像。
# 放在 COPY . . 之前，使应用代码变更时不必重新下载 59 MB 模型。
# PADDLE_PDX_CACHE_HOME 已在 ENV 中指向 /opt/paddlex-cache。
COPY warmup.py .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    python warmup.py

# 复制应用代码与生产权重。
# .dockerignore 已确保不会带入数据集、训练代码与缓存。
COPY . .

# 数据目录：已存在的镜像层内目录，避免挂载点缺失
RUN mkdir -p /data

EXPOSE 9090

# 健康检查：Cloud Studio 端口插件与编排系统据此判断服务是否就绪。
# start-period 给足时间，因为启动时要加载 YOLO 与 PaddleOCR。
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# --preload：在 fork worker 之前完成一次模型加载，多个 worker 共享同一份内存。
# --timeout 0：推理耗时不可控，禁用 worker 超时，避免长请求被强杀。
CMD ["sh", "-c", "gunicorn --preload --bind :${PORT} --workers ${WORKERS} --threads ${THREADS} --timeout 0 _wsgi:app"]
