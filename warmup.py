# ============================================================
# 模型预热脚本
#
# 用途：在镜像构建阶段把 PaddleOCR 的检测/识别模型下载并固化到镜像里。
#
# 背景：PaddleOCR 在首次调用 predict() 时才去联网下载模型
# （本地实测缓存约 59.4 MB，位于 ~/.paddlex/official_models）。
# 如果留到运行期下载：
#   - 容器内第一个推理请求会阻塞数分钟，Cloud Studio 预览必然超时；
#   - 该请求受网络波动影响，失败后表现为“服务起来了但一调用就 500”。
#
# 本脚本把初始化提到构建阶段，并把缓存写进 PADDLE_PDX_CACHE_HOME，
# 由 Dockerfile 复制到运行阶段的 HOME 下。
#
# 直接运行（不经过 Docker）也可用，用于提前把模型缓存到本机：
#   python warmup.py
# ============================================================

import os
import sys
from pathlib import Path

# 让缓存落在固定位置，便于 Dockerfile 精确复制。
# 必须在导入 paddleocr 之前设置，否则不生效。
#
# 默认值区分平台：容器里要固定到 /opt/paddlex-cache 以便复制，
# 本机直接运行时则回落到与 PaddleOCR 原生一致的 ~/.paddlex，
# 否则 Windows 上会尝试写入 \opt\ 而失败。
if "PADDLE_PDX_CACHE_HOME" not in os.environ:
    if sys.platform.startswith("linux"):
        os.environ["PADDLE_PDX_CACHE_HOME"] = "/opt/paddlex-cache"
    else:
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(
            Path.home() / ".paddlex"
        )

from paddleocr import PaddleOCR  # noqa: E402


def main() -> None:
    print("开始预热 PaddleOCR，缓存目录："
          f"{os.environ['PADDLE_PDX_CACHE_HOME']}", flush=True)

    # 参数必须与 model.py 中 NewModel.setup() 完全一致，
    # 否则会下载出另一套模型文件，预热失去意义。
    PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print("PaddleOCR 预热完成。", flush=True)


if __name__ == "__main__":
    main()
