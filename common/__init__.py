"""跨阶段共享代码:数据归一化、字符表、Dataset、模型定义。

数据 pipeline (`data/`)、训练 (`fit/`)、未来的导出阶段都从这里 import。
"""

from common.charset import Charset
from common.normalize import normalize, CANVAS_SIZE, CONTENT_SIZE, FG_THRESHOLD

__all__ = [
    "Charset",
    "normalize",
    "CANVAS_SIZE",
    "CONTENT_SIZE",
    "FG_THRESHOLD",
]
