"""HCCR preprocess —— ctypes wrapper for ``common/cpp/libhccr_preprocess.so``。

跟 Android JNI 共用同一份 C 实现(``common/cpp/preprocess.{h,c}``)。
保证训练 / Python demo / Android 三端字节级对齐。

替换原来的 ``common.normalize.normalize`` —— 那个是纯 Python + PIL.Image.BILINEAR,
现在改用 C 版本(算法字节一致,Python 调用通过 ctypes,Android 调用通过 JNI)。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

CANVAS_SIZE = 64
CONTENT_SIZE = 56
FG_THRESHOLD = 220


def _load_lib() -> ctypes.CDLL:
    here = Path(__file__).parent / "cpp"
    candidates = [
        here / "libhccr_preprocess.so",
        here / "libhccr_preprocess.dylib",
        here / "hccr_preprocess.dll",
    ]
    for p in candidates:
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        f"libhccr_preprocess 未找到。先在 {here} 跑 `make` 编译"
    )


_lib = _load_lib()
_lib.hccr_preprocess.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
_lib.hccr_preprocess.restype = ctypes.c_int


def preprocess(gray: np.ndarray) -> np.ndarray:
    """灰度图 (任意尺寸 uint8, 255=白底 0=黑笔画) → float32 [1, 64, 64] 模型输入。

    跟 ``common.normalize.normalize`` + 后续 ``(255 - x) / 255`` 等价(且字节级一致)。
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"期望 2D 灰度图,得到 shape={gray.shape}")
    if not gray.flags["C_CONTIGUOUS"]:
        gray = np.ascontiguousarray(gray)
    h, w = gray.shape
    out = np.empty((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    _lib.hccr_preprocess(
        gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(w),
        ctypes.c_int(h),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out[np.newaxis, :, :]   # [1, 64, 64]


# 对比测试 / 自检 CLI
if __name__ == "__main__":
    import argparse
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--save-preview", type=Path, default=None,
                    help="保存预处理后的 64x64 PNG(stroke 翻回 0=黑笔画 255=白底)")
    args = ap.parse_args()

    img = Image.open(args.image).convert("L")
    arr = np.array(img, dtype=np.uint8)
    out = preprocess(arr)
    print(f"input: {arr.shape}, output: {out.shape} dtype={out.dtype}")
    print(f"output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"output non-zero pixels: {(out > 0.01).sum()}/{out.size}")

    if args.save_preview:
        preview = (255 - (out[0] * 255).clip(0, 255)).astype(np.uint8)
        Image.fromarray(preview, mode="L").save(args.save_preview)
        print(f"saved preview: {args.save_preview}")
