"""位图 → 64×64 归一化。

输入:HWDB GNT 原始灰度位图(uint8,255=白底,0=黑笔画,任意宽高)
输出:归一化的 64×64 uint8 灰度图,**保持原始规约**(255=白底,0=黑笔画)

为什么保持白底黑字:推理时手机端把用户笔迹渲染成"白纸黑笔"喂模型,训练数据
和推理输入分布对齐;若要反转让前景=高值(MNIST 风格),在 dataloader 用
`255 - x` 即可。

流程:
  1. 阈值化找笔画像素的最小外接矩形(笔画 = 像素值低)
  2. 等比缩放,长边对齐到 56 像素
  3. 居中放到 64×64 画布(白底,周围 4 像素白边距)
"""

from __future__ import annotations

import numpy as np
from PIL import Image


CANVAS_SIZE = 64
CONTENT_SIZE = 56
MARGIN = (CANVAS_SIZE - CONTENT_SIZE) // 2  # 4
FG_THRESHOLD = 220  # 像素 < 阈值算前景(笔画);GNT 背景 ≈ 255,笔画值偏低


def normalize(bitmap: np.ndarray) -> np.ndarray:
    """归一化到 64×64 uint8,白底黑字(255=白,0=黑)。"""
    if bitmap.ndim != 2:
        raise ValueError(f"期望 2D 位图,得到 shape={bitmap.shape}")

    # 1. 找前景(笔画)外接矩形
    fg = bitmap < FG_THRESHOLD
    if not fg.any():
        # 全白(空样本),返回纯白画布
        return np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)
    ys, xs = np.where(fg)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = bitmap[y0:y1, x0:x1]

    # 2. 等比缩放,长边对齐到 CONTENT_SIZE
    h, w = cropped.shape
    scale = CONTENT_SIZE / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray(cropped, mode="L")
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    resized = np.asarray(pil, dtype=np.uint8)

    # 3. 居中放到 64×64 画布,白底
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)
    off_y = (CANVAS_SIZE - new_h) // 2
    off_x = (CANVAS_SIZE - new_w) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


if __name__ == "__main__":
    # 自测:造一个 80×100 的白底图,中间一块黑笔画,跑归一化
    src = np.full((100, 80), 255, dtype=np.uint8)
    src[20:80, 15:65] = 50  # 一块笔画(低值)
    out = normalize(src)
    print(f"input shape: {src.shape}, output shape: {out.shape}")
    print(f"output range: [{out.min()}, {out.max()}]")
    print(f"前景像素数(< {FG_THRESHOLD}): {(out < FG_THRESHOLD).sum()}")
