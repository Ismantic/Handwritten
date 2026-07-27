"""HWDB 数据集 PyTorch 接口。

读 data/npy/{train,test}/{images,labels}.npy(由 prepare/build_npy.py 生成)。

约定:
  - 原始 npy 是 uint8,白底黑字(255=白底,0=黑笔画)
  - 喂给模型时:翻转使笔画=高值(模型学到的"前景=正信号"),float [0,1]
  - 加 channel 维度变 [1, 64, 64]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class HWDBDataset(Dataset):
    """读 npy memmap;__getitem__ 返回 (image_tensor[1,64,64] float, label int)。"""

    def __init__(
        self,
        npy_dir: str | Path,
        *,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_memory: bool = False,
    ) -> None:
        npy_dir = Path(npy_dir)
        img_path = npy_dir / "images.npy"
        lbl_path = npy_dir / "labels.npy"

        if in_memory:
            self.images = np.load(img_path)
        else:
            self.images = np.load(img_path, mmap_mode="r")
        self.labels = np.load(lbl_path)
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"images/labels 数量不一致: {self.images.shape[0]} vs {self.labels.shape[0]}"
            )
        self.augment = augment

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # uint8 (64,64),memmap → 拷贝一份再处理(避免 dataloader worker 跨进程问题)
        img_u8 = np.array(self.images[idx], dtype=np.uint8)
        # 翻转:255=白底变 0=低,0=黑笔画变 255=高;归一化到 [0,1]
        img = (255 - img_u8).astype(np.float32) / 255.0
        # [1, 64, 64]
        x = torch.from_numpy(img).unsqueeze(0)
        if self.augment is not None:
            x = self.augment(x)
        return x, int(self.labels[idx])
