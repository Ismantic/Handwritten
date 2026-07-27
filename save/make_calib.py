"""为 ncnn2table INT8 校准准备数据。

从训练集 npy 随机抽 N 张,做跟 dataset.py 一致的预处理(翻转 + 归一化),
存成 float32 [1, 64, 64] 的单图 .npy 文件,生成 ncnn2table 用的文件列表。

ncnn2table 用 type=1 模式直接读 npy,跳过 image 解码 + mean/norm 转换。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", type=Path, default=Path("data/npy/train"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True, help="路径列表 txt 输出")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = np.load(args.npy_dir / "images.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(images.shape[0], size=args.n, replace=False).tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, idx in enumerate(indices):
        img_u8 = np.array(images[idx], dtype=np.uint8)
        # dataset.py 里的预处理:翻转(笔画=高)+ 归一化到 [0, 1]
        img = (255 - img_u8).astype(np.float32) / 255.0
        # NCNN 期望 [C, H, W],单通道
        arr = img[np.newaxis, :, :]
        out_path = args.out_dir / f"calib_{i:05d}.npy"
        np.save(out_path, arr)
        paths.append(out_path)

    with open(args.list, "w") as f:
        for p in paths:
            f.write(str(p.absolute()) + "\n")
    print(f"[calib] 写入 {len(paths)} 张 npy 到 {args.out_dir}")
    print(f"[calib] 列表文件: {args.list}")


if __name__ == "__main__":
    main()
