"""验证 npy 数据集正确性,抽样可视化。

读 images.npy + labels.npy,反查 charset,渲染 N 个样本到 grid 图供人眼复核。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from common.normalize import CANVAS_SIZE

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _register_cjk_font() -> str | None:
    import subprocess
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\n", "sans-serif:lang=zh"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None
    if not out or not Path(out).exists():
        return None
    font_manager.fontManager.addfont(out)
    from matplotlib.font_manager import FontProperties
    return FontProperties(fname=out).get_name()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", type=Path, required=True)
    ap.add_argument("--charset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.charset, encoding="utf-8") as f:
        cs = json.load(f)
    idx_to_char = {int(i): ch for ch, i in cs["char_to_idx"].items()}

    images = np.load(args.npy_dir / "images.npy", mmap_mode="r")
    labels = np.load(args.npy_dir / "labels.npy")
    print(f"[verify] images: shape={images.shape} dtype={images.dtype}")
    print(f"[verify] labels: shape={labels.shape} dtype={labels.dtype}")
    if images.shape[1:] != (CANVAS_SIZE, CANVAS_SIZE):
        raise RuntimeError(f"images shape 末两维不是 (64,64): {images.shape}")
    if images.shape[0] != labels.shape[0]:
        raise RuntimeError(f"images/labels 数量不一致: {images.shape[0]} vs {labels.shape[0]}")
    if images.dtype != np.uint8 or labels.dtype != np.uint16:
        raise RuntimeError(f"dtype 不对: images={images.dtype}, labels={labels.dtype}")

    side = int(math.sqrt(args.n))
    n_pick = side * side
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(images.shape[0]), n_pick))

    samples: list[tuple[str, np.ndarray]] = []
    for idx in indices:
        label_idx = int(labels[idx])
        char = idx_to_char.get(label_idx)
        if char is None:
            raise RuntimeError(f"label_idx={label_idx} 不在 charset 里")
        samples.append((char, np.array(images[idx])))
    print(f"[verify] 抽 {len(samples)} 个,unique 标签 {len({c for c,_ in samples})}")

    fam = _register_cjk_font()
    if fam:
        plt.rcParams["font.sans-serif"] = [fam]
        plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(side, side, figsize=(side * 1.2, side * 1.4))
    if side == 1:
        axes = np.array([[axes]])
    for k, (char, bm) in enumerate(samples):
        r, c = divmod(k, side)
        ax = axes[r][c]
        ax.imshow(bm, cmap="gray", vmin=0, vmax=255)
        ax.set_title(char, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"npy 抽样: {args.npy_dir.name} (N={len(samples)})")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[verify] 写入 {args.out}")


if __name__ == "__main__":
    main()
