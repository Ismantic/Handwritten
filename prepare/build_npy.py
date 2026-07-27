"""把 .gnt 数据归一化到 64×64 后存成两个 .npy 文件。

输出布局:
  out_dir/images.npy   shape=(N, 64, 64) uint8  (255=白底, 0=黑笔画)
  out_dir/labels.npy   shape=(N,)        uint16
  out_dir/meta.json    引用的 charset 路径 + sample 数

不在 charset 里的样本(非一级字)直接跳过。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from prepare.gnt_parser import iter_gnt_dir, iter_gnt_dir_labels

from src.normalize import normalize, CANVAS_SIZE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnt-dir", type=Path, required=True)
    ap.add_argument("--charset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    with open(args.charset, encoding="utf-8") as f:
        cs = json.load(f)
    char_to_idx: dict[str, int] = cs["char_to_idx"]
    if len(char_to_idx) > 65535:
        raise ValueError("超过 uint16 容量,需要把 labels 改成 uint32")
    print(f"[npy] charset 类数: {len(char_to_idx)}")

    # Pass 1: 统计有效样本数(只读 header,跳过 bitmap)
    print(f"[npy] Pass 1: 统计样本数(header-only)...", flush=True)
    n_total = 0
    n_valid = 0
    for char in tqdm(iter_gnt_dir_labels(args.gnt_dir), unit=" samples"):
        n_total += 1
        if char in char_to_idx:
            n_valid += 1
    print(f"[npy] 总样本: {n_total}, 有效(在 charset 里): {n_valid}, 跳过: {n_total - n_valid}")

    # 预分配
    images = np.empty((n_valid, CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    labels = np.empty((n_valid,), dtype=np.uint16)

    # Pass 2: 填充
    print(f"[npy] Pass 2: 归一化 + 写入 ndarray...", flush=True)
    i = 0
    for char, bm, _src in tqdm(iter_gnt_dir(args.gnt_dir), total=n_total, unit=" samples"):
        idx = char_to_idx.get(char)
        if idx is None:
            continue
        images[i] = normalize(bm)
        labels[i] = idx
        i += 1
    assert i == n_valid, f"pass1/pass2 计数不一致: {i} vs {n_valid}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    img_path = args.out_dir / "images.npy"
    lbl_path = args.out_dir / "labels.npy"
    meta_path = args.out_dir / "meta.json"

    print(f"[npy] 写入 {img_path} ({images.nbytes / 1e9:.2f} GB)...", flush=True)
    np.save(img_path, images)
    print(f"[npy] 写入 {lbl_path} ({labels.nbytes / 1e6:.2f} MB)...", flush=True)
    np.save(lbl_path, labels)

    meta = {
        "version": 1,
        "source_gnt_dir": str(args.gnt_dir),
        "charset_path": str(args.charset),
        "num_classes": len(char_to_idx),
        "num_samples": int(n_valid),
        "skipped": int(n_total - n_valid),
        "image_shape": [CANVAS_SIZE, CANVAS_SIZE],
        "image_dtype": "uint8",
        "image_convention": "255=white_bg, 0=black_stroke",
        "label_dtype": "uint16",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[npy] 写入 {meta_path}")


if __name__ == "__main__":
    main()
