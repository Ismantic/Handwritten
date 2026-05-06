"""GNT 二进制格式解析器 (CASIA HWDB1.1)。

GNT 文件由若干样本紧凑拼接而成,每个样本结构:
  [4B sample_size LE uint32]
  [2B tag_code  GB 编码,2 字节字符]
  [2B width  LE uint16]
  [2B height LE uint16]
  [width * height B  灰度位图,uint8,255=白底,0=黑笔画]

样本之间无 magic / 边界标记,只能顺序读。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, Iterable

import numpy as np


_HEADER_FMT = "<I 2s H H"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 10


def _decode_tag(tag: bytes) -> str:
    """tag_code 是 GB 编码的 2 字节;直接 decode。

    HWDB1.1 训练集里少数样本是非 GB2312 字符(GBK / 二级字),decode 可能抛错。
    解析阶段不过滤 —— 让上游 build_charset 决定怎么处理。
    """
    try:
        return tag.decode("gb2312")
    except UnicodeDecodeError:
        # 回退到 gbk(覆盖二级字 / 部分 GBK 扩展)
        try:
            return tag.decode("gbk")
        except UnicodeDecodeError:
            # 最后兜底:返回十六进制字符串,build_charset 会跳过
            return f"<HEX:{tag.hex()}>"


def iter_gnt_file(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """遍历单个 .gnt,yield (char, bitmap_HxW_uint8)。"""
    with open(path, "rb") as f:
        while True:
            header = f.read(_HEADER_SIZE)
            if not header:
                return
            if len(header) < _HEADER_SIZE:
                raise ValueError(f"{path}: 末尾 header 截断,残余 {len(header)} 字节")
            sample_size, tag, width, height = struct.unpack(_HEADER_FMT, header)
            expected = _HEADER_SIZE + width * height
            if sample_size != expected:
                raise ValueError(
                    f"{path}: sample_size={sample_size} 与 header+w*h={expected} 不符"
                )
            buf = f.read(width * height)
            if len(buf) != width * height:
                raise ValueError(f"{path}: 位图截断,期望 {width*height} 字节,实读 {len(buf)}")
            bitmap = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            char = _decode_tag(tag)
            yield char, bitmap


def iter_gnt_dir(gnt_dir: Path) -> Iterator[tuple[str, np.ndarray, str]]:
    """遍历目录下所有 .gnt,yield (char, bitmap, source_filename)。"""
    files = sorted(Path(gnt_dir).glob("*.gnt"))
    if not files:
        raise FileNotFoundError(f"目录下没有 .gnt: {gnt_dir}")
    for fp in files:
        for char, bitmap in iter_gnt_file(fp):
            yield char, bitmap, fp.name


def count_samples(gnt_dir: Path) -> int:
    """快速统计样本数(不读位图,只跳 header)。"""
    total = 0
    for fp in sorted(Path(gnt_dir).glob("*.gnt")):
        with open(fp, "rb") as f:
            while True:
                header = f.read(_HEADER_SIZE)
                if not header:
                    break
                sample_size = struct.unpack("<I", header[:4])[0]
                f.seek(sample_size - _HEADER_SIZE, 1)
                total += 1
    return total


def iter_gnt_file_labels(path: Path) -> Iterator[str]:
    """只读 header,yield 字符。用于快速扫描(免读位图)。"""
    with open(path, "rb") as f:
        while True:
            header = f.read(_HEADER_SIZE)
            if not header:
                return
            sample_size, tag, _w, _h = struct.unpack(_HEADER_FMT, header)
            f.seek(sample_size - _HEADER_SIZE, 1)
            yield _decode_tag(tag)


def iter_gnt_dir_labels(gnt_dir: Path) -> Iterator[str]:
    """目录下所有 .gnt,只 yield 标签字符(快速扫描用)。"""
    for fp in sorted(Path(gnt_dir).glob("*.gnt")):
        yield from iter_gnt_file_labels(fp)


# 简单 CLI:扫一个 gnt 目录,打印基本统计
if __name__ == "__main__":
    import argparse
    import collections

    ap = argparse.ArgumentParser()
    ap.add_argument("gnt_dir", type=Path)
    ap.add_argument("--max-samples", type=int, default=0, help="只看前 N 个样本(0=全部)")
    args = ap.parse_args()

    char_counter: collections.Counter[str] = collections.Counter()
    n_total = 0
    files: set[str] = set()
    for char, bitmap, src in iter_gnt_dir(args.gnt_dir):
        char_counter[char] += 1
        n_total += 1
        files.add(src)
        if args.max_samples and n_total >= args.max_samples:
            break

    print(f"扫了 {len(files)} 个 .gnt 文件,共 {n_total} 个样本")
    print(f"unique 字符数: {len(char_counter)}")
    print("出现频次 top 10:")
    for ch, c in char_counter.most_common(10):
        print(f"  {ch!r}: {c}")
    n_low = sum(1 for c in char_counter.values() if c < 100)
    print(f"低频字符 (<100 次): {n_low}")
