"""扫训练集,生成字符表 charset.json。

策略:
  - 全集统计每个字符的出现频次
  - 过滤出 GB2312 一级字(B0-D7 区,3755 个)
  - 输出 char ↔ index 双向映射 + 频次

非一级字(二级字 / GBK 扩展 / 控制符)单独记录到 stats,不进 vocab。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prepare.gnt_parser import iter_gnt_dir

from tqdm import tqdm


def is_gb2312_level1(char: str) -> bool:
    """判断是否 GB2312 一级字(区位码 16-55,GB 编码 0xB0A1 - 0xD7FE)。"""
    if len(char) != 1:
        return False
    try:
        b = char.encode("gb2312")
    except UnicodeEncodeError:
        return False
    if len(b) != 2:
        return False
    return 0xB0 <= b[0] <= 0xD7 and 0xA1 <= b[1] <= 0xFE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    counter: collections.Counter[str] = collections.Counter()
    print(f"[charset] 扫描 {args.gnt_dir}/*.gnt ...", flush=True)
    pbar = tqdm(unit=" samples")
    for char, _bitmap, _src in iter_gnt_dir(args.gnt_dir):
        counter[char] += 1
        pbar.update(1)
    pbar.close()

    total = sum(counter.values())
    level1 = {ch: cnt for ch, cnt in counter.items() if is_gb2312_level1(ch)}
    others = {ch: cnt for ch, cnt in counter.items() if not is_gb2312_level1(ch)}

    # 按频次降序排序后取索引
    chars_sorted = sorted(level1.keys(), key=lambda c: (-level1[c], c))
    char_to_idx = {ch: i for i, ch in enumerate(chars_sorted)}

    print(f"[charset] 总样本: {total}")
    print(f"[charset] unique 字符: {len(counter)}")
    print(f"[charset] GB2312 一级字 (写入): {len(level1)}")
    print(f"[charset] 其它字符 (跳过): {len(others)}")
    if len(level1) != 3755:
        print(f"[charset] WARNING: 一级字数量 {len(level1)} ≠ 3755,数据集可能不完整或编码异常")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "source": str(args.gnt_dir),
        "num_classes": len(char_to_idx),
        "char_to_idx": char_to_idx,
        "freq": level1,
        "skipped": {
            "count": len(others),
            "samples": sum(others.values()),
            "examples": dict(collections.Counter(others).most_common(20)),
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[charset] 写入 {args.out}")


if __name__ == "__main__":
    main()
