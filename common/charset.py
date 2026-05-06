"""字符表加载/反查。

charset.json 由 data/scripts/build_charset.py 生成,结构见那边。这里只负责加载和
提供快速 char ↔ index 映射,以及 num_classes(模型分类头维度从这取)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass
class Charset:
    """char ↔ index 双向映射 + 类数。"""

    char_to_idx: Mapping[str, int]
    idx_to_char: Mapping[int, str]
    num_classes: int
    freq: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Charset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        c2i = dict(data["char_to_idx"])
        i2c = {int(i): ch for ch, i in c2i.items()}
        return cls(
            char_to_idx=c2i,
            idx_to_char=i2c,
            num_classes=int(data["num_classes"]),
            freq=dict(data.get("freq", {})),
        )

    def __len__(self) -> int:
        return self.num_classes
