"""随机抽样可视化,人眼检查 GNT 解析是否正确。

产出:`out_dir/sample_grid.png` —— 一张 N×N 网格图,每格上面是原始位图、下面是
解码后的汉字标签。能看出字图对得上即可。

另外把每个原始位图独立存一份(方便单独看),写在 `out_dir/raw/`。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from prepare.gnt_parser import iter_gnt_dir

from src.normalize import normalize

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402


def _pick_cjk_font() -> str | None:
    """系统里挑一个能渲染汉字的字体并注册到 matplotlib。返回 family name 或 None。

    matplotlib 默认不扫 .ttc(TrueType Collection),Linux 上 Noto CJK 多是 .ttc;
    所以先用 fc-match 拿到字体文件路径,再 addfont 强制加载。
    """
    import subprocess

    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}\n%{family[0]}\n", "sans-serif:lang=zh"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = []
    if len(out) >= 2 and Path(out[0]).exists():
        font_path, family = out[0], out[1]
        try:
            font_manager.fontManager.addfont(font_path)
            # addfont 注册的 family 名字与 fc 报告的可能差一截(.ttc 内部子字体名),
            # 用 FontProperties 反查实际可用 family 名
            from matplotlib.font_manager import FontProperties
            fp = FontProperties(fname=font_path)
            return fp.get_name()
        except Exception:
            return family

    # 退路:扫 ttflist 看有没有手动装过的 ttf
    candidates = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
                  "Source Han Sans CN", "SimHei", "Microsoft YaHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnt-dir", type=Path, required=True)
    ap.add_argument("--charset", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64, help="抽样数量(会取最近的完全平方数)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reservoir-size",
        type=int,
        default=20000,
        help="蓄水池采样规模 —— 全量扫太慢,先在前 N 样本里随机选",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    with open(args.charset, encoding="utf-8") as f:
        cs = json.load(f)
    valid_chars = set(cs["char_to_idx"].keys())

    rng = random.Random(args.seed)

    # 蓄水池采样:扫前 reservoir_size 个样本,从中随机挑 n 个
    side = int(math.sqrt(args.n))
    n_pick = side * side
    pool: list[tuple[str, np.ndarray]] = []
    print(f"[viz] 蓄水池采样(前 {args.reservoir_size} 样本中挑 {n_pick})...", flush=True)
    for i, (char, bitmap, _src) in enumerate(iter_gnt_dir(args.gnt_dir)):
        if i >= args.reservoir_size:
            break
        if char not in valid_chars:
            continue
        if len(pool) < n_pick:
            pool.append((char, bitmap.copy()))
        else:
            j = rng.randint(0, i)
            if j < n_pick:
                pool[j] = (char, bitmap.copy())

    if len(pool) < n_pick:
        print(f"[viz] WARN: 池子只有 {len(pool)} 个样本,够不上 {n_pick}")
        n_pick = len(pool)
        side = int(math.sqrt(n_pick))
        pool = pool[: side * side]

    print(f"[viz] 渲染 {len(pool)} 个样本到网格图...")

    cjk_font = _pick_cjk_font()
    if cjk_font is None:
        print("[viz] WARN: 系统里找不到 CJK 字体,标签会显示成方框。")
    else:
        print(f"[viz] 用字体: {cjk_font}")
        plt.rcParams["font.sans-serif"] = [cjk_font]
        plt.rcParams["axes.unicode_minus"] = False

    # 主图:每行展示原图 + 归一化后图(便于对比),两组并排放
    fig, axes = plt.subplots(side, side * 2, figsize=(side * 2.4, side * 1.4))
    if side == 1:
        axes = np.array([[axes[0], axes[1]]])
    for k, (char, bitmap) in enumerate(pool):
        r, c = divmod(k, side)
        ax_raw = axes[r][c * 2]
        ax_norm = axes[r][c * 2 + 1]
        ax_raw.imshow(bitmap, cmap="gray", vmin=0, vmax=255)
        ax_raw.set_title(char, fontsize=8)
        ax_raw.axis("off")
        norm = normalize(bitmap)
        ax_norm.imshow(norm, cmap="gray", vmin=0, vmax=255)
        ax_norm.set_title("→64", fontsize=7)
        ax_norm.axis("off")
    fig.suptitle(f"GNT 抽样: {args.gnt_dir.name} (N={len(pool)},左=原图,右=64×64 归一化)")
    fig.tight_layout()
    out_path = args.out_dir / "sample_grid.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz] 写入 {out_path}")

    # 每个样本单独存一份原图 + 归一化图(命名带标签,便于检查)
    for k, (char, bitmap) in enumerate(pool[:32]):
        from PIL import Image
        Image.fromarray(bitmap, "L").save(raw_dir / f"{k:03d}_{char}_raw.png")
        Image.fromarray(normalize(bitmap), "L").save(raw_dir / f"{k:03d}_{char}_norm.png")
    print(f"[viz] 单图样本写入 {raw_dir}/ (前 32 个)")


if __name__ == "__main__":
    main()
