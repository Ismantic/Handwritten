"""把 checkpoint、NCNN 权重和真实推理代码导出成可上传 HF 的目录。

    python -m save.export                    # 导出全部发布
    python -m save.export Handwritten        # 导出单个
    python -m save.export --code-only        # 只刷新代码、模型卡和 metadata
    python -m save.export --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from save import cards
from save.releases import RELEASES

ROOT = Path(__file__).resolve().parents[1]
SAVE = Path(__file__).resolve().parent
ASSETS = SAVE / "assets"
DEFAULT_OUT = SAVE / "releases"


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        sys.exit(f"缺少发布源文件:{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(out: Path) -> None:
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _copy_code(out: Path) -> None:
    _copy(ROOT / "src/models.py", out / "model.py")
    _copy(ROOT / "src/normalize.py", out / "normalize.py")
    _copy(ASSETS / "inference.py", out / "inference.py")
    _copy(ASSETS / "example.py", out / "example.py")


def export_release(name: str, spec: dict, out_root: Path, code_only: bool) -> Path:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / spec["checkpoint"]
    ncnn_param = ROOT / spec["ncnn_param"]
    ncnn_bin = ROOT / spec["ncnn_bin"]
    charset = ROOT / spec["charset"]

    if not code_only:
        _copy(checkpoint, out / "best.pt")
        _copy(ncnn_param, out / "ncnn/model.ncnn.param")
        _copy(ncnn_bin, out / "ncnn/model.ncnn.bin")
        _copy(charset, out / "ncnn/charset.json")
    elif not (out / "best.pt").exists():
        sys.exit(f"{out} 尚未完整导出,--code-only 无法刷新")

    _copy_code(out)
    (out / "README.md").write_text(cards.model_card(name, spec), encoding="utf-8")

    published = [
        "best.pt",
        "ncnn/model.ncnn.param",
        "ncnn/model.ncnn.bin",
        "ncnn/charset.json",
    ]
    missing = [relative for relative in published if not (out / relative).exists()]
    if missing:
        sys.exit(f"{out} 缺少发布权重:{missing}")
    metadata = {
        "name": name,
        "source_checkpoint": spec["checkpoint"],
        "source_ncnn_param": spec["ncnn_param"],
        "source_ncnn_bin": spec["ncnn_bin"],
        "source_charset": spec["charset"],
        "model": spec["model"],
        "classes": spec["classes"],
        "metrics": spec["metrics"],
        "sha256": {relative: _sha256(out / relative) for relative in published},
    }
    (out / "release_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _clean(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, spec in RELEASES.items():
            checkpoint = ROOT / spec["checkpoint"]
            exported = args.out / name / "best.pt"
            print(
                f"{'✓' if checkpoint.exists() else '✗'} source  "
                f"{'✓' if exported.exists() else '✗'} release  {name}"
            )
        return

    names = args.names or list(RELEASES)
    unknown = [name for name in names if name not in RELEASES]
    if unknown:
        sys.exit(f"未知发布名:{unknown};可用:{list(RELEASES)}")
    for name in names:
        out = export_release(name, RELEASES[name], args.out, args.code_only)
        size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        action = "刷新代码" if args.code_only else "导出"
        print(f"✓ {name} {action} → {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
