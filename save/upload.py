"""把 save/releases 下已验证的发布目录上传到 Hugging Face。

    python -m save.upload --namespace Ismantic --dry-run
    python -m save.upload --namespace Ismantic
    python -m save.upload --namespace Ismantic --code-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from save.releases import RELEASES

DEFAULT_DIR = Path(__file__).resolve().parent / "releases"
IGNORE = ["__pycache__/**", "**/__pycache__/**", "**/*.pyc"]
WEIGHTS = ["best.pt", "ncnn/model.ncnn.bin"]


def _files(folder: Path, code_only: bool) -> list[Path]:
    skipped = set(WEIGHTS) if code_only else set()
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.relative_to(folder).as_posix() not in skipped
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = args.names or [
        name for name in RELEASES if (args.release_dir / name).exists()
    ]
    if not names:
        sys.exit("没有发布目录,先运行 python -m save.export")
    missing = [name for name in names if not (args.release_dir / name).exists()]
    if missing:
        sys.exit(f"尚未导出:{missing}")

    for name in names:
        folder = args.release_dir / name
        repo_id = f"{args.namespace}/{name}"
        files = _files(folder, args.code_only)
        size = sum(path.stat().st_size for path in files)
        print(f"{repo_id} ← {folder} ({size / 1e6:.1f} MB,{len(files)} files)")
        for path in files:
            print(f"  {path.relative_to(folder)}")
        if args.dry_run:
            continue

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        ignore = IGNORE + (WEIGHTS if args.code_only else [])
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            ignore_patterns=ignore,
            commit_message=(
                "Update inference code and model card"
                if args.code_only
                else f"Upload {name} release"
            ),
        )
        print(f"→ https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("--dry-run:未上传任何文件")


if __name__ == "__main__":
    main()
