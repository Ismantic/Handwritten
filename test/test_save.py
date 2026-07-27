"""Hugging Face 发布目录的忠实性和自包含性验证。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "save/releases/Handwritten"
SOURCE_CHECKPOINT = ROOT / "runs/mbv2_phase1_aug/best.pt"
SOURCE_PARAM = ROOT / "save/output/mbv2_aug.int8.ncnn.param"
SOURCE_BIN = ROOT / "save/output/mbv2_aug.int8.ncnn.bin"
SOURCE_CHARSET = ROOT / "data/processed/charset.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_in_release(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=RELEASE,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@unittest.skipUnless(RELEASE.exists(), "run `make -C save export` first")
class SaveReleaseTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        for cache in RELEASE.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)

    def test_required_files(self) -> None:
        required = [
            "README.md",
            "best.pt",
            "model.py",
            "normalize.py",
            "inference.py",
            "example.py",
            "release_metadata.json",
            "ncnn/model.ncnn.param",
            "ncnn/model.ncnn.bin",
            "ncnn/charset.json",
        ]
        self.assertEqual(
            [path for path in required if not (RELEASE / path).is_file()],
            [],
        )

    def test_weights_are_byte_identical(self) -> None:
        pairs = [
            (SOURCE_CHECKPOINT, RELEASE / "best.pt"),
            (SOURCE_PARAM, RELEASE / "ncnn/model.ncnn.param"),
            (SOURCE_BIN, RELEASE / "ncnn/model.ncnn.bin"),
            (SOURCE_CHARSET, RELEASE / "ncnn/charset.json"),
        ]
        for source, published in pairs:
            self.assertEqual(sha256(source), sha256(published), source.name)

        metadata = json.loads(
            (RELEASE / "release_metadata.json").read_text(encoding="utf-8")
        )
        for relative, expected in metadata["sha256"].items():
            self.assertEqual(sha256(RELEASE / relative), expected, relative)

    def test_pytorch_checkpoint_loads_inside_release(self) -> None:
        result = run_in_release(
            """
            import torch
            from model import build_model

            checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
            model = build_model(checkpoint["cfg"]["model"], 3755)
            model.load_state_dict(checkpoint["model"], strict=True)
            print(sum(p.numel() for p in model.parameters()))
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "4163243")

    def test_ncnn_inference_runs_inside_release(self) -> None:
        result = run_in_release(
            """
            import numpy as np
            from PIL import Image
            from inference import HCCRRecognizer

            image = np.full((128, 128), 255, dtype=np.uint8)
            image[20:108, 58:70] = 0
            image[58:70, 20:108] = 0
            model = HCCRRecognizer.from_pretrained(".")
            result = model.predict(Image.fromarray(image), k=3)
            assert len(result) == 3
            assert all(len(char) == 1 and 0 <= prob <= 1 for char, prob in result)
            print("OK")
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "OK")


if __name__ == "__main__":
    unittest.main()
