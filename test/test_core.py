from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from src.charset import Charset
    from src.dataset import HWDBDataset
    from src.models import build_model
    from src.normalize import CANVAS_SIZE, normalize
    from src.train import cosine_warmup_lr
except ModuleNotFoundError:
    np = torch = None


@unittest.skipIf(torch is None, "install requirements.txt to run core tests")
class CoreTest(unittest.TestCase):
    def test_normalize_shape_and_empty_canvas(self) -> None:
        image = np.full((17, 23), 255, dtype=np.uint8)
        actual = normalize(image)
        self.assertEqual(actual.shape, (CANVAS_SIZE, CANVAS_SIZE))
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(np.all(actual == 255))

    def test_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = np.full((2, 64, 64), 255, dtype=np.uint8)
            images[0, 31:33, 31:33] = 0
            np.save(root / "images.npy", images)
            np.save(root / "labels.npy", np.array([7, 9], dtype=np.uint16))

            dataset = HWDBDataset(root)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 64, 64))
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(label, 7)
            self.assertEqual(float(image.max()), 1.0)

    def test_charset_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "charset.json"
            path.write_text(
                json.dumps({"num_classes": 2, "char_to_idx": {"中": 0, "国": 1}}),
                encoding="utf-8",
            )
            charset = Charset.load(path)
            self.assertEqual(charset.idx_to_char[1], "国")

    def test_plain_model_output_contract(self) -> None:
        model = build_model("plain_cnn", 11).eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 11))

    def test_warmup_and_decay_boundaries(self) -> None:
        self.assertAlmostEqual(cosine_warmup_lr(0, 100, 10, 1e-3), 1e-4)
        self.assertGreater(
            cosine_warmup_lr(10, 100, 10, 1e-3),
            cosine_warmup_lr(99, 100, 10, 1e-3),
        )


if __name__ == "__main__":
    unittest.main()
