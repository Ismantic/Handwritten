from __future__ import annotations

import os
import unittest
from pathlib import Path

@unittest.skipUnless(
    os.environ.get("HCCR_REPRODUCE") == "1",
    "set HCCR_REPRODUCE=1 to run the full checkpoint baseline",
)
class ReproduceAccuracyTest(unittest.TestCase):
    def test_mobilenet_v2_baseline(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        from src.charset import Charset
        from src.dataset import HWDBDataset
        from src.models import build_model
        from src.train import evaluate

        checkpoint = Path("runs/mbv2_phase1_aug/best.pt")
        charset_path = Path("data/processed/charset.json")
        dataset_path = Path("data/npy/test")
        for path in (checkpoint, charset_path, dataset_path):
            self.assertTrue(path.exists(), f"missing baseline artifact: {path}")

        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        charset = Charset.load(charset_path)
        model = build_model(saved["cfg"]["model"], charset.num_classes)
        model.load_state_dict(saved["model"])
        loader = DataLoader(HWDBDataset(dataset_path), batch_size=512, num_workers=0)
        top1, _top5, top10 = evaluate(model, loader, torch.device("cpu"))

        self.assertGreaterEqual(top1, 0.9540)
        self.assertGreaterEqual(top10, 0.9950)


if __name__ == "__main__":
    unittest.main()
