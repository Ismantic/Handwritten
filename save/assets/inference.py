"""随 HF 模型发布的自包含 NCNN 推理入口。"""

from __future__ import annotations

import json
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

from normalize import normalize


class HCCRRecognizer:
    def __init__(
        self,
        param_path: str | Path,
        bin_path: str | Path,
        charset_path: str | Path,
        num_threads: int = 4,
    ) -> None:
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))
        data = json.loads(Path(charset_path).read_text(encoding="utf-8"))
        self.idx_to_char = {int(i): ch for ch, i in data["char_to_idx"].items()}

    @classmethod
    def from_pretrained(cls, model_dir: str | Path = ".", num_threads: int = 4):
        root = Path(model_dir)
        return cls(
            root / "ncnn/model.ncnn.param",
            root / "ncnn/model.ncnn.bin",
            root / "ncnn/charset.json",
            num_threads=num_threads,
        )

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        normalized = normalize(gray)
        return ((255 - normalized).astype(np.float32) / 255.0)[None, :, :]

    def predict(self, image: Image.Image, k: int = 10) -> list[tuple[str, float]]:
        x = self.preprocess(image)
        if not x.any():
            return []
        extractor = self.net.create_extractor()
        extractor.input("in0", ncnn.Mat(x))
        ret, out = extractor.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed: {ret}")
        logits = np.asarray(out)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        indices = np.argsort(-probs)[:k]
        return [(self.idx_to_char[int(i)], float(probs[i])) for i in indices]

    def predict_file(self, path: str | Path, k: int = 10) -> list[tuple[str, float]]:
        with Image.open(path) as image:
            return self.predict(image, k=k)
