"""HCCR 推理封装(NCNN backend)。

复用训练时的归一化逻辑(src.normalize),保证 demo 输入跟训练分布一致。

预处理流水线:
  rendered_canvas (PIL grayscale, 任意尺寸,255=白底 0=黑笔)
    → src.normalize.normalize  (找前景外接矩形 → 长边 56 → 居中放 64×64)
    → uint8 [64, 64], 仍是 255=白底
    → (255 - x) / 255  → float32 [1, 64, 64], stroke=高 (跟 dataset.py 一致)
    → NCNN forward
"""

from __future__ import annotations

import json
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image

from src.normalize import normalize


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

        with open(charset_path, encoding="utf-8") as f:
            data = json.load(f)
        c2i = data["char_to_idx"]
        self.idx_to_char = {int(i): ch for ch, i in c2i.items()}
        self.num_classes = len(c2i)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL → float32 [1, 64, 64] (stroke 高、bg 低,跟训练时一致)。"""
        gray = image.convert("L")
        arr = np.array(gray, dtype=np.uint8)  # 255=白底, 0=黑笔(canvas 渲染惯例)
        norm_64 = normalize(arr)              # uint8 [64,64], 仍 255=白底
        flipped = (255 - norm_64).astype(np.float32) / 255.0
        return flipped[np.newaxis, :, :]       # [1, 64, 64]

    def predict(self, image: Image.Image, k: int = 10) -> list[tuple[str, float]]:
        x = self.preprocess(image)
        if not x.any():
            return []
        # DEBUG:存预处理后的 64×64 给调试用(stroke 反翻回 0=黑笔画 255=白底,直观)
        try:
            preview = (255 - (x[0] * 255).clip(0, 255)).astype(np.uint8)
            Image.fromarray(preview, mode="L").save("/tmp/python_last_input.png")
        except Exception:
            pass
        mat = ncnn.Mat(x)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed: {ret}")
        logits = np.array(out)
        # softmax 给出概率,方便候选栏显示置信度
        z = logits - logits.max()
        probs = np.exp(z)
        probs /= probs.sum()
        top_idx = np.argsort(-probs)[:k]
        return [(self.idx_to_char[int(i)], float(probs[i])) for i in top_idx]


# CLI 模式:取一张图片做推理(测试用,不开 GUI)
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument(
        "--ncnn-param",
        type=Path,
        default=Path("save/output/mbv2_aug.int8.ncnn.param"),
    )
    ap.add_argument(
        "--ncnn-bin",
        type=Path,
        default=Path("save/output/mbv2_aug.int8.ncnn.bin"),
    )
    ap.add_argument(
        "--charset",
        type=Path,
        default=Path("data/processed/charset.json"),
    )
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rec = HCCRRecognizer(args.ncnn_param, args.ncnn_bin, args.charset)
    img = Image.open(args.image)
    candidates = rec.predict(img, k=args.k)
    print(f"top-{args.k} 候选:")
    for ch, p in candidates:
        print(f"  {ch}  {p:.1%}")
