"""用发布包识别一张白底黑字图片。"""

import argparse
from pathlib import Path

from inference import HCCRRecognizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    model = HCCRRecognizer.from_pretrained(args.model_dir)
    for char, probability in model.predict_file(args.image, k=args.k):
        print(f"{char}\t{probability:.2%}")


if __name__ == "__main__":
    main()
