"""把 src.train 训出的 PyTorch checkpoint 转成 NCNN 模型。

流程:
  1. 加载 ckpt(runs/<run>/best.pt),按 cfg.model 重建模型
  2. load_state_dict + eval mode
  3. torch.jit.trace 用固定输入 shape [1, 1, 64, 64]
  4. 保存 traced .pt
  5. 调 pnnx CLI 转成 .ncnn.param / .ncnn.bin

产出:
  out_dir/<name>.traced.pt
  out_dir/<name>.ncnn.param
  out_dir/<name>.ncnn.bin
  out_dir/<name>.pnnx.param
  out_dir/<name>.pnnx.bin
  out_dir/<name>_pnnx.py
  out_dir/<name>_ncnn.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pnnx
import torch

from src.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="path to best.pt / latest.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", type=str, required=True, help="basename for output files")
    ap.add_argument("--input-shape", type=str, default="1,1,64,64")
    ap.add_argument("--fp16", action="store_true", help="pnnx fp16 模式(默认 FP32)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] 加载 ckpt: {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model_name = cfg.get("model", "mobilenet_v2")
    print(f"[export] model: {model_name}, ckpt epoch: {ck.get('epoch')}, "
          f"acc1: {ck.get('acc1', 'n/a')}")

    # 类数从 charset 推断,但这里偷懒用 ckpt 的 classifier 输出维度
    sd = ck["model"]
    # 找最后一个 Linear 层的 out_features
    classifier_keys = [k for k in sd if k.endswith("classifier.weight") or k.endswith(".1.weight")]
    classifier_keys = [k for k in classifier_keys if "classifier" in k]
    num_classes = sd[classifier_keys[-1]].shape[0] if classifier_keys else 3755
    print(f"[export] num_classes: {num_classes}")

    model = build_model(model_name, num_classes)
    model.load_state_dict(sd)
    model.eval()

    shape = tuple(int(x) for x in args.input_shape.split(","))
    example = torch.randn(*shape)

    # pnnx.export 内部做 trace + convert,一步到位
    base = args.out_dir / args.name
    traced_path = f"{base}.traced.pt"
    print(f"[export] pnnx.export → {base}.ncnn.param/bin (fp16={args.fp16})", flush=True)
    pnnx.export(
        model,
        ptpath=traced_path,
        inputs=example,
        pnnxparam=f"{base}.pnnx.param",
        pnnxbin=f"{base}.pnnx.bin",
        pnnxpy=f"{base}_pnnx.py",
        ncnnparam=f"{base}.ncnn.param",
        ncnnbin=f"{base}.ncnn.bin",
        ncnnpy=f"{base}_ncnn.py",
        fp16=args.fp16,
    )
    print("[export] OK")

    # 整理产物 size
    print("\n[export] 产出文件:")
    for f in sorted(args.out_dir.glob(f"{args.name}*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:40s}  {size_mb:6.2f} MB")


if __name__ == "__main__":
    main()
