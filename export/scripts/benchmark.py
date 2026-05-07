"""对比 PyTorch 和 NCNN(FP32)推理一致性 + 延迟。

输入:
  --ckpt:     PyTorch checkpoint(用来跑参考输出)
  --ncnn-param / --ncnn-bin: NCNN 模型
  --npy-dir:  data/npy/test(取测试集子集做精度对比)
  --charset:  charset.json
  --n:        采样数(默认 5000,跑全 22w 太慢)

产出:
  - PyTorch top-1 / NCNN top-1 / 一致率
  - 平均推理延迟(PyTorch CPU、NCNN CPU 单线程)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import ncnn

from common.charset import Charset
from common.dataset import HWDBDataset
from common.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ncnn-param", type=Path, required=True)
    ap.add_argument("--ncnn-bin", type=Path, required=True)
    ap.add_argument("--npy-dir", type=Path, default=Path("data/npy/test"))
    ap.add_argument("--charset", type=Path, default=Path("data/processed/charset.json"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncnn-threads", type=int, default=1)
    args = ap.parse_args()

    # PyTorch 模型
    print(f"[bench] 加载 PyTorch ckpt: {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model_name = cfg.get("model", "mobilenet_v2")
    sd = ck["model"]
    classifier_keys = [k for k in sd if "classifier" in k and "weight" in k and sd[k].dim() == 2]
    num_classes = sd[classifier_keys[-1]].shape[0]
    pt_model = build_model(model_name, num_classes)
    pt_model.load_state_dict(sd)
    pt_model.eval()

    # NCNN 模型
    print(f"[bench] 加载 NCNN: {args.ncnn_param}", flush=True)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = args.ncnn_threads
    net.load_param(str(args.ncnn_param))
    net.load_model(str(args.ncnn_bin))

    # 数据
    test_ds = HWDBDataset(args.npy_dir)
    rng = np.random.default_rng(args.seed)
    n_total = len(test_ds)
    n_pick = min(args.n, n_total)
    indices = rng.choice(n_total, size=n_pick, replace=False)
    print(f"[bench] 测试集采样 {n_pick}/{n_total}", flush=True)

    # 推理
    pt_correct = 0
    ncnn_correct = 0
    agree = 0
    pt_times = []
    ncnn_times = []

    with torch.no_grad():
        for k, idx in enumerate(indices):
            x_tensor, y = test_ds[int(idx)]   # [1, 64, 64] float
            x_np = x_tensor.numpy()           # [1, 64, 64]

            # PyTorch
            x_pt = x_tensor.unsqueeze(0)      # [1, 1, 64, 64]
            t0 = time.perf_counter()
            logits_pt = pt_model(x_pt)
            pt_times.append(time.perf_counter() - t0)
            pt_pred = int(logits_pt.argmax(1).item())

            # NCNN
            mat = ncnn.Mat(x_np)              # ncnn 自动按 [C, H, W] 解析 (1,64,64) -> shape (1,64,64)
            ex = net.create_extractor()
            ex.input("in0", mat)
            t0 = time.perf_counter()
            ret, out = ex.extract("out0")
            ncnn_times.append(time.perf_counter() - t0)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed: {ret}")
            logits_nc = np.array(out)         # [num_classes]
            nc_pred = int(logits_nc.argmax())

            if pt_pred == y:
                pt_correct += 1
            if nc_pred == y:
                ncnn_correct += 1
            if pt_pred == nc_pred:
                agree += 1

            if (k + 1) % 500 == 0:
                print(
                    f"  [{k+1}/{n_pick}] pt acc={pt_correct/(k+1):.4f} "
                    f"ncnn acc={ncnn_correct/(k+1):.4f} agree={agree/(k+1):.4f}",
                    flush=True,
                )

    pt_acc = pt_correct / n_pick
    nc_acc = ncnn_correct / n_pick
    agree_rate = agree / n_pick
    pt_ms = sum(pt_times) / len(pt_times) * 1000
    nc_ms = sum(ncnn_times) / len(ncnn_times) * 1000

    print()
    print(f"=== 结果({n_pick} samples)===")
    print(f"  PyTorch  top-1: {pt_acc:.4f}  avg latency: {pt_ms:.2f} ms")
    print(f"  NCNN FP32 top-1: {nc_acc:.4f}  avg latency: {nc_ms:.2f} ms")
    print(f"  predict 一致率: {agree_rate:.4f}")
    print(f"  延迟比 PT/NCNN: {pt_ms/nc_ms:.2f}x")


if __name__ == "__main__":
    main()
