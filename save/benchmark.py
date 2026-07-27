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

from src.charset import Charset
from src.dataset import HWDBDataset
from src.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ncnn-param", type=Path, required=True)
    ap.add_argument("--ncnn-bin", type=Path, required=True)
    ap.add_argument("--ncnn-fp16-param", type=Path, default=None)
    ap.add_argument("--ncnn-fp16-bin", type=Path, default=None)
    ap.add_argument("--ncnn-int8-param", type=Path, default=None)
    ap.add_argument("--ncnn-int8-bin", type=Path, default=None)
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

    # NCNN FP32 模型
    print(f"[bench] 加载 NCNN FP32: {args.ncnn_param}", flush=True)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = args.ncnn_threads
    net.load_param(str(args.ncnn_param))
    net.load_model(str(args.ncnn_bin))

    # NCNN FP16 模型(可选)
    net_fp16 = None
    if args.ncnn_fp16_param and args.ncnn_fp16_bin:
        print(f"[bench] 加载 NCNN FP16: {args.ncnn_fp16_param}", flush=True)
        net_fp16 = ncnn.Net()
        net_fp16.opt.use_vulkan_compute = False
        net_fp16.opt.num_threads = args.ncnn_threads
        net_fp16.load_param(str(args.ncnn_fp16_param))
        net_fp16.load_model(str(args.ncnn_fp16_bin))

    # NCNN INT8 模型(可选)
    net_int8 = None
    if args.ncnn_int8_param and args.ncnn_int8_bin:
        print(f"[bench] 加载 NCNN INT8: {args.ncnn_int8_param}", flush=True)
        net_int8 = ncnn.Net()
        net_int8.opt.use_vulkan_compute = False
        net_int8.opt.num_threads = args.ncnn_threads
        net_int8.load_param(str(args.ncnn_int8_param))
        net_int8.load_model(str(args.ncnn_int8_bin))

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
    fp16_correct = 0
    int8_correct = 0
    agree_pt_nc = 0
    agree_pt_fp16 = 0
    agree_pt_int8 = 0
    pt_times: list[float] = []
    ncnn_times: list[float] = []
    fp16_times: list[float] = []
    int8_times: list[float] = []

    with torch.no_grad():
        for k, idx in enumerate(indices):
            x_tensor, y = test_ds[int(idx)]   # [1, 64, 64] float
            x_np = x_tensor.numpy()

            # PyTorch
            x_pt = x_tensor.unsqueeze(0)
            t0 = time.perf_counter()
            logits_pt = pt_model(x_pt)
            pt_times.append(time.perf_counter() - t0)
            pt_pred = int(logits_pt.argmax(1).item())

            # NCNN FP32
            mat = ncnn.Mat(x_np)
            ex = net.create_extractor()
            ex.input("in0", mat)
            t0 = time.perf_counter()
            ret, out = ex.extract("out0")
            ncnn_times.append(time.perf_counter() - t0)
            if ret != 0:
                raise RuntimeError(f"ncnn extract failed: {ret}")
            nc_pred = int(np.array(out).argmax())

            # NCNN FP16(可选)
            fp16_pred = None
            if net_fp16 is not None:
                mat16 = ncnn.Mat(x_np)
                ex16 = net_fp16.create_extractor()
                ex16.input("in0", mat16)
                t0 = time.perf_counter()
                ret16, out16 = ex16.extract("out0")
                fp16_times.append(time.perf_counter() - t0)
                if ret16 != 0:
                    raise RuntimeError(f"ncnn fp16 extract failed: {ret16}")
                fp16_pred = int(np.array(out16).argmax())

            # NCNN INT8(可选)
            int8_pred = None
            if net_int8 is not None:
                mat8 = ncnn.Mat(x_np)
                ex8 = net_int8.create_extractor()
                ex8.input("in0", mat8)
                t0 = time.perf_counter()
                ret8, out8 = ex8.extract("out0")
                int8_times.append(time.perf_counter() - t0)
                if ret8 != 0:
                    raise RuntimeError(f"ncnn int8 extract failed: {ret8}")
                int8_pred = int(np.array(out8).argmax())

            if pt_pred == y:
                pt_correct += 1
            if nc_pred == y:
                ncnn_correct += 1
            if pt_pred == nc_pred:
                agree_pt_nc += 1
            if fp16_pred is not None:
                if fp16_pred == y:
                    fp16_correct += 1
                if pt_pred == fp16_pred:
                    agree_pt_fp16 += 1
            if int8_pred is not None:
                if int8_pred == y:
                    int8_correct += 1
                if pt_pred == int8_pred:
                    agree_pt_int8 += 1

            if (k + 1) % 1000 == 0:
                msg = (f"  [{k+1}/{n_pick}] pt={pt_correct/(k+1):.4f} "
                       f"ncnn={ncnn_correct/(k+1):.4f}")
                if net_fp16 is not None:
                    msg += f" fp16={fp16_correct/(k+1):.4f}"
                if net_int8 is not None:
                    msg += f" int8={int8_correct/(k+1):.4f}"
                print(msg, flush=True)

    n = n_pick
    pt_acc = pt_correct / n
    nc_acc = ncnn_correct / n
    pt_ms = sum(pt_times) / len(pt_times) * 1000
    nc_ms = sum(ncnn_times) / len(ncnn_times) * 1000

    print()
    print(f"=== 结果({n} samples,NCNN threads={args.ncnn_threads})===")
    print(f"{'Backend':<14} {'top-1':>8}  {'latency':>10}  {'agree w/ PT':>12}")
    print(f"{'PyTorch':<14} {pt_acc:>8.4f}  {pt_ms:>8.2f} ms  {'-':>12}")
    print(f"{'NCNN FP32':<14} {nc_acc:>8.4f}  {nc_ms:>8.2f} ms  {agree_pt_nc/n:>12.4f}")
    if net_fp16 is not None:
        f16_acc = fp16_correct / n
        f16_ms = sum(fp16_times) / len(fp16_times) * 1000
        print(f"{'NCNN FP16':<14} {f16_acc:>8.4f}  {f16_ms:>8.2f} ms  {agree_pt_fp16/n:>12.4f}")
    if net_int8 is not None:
        i8_acc = int8_correct / n
        i8_ms = sum(int8_times) / len(int8_times) * 1000
        print(f"{'NCNN INT8':<14} {i8_acc:>8.4f}  {i8_ms:>8.2f} ms  {agree_pt_int8/n:>12.4f}")


if __name__ == "__main__":
    main()
