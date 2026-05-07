"""HCCR 训练入口。

最小可用版本:
- 读 data/npy/{train,test} + data/processed/charset.json
- plain_cnn baseline,AdamW + CosineAnnealing + 5 epoch warmup,AMP
- 每 epoch 末在测试集评估 top-1/5/10,保存 best + latest checkpoint
- subset_frac 支持 smoke test(只用部分训练数据)

后续要加:数据增强、TensorBoard、resume、多卡等。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

# 训练长跑,redirect 到文件时默认块缓冲会让我们看不到进度。强制 line buffering。
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common.charset import Charset
from common.dataset import HWDBDataset
from common.models import build_model, count_params


@dataclass
class TrainConfig:
    npy_train: str = "data/npy/train"
    npy_test: str = "data/npy/test"
    charset: str = "data/processed/charset.json"

    model: str = "plain_cnn"

    epochs: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    min_lr: float = 0.0
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    num_workers: int = 4
    amp: bool = True
    augment: bool = False

    subset_frac: float = 1.0
    seed: int = 0

    out_dir: str = "runs/baseline"
    log_every: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_augment():
    """训练时数据增强 —— 仿射 + 擦除。dataset.py 已经把图翻成 笔画=高,背景=0,所以 fill=0。"""
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.85, 1.15),
            fill=0,  # 0 = 背景(我们的图已经 255-x 翻过)
        ),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.10), value=0),
    ])


def make_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, Charset]:
    charset = Charset.load(cfg.charset)
    augment_train = _build_augment() if cfg.augment else None
    train_ds = HWDBDataset(cfg.npy_train, augment=augment_train)
    test_ds = HWDBDataset(cfg.npy_test)
    if cfg.augment:
        print("[train] 数据增强:RandomAffine(±10° / ±15% scale / ±5% translate) + RandomErasing(p=0.2)")

    if cfg.subset_frac < 1.0:
        n = len(train_ds)
        k = max(1, int(n * cfg.subset_frac))
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(n, size=k, replace=False)
        train_ds = Subset(train_ds, idx.tolist())
        print(f"[train] subset_frac={cfg.subset_frac} → {k}/{n} 训练样本")

    # 内存紧时 num_workers > 0 用默认 fork 会 OOM(父进程大,fork 复制开销高)。
    # 用 spawn 启个干净的 Python 子进程,内存占用 ~500MB 而非父进程 GB 级。
    mp_ctx = "spawn" if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_ctx,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        multiprocessing_context=mp_ctx,
    )
    return train_loader, test_loader, charset


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_factor


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    correct1 = correct5 = correct10 = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        _, pred = logits.topk(10, dim=1)
        target = y.unsqueeze(1).expand_as(pred)
        hits = (pred == target)
        correct1 += hits[:, :1].any(1).sum().item()
        correct5 += hits[:, :5].any(1).sum().item()
        correct10 += hits[:, :10].any(1).sum().item()
        total += y.size(0)
    return correct1 / total, correct5 / total, correct10 / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> int:
    model.train()
    n_batches = len(loader)
    t0 = time.time()
    running_loss = 0.0
    running_n = 0
    for it, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # cosine + warmup LR
        lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, cfg.lr, cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_n += y.size(0)
        global_step += 1

        if (it + 1) % cfg.log_every == 0 or (it + 1) == n_batches:
            avg = running_loss / running_n
            elapsed = time.time() - t0
            it_per_s = (it + 1) / elapsed
            print(
                f"[epoch {epoch} {it+1}/{n_batches}] loss={avg:.4f} lr={lr:.2e} "
                f"{it_per_s:.1f} it/s elapsed={elapsed:.1f}s"
            )
    return global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    # 把 dataclass 字段全部映射成 cli 参数
    defaults = TrainConfig()
    for k, v in asdict(defaults).items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"[train] config:\n{json.dumps(asdict(cfg), indent=2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")
    if device.type == "cuda":
        print(f"[train] gpu: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader, charset = make_loaders(cfg)
    print(f"[train] charset: {charset.num_classes} classes")
    print(f"[train] train batches/epoch: {len(train_loader)}, test batches: {len(test_loader)}")

    model = build_model(cfg.model, charset.num_classes).to(device)
    print(f"[train] model={cfg.model} params={count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = len(train_loader) * min(cfg.warmup_epochs, cfg.epochs)

    global_step = 0
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch, cfg, global_step, total_steps, warmup_steps,
        )
        acc1, acc5, acc10 = evaluate(model, test_loader, device)
        print(f"[eval] epoch {epoch}: top1={acc1:.4f} top5={acc5:.4f} top10={acc10:.4f}")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "acc1": acc1, "cfg": asdict(cfg)},
            out_dir / "latest.pt",
        )
        if acc1 > best_acc:
            best_acc = acc1
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "acc1": acc1, "cfg": asdict(cfg)},
                out_dir / "best.pt",
            )
            print(f"[train] new best: {acc1:.4f}")

    print(f"[train] done. best top1: {best_acc:.4f}")


if __name__ == "__main__":
    main()
