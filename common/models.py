"""HCCR 模型定义。

baseline:plain CNN(5 个 conv block + GAP + Linear)。
- 输入:  [B, 1, 64, 64]
- 输出:  [B, num_classes] (logits)
- 参数量:约 1.1M(num_classes=3755)
- 不用 dropout(BN 已经有正则效果);如需可在 classifier 前加

约束:
- 只用标准算子(Conv2d / BN / ReLU / MaxPool / Linear)便于后续导出
- 固定输入 shape(避免动态控制流)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlainCNN(nn.Module):
    """5 stage CNN,每 stage 后 MaxPool 下采样一次。

    [1,64,64] → 32 → 32(/2)=32 → 64(/2)=16 → 96(/2)=8 → 128(/2)=4 → 192(/2)=2 → GAP → Linear
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_relu(in_channels, 32),
            nn.MaxPool2d(2),  # 64→32
            _conv_bn_relu(32, 64),
            nn.MaxPool2d(2),  # 32→16
            _conv_bn_relu(64, 96),
            nn.MaxPool2d(2),  # 16→8
            _conv_bn_relu(96, 128),
            nn.MaxPool2d(2),  # 8→4
            _conv_bn_relu(128, 192),
            nn.MaxPool2d(2),  # 4→2
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(192, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


def _build_mobilenet_v3_small_hccr(num_classes: int, in_channels: int = 1) -> nn.Module:
    """torchvision MobileNetV3-Small 改造:
    - 第一层 conv 改成单通道输入(3 → 1)
    - 分类头去掉 576 → 1024 隐层,直接 576 → num_classes(省 ~600K params,3755 类下分类头本身就是大头)
    - 保留 HSwish / SE 块 / depthwise conv,PNNX/NCNN 都原生支持

    输入: [B, 1, 64, 64]   spatial 走 stride 2×5 → 64 → 32 → 16 → 8 → 4 → 2
    输出: [B, num_classes]
    """
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(num_classes=num_classes)
    model.features[0][0] = nn.Conv2d(
        in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(576, num_classes),
    )
    return model


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "plain_cnn":
        return PlainCNN(num_classes)
    if name == "mobilenet_v3_small":
        return _build_mobilenet_v3_small_hccr(num_classes)
    raise ValueError(f"未知模型: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # smoke:确认能 forward,参数量符合预期
    x = torch.randn(2, 1, 64, 64)
    for name in ("plain_cnn", "mobilenet_v3_small"):
        model = build_model(name, num_classes=3755)
        n = count_params(model)
        bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
        y = model(x)
        print(f"{name}: {n:,} params ({n/1e6:.2f}M)  fp32={bytes_/1e6:.2f}MB  "
              f"int8≈{bytes_/4/1e6:.2f}MB  out={tuple(y.shape)}")
