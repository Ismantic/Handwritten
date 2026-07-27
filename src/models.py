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


class GWAP(nn.Module):
    """Global Weighted Average Pooling(Melnyk et al. 2020)。

    学一个空间权重图代替朴素 GAP:
        out[c] = sum_{i,j} W[c, i, j] * x[c, i, j]
    初始化为 1/(H*W) 即等价 GAP,有梯度后偏离学到"哪些位置更重要"。

    可学参数:C × H × W(对 448 ch / 4×4 spatial 是 7168,可忽略)。
    """

    def __init__(self, channels: int, spatial_h: int, spatial_w: int) -> None:
        super().__init__()
        init_val = 1.0 / (spatial_h * spatial_w)
        self.weight = nn.Parameter(
            torch.full((channels, spatial_h, spatial_w), init_val, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, C]
        return (x * self.weight).sum(dim=(2, 3))


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


class MelnykNet(nn.Module):
    """Melnyk-Net(2020 Soft Computing,Melnyk/You/Li)。

    原版 96×96 输入,这里改造为 64×64(我们的部署口径)。
    14 conv(3×3 + BN + ReLU)分 5 stage,4 次 avg pool,GWAP,FC。

    spatial 流动:64 → 32 → 16 → 8 → 4
    通道流动(注意"瓶颈"模式 high-low-high):
        stage 1: 64, 64                        @ 64×64
        stage 2: 96, 64, 96                    @ 32×32
        stage 3: 128, 96, 128                  @ 16×16
        stage 4: 256, 192, 256                 @ 8×8
        stage 5: 448, 256, 448                 @ 4×4

    总参数 ~6.5M(同原版),FP32 ~25 MB,INT8 ~6.5 MB(超 plan 5MB,需要后续考虑)。
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return _conv_bn_relu(in_ch, out_ch)

        self.stage1 = nn.Sequential(block(in_channels, 64), block(64, 64))
        self.stage2 = nn.Sequential(block(64, 96), block(96, 64), block(64, 96))
        self.stage3 = nn.Sequential(block(96, 128), block(128, 96), block(96, 128))
        self.stage4 = nn.Sequential(block(128, 256), block(256, 192), block(192, 256))
        self.stage5 = nn.Sequential(block(256, 448), block(448, 256), block(256, 448))
        self.pool = nn.AvgPool2d(2)
        self.gwap = GWAP(448, 4, 4)
        self.classifier = nn.Linear(448, num_classes)

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
        x = self.stage1(x); x = self.pool(x)
        x = self.stage2(x); x = self.pool(x)
        x = self.stage3(x); x = self.pool(x)
        x = self.stage4(x); x = self.pool(x)
        x = self.stage5(x)  # 4×4
        x = self.gwap(x)
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
    # torchvision 在 mobilenet_v3_small() 构造时跑过 init loop(Conv kaiming_normal,
    # Linear normal(0, 0.01))。我们之后才替换层,新层是 PyTorch 默认 init,
    # stddev 偏大(分类头 2.3x),早期梯度信号偏强 —— 重新走一遍同样的 init。
    for layer in (model.features[0][0], model.classifier[1]):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)
    return model


class MobileNetV2WithGWAP(nn.Module):
    """V2 改造 + 借 Melnyk-Net 的 GWAP 替换 GAP。

    流程:V2 features → GWAP(C × 2 × 2 学习权重)→ Dropout → Linear。
    输入 [B,1,64,64]:V2 5 次 stride 2 下采样 → 2×2 spatial。
    """

    def __init__(
        self, num_classes: int, in_channels: int = 1, last_channel: int = 576
    ) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v2

        base = mobilenet_v2()
        base.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        base.features[18] = nn.Sequential(
            nn.Conv2d(320, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
        )
        self.features = base.features
        # 64 input → 5 次 stride 2 → 2×2 spatial 在末端
        self.gwap = GWAP(last_channel, 2, 2)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(last_channel, num_classes),
        )

        for layer in (
            self.features[0][0],
            self.features[18][0],
            self.features[18][1],
            self.classifier[1],
        ):
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # [B, last_ch, 2, 2]
        x = self.gwap(x)      # [B, last_ch]
        return self.classifier(x)


def _replace_relu6_with_relu(module: nn.Module) -> None:
    """递归把 ReLU6 替换为 ReLU。

    用途:MobileNetV2 默认 ReLU6 在 INT8 量化时(activation 被 clip 到 [0, 6]
    + 1×1 expand 高动态范围 + DW 各通道差异大)精度损失大。换成 ReLU 后
    激活分布更线性,PTQ 友好,文献和实测都验证过。

    FP32 精度通常基本不变(< 0.3 点),换来 INT8 PTQ 损失从 2-3 点 → < 1 点。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU(inplace=child.inplace))
        else:
            _replace_relu6_with_relu(child)


def _build_mobilenet_v2_relu_hccr(
    num_classes: int, in_channels: int = 1, last_channel: int = 576
) -> nn.Module:
    """V2-HCCR + ReLU(替换全部 ReLU6),量化友好版。"""
    model = _build_mobilenet_v2_hccr(num_classes, in_channels, last_channel)
    _replace_relu6_with_relu(model)
    return model


def _build_mobilenet_v2_hccr(
    num_classes: int, in_channels: int = 1, last_channel: int = 576
) -> nn.Module:
    """torchvision MobileNetV2 改造 for HCCR:
    - 第一层 conv 改单通道输入(3 → 1)
    - 最后 1×1 expand 从 320→1280 改为 320→last_channel(默认 576),
      避免分类头 Linear(1280, 3755) 单层就 4.8M params
    - 分类头 Linear(last_channel, num_classes)

    保留 ReLU6 + 标准 inverted residual(无 SE,无 HSwish,无 NAS 不规则宽度),
    量化最稳。skip connection(ResNet 思想)在 inverted residual 块内自动包含。

    输入: [B, 1, 64, 64]
    输出: [B, num_classes]
    """
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(num_classes=num_classes)
    # 1. 第一层 conv:3 → 1
    model.features[0][0] = nn.Conv2d(
        in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
    )
    # 2. 最后 1×1 expand:320 → last_channel
    model.features[18] = nn.Sequential(
        nn.Conv2d(320, last_channel, kernel_size=1, bias=False),
        nn.BatchNorm2d(last_channel),
        nn.ReLU6(inplace=True),
    )
    # 3. 分类头:Dropout + Linear
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(last_channel, num_classes),
    )
    # 4. 替换层后重新跑 torchvision 的 init 约定(同 V3 处理)
    for layer in (
        model.features[0][0],
        model.features[18][0],
        model.features[18][1],
        model.classifier[1],
    ):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)
    return model


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "plain_cnn":
        return PlainCNN(num_classes)
    if name == "mobilenet_v3_small":
        return _build_mobilenet_v3_small_hccr(num_classes)
    if name == "mobilenet_v2":
        return _build_mobilenet_v2_hccr(num_classes)
    if name == "mobilenet_v2_relu":
        return _build_mobilenet_v2_relu_hccr(num_classes)
    if name == "mobilenet_v2_gwap":
        return MobileNetV2WithGWAP(num_classes)
    if name == "melnyk_net":
        return MelnykNet(num_classes)
    raise ValueError(f"未知模型: {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # smoke:确认能 forward,参数量符合预期
    x = torch.randn(2, 1, 64, 64)
    for name in ("plain_cnn", "mobilenet_v3_small", "mobilenet_v2", "mobilenet_v2_relu", "mobilenet_v2_gwap", "melnyk_net"):
        model = build_model(name, num_classes=3755)
        n = count_params(model)
        bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
        y = model(x)
        print(f"{name}: {n:,} params ({n/1e6:.2f}M)  fp32={bytes_/1e6:.2f}MB  "
              f"int8≈{bytes_/4/1e6:.2f}MB  out={tuple(y.shape)}")
