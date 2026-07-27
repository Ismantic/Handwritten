"""生成 Hugging Face 模型卡。"""


def model_card(name: str, spec: dict) -> str:
    metrics = spec["metrics"]
    return f"""---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- handwritten-chinese-character-recognition
- casia-hwdb
- mobilenet-v2
- ncnn
- int8
---

# {name}

MobileNetV2 手写汉字识别模型,覆盖 3755 个 GB2312 一级汉字。输入为 64×64
灰度图,训练数据来自 CASIA HWDB1.1。

## 指标

| 指标 | 数值 |
|---|---:|
| PyTorch top-1 | {metrics['top1']:.2%} |
| PyTorch top-10 | {metrics['top10']:.2%} |
| NCNN INT8 权重 | 4.1 MB |
| 预估移动端延迟 | 2–5 ms |

## 直接使用 NCNN

```bash
pip install numpy Pillow ncnn
python example.py path/to/image.png
```

```python
from inference import HCCRRecognizer

model = HCCRRecognizer.from_pretrained(".")
print(model.predict_file("path/to/image.png", k=10))
```

## 加载 PyTorch checkpoint

```python
import torch
from model import build_model

checkpoint = torch.load("best.pt", map_location="cpu", weights_only=False)
model = build_model(checkpoint["cfg"]["model"], num_classes=3755)
model.load_state_dict(checkpoint["model"])
model.eval()
```

## 文件

| 文件 | 内容 |
|---|---|
| `best.pt` | PyTorch checkpoint、训练配置、epoch 和 top-1 |
| `model.py` | PyTorch 模型定义 |
| `inference.py` | 自包含 NCNN 推理入口 |
| `normalize.py` | 与训练一致的图像归一化 |
| `example.py` | 单图命令行示例 |
| `ncnn/model.ncnn.*` | NCNN INT8 网络和权重 |
| `ncnn/charset.json` | 输出索引到汉字的映射 |
| `release_metadata.json` | 发布来源、指标和文件校验值 |

完整训练、导出与 Android 部署见 [{spec['source']}]({spec['source']})。

## 数据许可

本仓库不再分发 CASIA HWDB1.1。请从官方渠道申请数据并遵守其许可。
"""
