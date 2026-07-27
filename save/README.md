# Save

本层消费 `runs/<name>/best.pt`，生成 NCNN FP32、FP16 和 INT8 模型，并比较
PyTorch/NCNN 的准确率与延迟。

```bash
make -C save status
make -C save all RUN_NAME=mbv2 NAME=mbv2
```

产物写入 `save/output/` 且不进 Git。INT8 默认跳过 MobileNetV2 的首层敏感卷积；
切换模型时必须重新检查层名、准确率损失和模型大小。
