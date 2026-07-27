# 关键设计理由

## 单一预处理实现

Android 的 `Bitmap.createScaledBitmap` 在大比例缩小时与 PIL 的 bilinear 语义
不同，曾使准确率从 80% 以上降到约 12%。部署路径因此统一使用
`src/cpp/preprocess.c`：前景阈值 220、长边缩到 56、居中到 64×64，最后翻转到
笔画为高值。`src/normalize.py` 保留为可读参考和数据构建实现。

## 数值数据边界

GNT 解码和字符过滤只在 `prepare/`。`src/` 接收固定字符索引和 NumPy 数组，
避免训练阶段悄悄改变编码、图像方向或类别顺序。

## 选择性 INT8

MobileNetV2 首层激活动态范围很大，per-tensor INT8 全量量化曾损失 2.84 个
百分点。跳过首层卷积后损失约 0.08 点，因此 `save/Makefile` 默认设置敏感层
pattern。模型结构变化会改变 PNNX 层名，不能盲用旧 pattern。

## 数据获取

CASIA HWDB 的授权条件不同于可公开镜像的数据集。本仓库提供状态、解压和加工
命令，但不伪造自动下载地址；可复现性必须建立在合法获取步骤写清楚的基础上。
