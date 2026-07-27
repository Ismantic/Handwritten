# Handwritten

端到端手写汉字识别：**CASIA HWDB1.1 → PyTorch → NCNN INT8 → Android**。
MobileNetV2 在 3755 类 GB2312 一级字上的 top-1 为 95.47%，INT8 模型 4.1 MB，
移动端推理约 2–5 ms。

## 代码分层

```text
data/       原始数据状态、解压；不包含可再分发的数据
prepare/    GNT 解析、字符表、64×64 NumPy 数据集
src/        模型、Dataset、训练、评测、共享 C 预处理
save/       PNNX/NCNN 导出、FP16/INT8 量化、基准
demo/       Tkinter 与命令行推理
android/    Java + JNI + NCNN 独立应用
test/       单元测试、跨实现一致性、指标复现
docs/       训练、部署和设计理由
```

`src/` 不解析 GNT，也不负责模型发布；`prepare/` 只把有许可的本地原始数据转换
成数值输入；Python 和 Android 均编译 `src/cpp/preprocess.c`，避免预处理漂移。

## 从新克隆开始

环境要求为 Python ≥3.10、PyTorch、C 编译器；Android 另需 JDK 17、SDK 35、
NDK 和 CMake。建议使用独立虚拟环境：

```bash
uv pip install -r requirements.txt
cp local.mk.example local.mk     # 可选：覆盖 PYTHON
make -C data status
```

只使用 CPU 时可先从 PyTorch CPU 源安装，避免下载 CUDA 运行库：

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy Pillow tqdm matplotlib ncnn pnnx
```

CASIA HWDB1.1 不能随仓库再分发。请从官方渠道申请并把
`HWDB1.1trn_gnt.zip`、`HWDB1.1tst_gnt.zip` 放入 `data/`，随后运行：

```bash
make -C data extract
make -C prepare all
make -C src smoke
```

完整训练：

```bash
make -C src train RUN_DIR=runs/mbv2
```

## 导出与运行

```bash
make -C save all RUN_NAME=mbv2 NAME=mbv2
make -C demo cli IMG=path/to/image.png
make -C demo run
```

Android 依赖和模型复制步骤见 [`docs/DEPLOY.md`](docs/DEPLOY.md)。

## 验证

```bash
python -m unittest discover -s test -v
HCCR_REPRODUCE=1 python -m unittest test.test_reproduce -v
cd android && ./gradlew testDebugUnitTest assembleDebug
```

指标复现测试默认关闭，因为它需要完整测试集和 checkpoint。修改预处理、模型或
量化时必须同时报告 checkpoint、命令、top-1/top-10、延迟和模型大小。

## 文档

- [`docs/TRAIN.md`](docs/TRAIN.md)：数据准备与训练。
- [`docs/DEPLOY.md`](docs/DEPLOY.md)：NCNN 导出和 Android 构建。
- [`docs/WHY.md`](docs/WHY.md)：关键口径、量化结论与容易静默出错的地方。

Apache-2.0，见 [`LICENSE`](LICENSE)。CASIA 数据遵循其自身许可。
