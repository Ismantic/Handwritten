# Handwritten — 手机输入法手写汉字识别

端到端实现:**HWDB1.1 训练 → MobileNetV2 → NCNN INT8 → Android 独立 App**。
3755 类 GB2312 一级字,真机推理 < 5 ms。

## 结果

| 指标 | 数值 | plan 目标 | 状态 |
|------|------|-----------|------|
| top-1 准确率 | **95.47%**(V2 + 增强 30 epoch)| ≥ 95% | ✓ |
| top-10 准确率 | **99.58%** | ≥ 99% | ✓ |
| INT8 模型大小 | **4.1 MB** | ≤ 5 MB | ✓ |
| 推理延迟(预估手机)| **2-5 ms** | ≤ 30 ms | ✓ |

`top-1 96.47%` 也达到过(Melnyk-Net 30 epoch),但模型 6.5 MB 超 plan 限,作 reference。

## 架构

```
Handwritten/
├── data/        阶段一  原始 zip → GNT 解析 → 64×64 npy + charset.json
├── common/      共享代码:charset / dataset / models / 预处理(C 单一源)
│   └── cpp/     ★preprocess.c:Python (ctypes) 与 Android (JNI) 字节级共用★
├── fit/         阶段二  训练入口(plain CNN / mbv2 / mbv3 / Melnyk-Net)
├── export/      阶段三  PyTorch ckpt → PNNX → NCNN(FP32 / FP16 / INT8)
├── demo/        Tkinter 桌面 demo,实时手写 → top-10 候选
├── android/     阶段四(plan 路径 A)独立 App,Java + JNI 调 NCNN
├── private/     设计/实验文档,不入库
└── pyproject.toml  把 common/ 装成可编辑包(`pip install -e .`)
```

每个目录有自己的 `Makefile`,所有阶段通过 `make` 串联,产出文件依赖追踪。

## 阶段一:数据 pipeline (`data/`)

```bash
cd data
make extract     # 解 HWDB1.1trn_gnt.zip + .alz → raw/{train,test}_gnt/
make charset     # 扫训练集生成 processed/charset.json (3755 一级字)
make visualize   # 抽样可视化检查 GNT 解析正确
make npy         # 归一化 + 打包成 npy/{train,test}/{images,labels}.npy
                 #   training 897,758 / test 223,991 / 64×64 uint8
make verify      # 抽样反查 npy 人眼复核
```

要求先把 `data/HWDB1.1trn_gnt.zip` 和 `data/HWDB1.1tst_gnt.zip` 准备好(从 CASIA 下载)。

## 阶段二:训练 (`fit/`)

```bash
make -C fit smoke      # 5% 数据 / 1 epoch / 验证 pipeline 通
make -C fit baseline   # 完整训练(plain CNN, 10 epoch ≈ 12 min)

# 切换模型 / 调参
python fit/train.py --model mobilenet_v2 \
    --epochs 30 --batch-size 256 --warmup-epochs 3 \
    --min-lr 1e-5 --weight-decay 1e-4 --augment true \
    --num-workers 0 --out-dir runs/mbv2_phase1_aug
```

支持的模型(在 `common/models.py`):

| name | params | 说明 |
|------|--------|------|
| `plain_cnn` | 1.13 M | 5 stage VGG 风格 baseline |
| `mobilenet_v2` | 4.16 M | 改造单通道 + 简化分类头 |
| `mobilenet_v2_relu` | 4.16 M | V2 替换 ReLU6 → ReLU(量化更稳) |
| `mobilenet_v2_gwap` | 4.17 M | V2 末端用 GWAP 替换 GAP |
| `mobilenet_v3_small` | 3.09 M | torchvision V3-Small 改造 |
| `melnyk_net` | 6.51 M | Melnyk et al. 2020 论文复现 |

## 阶段三:导出 (`export/`)

```bash
cd export
make trace        # ckpt → pnnx → .ncnn.param/.bin (FP32)
make int8         # 校准数据 → ncnn2table → ncnn2int8(默认跳首层 conv)
make benchmark    # PyTorch / NCNN FP32 / FP16 / INT8 精度+延迟对比

# 切换模型
make trace RUN_NAME=plain_30ep_aug NAME=plain_aug
```

INT8 量化关键:**默认 skip 首层 conv**(`SKIP_LAYERS=convclip_0|convdwclip_0`),
量化损失从 -2.84 → -0.08 点(MobileNetV2 首层激活动态范围特别大,per-tensor INT8 表达不下)。

## 阶段四:Android App (`android/`)

```bash
cd android
# 第一次需要:
echo 'sdk.dir=/path/to/Android/Sdk' > local.properties
# 下 ncnn-android 预编译解压到 third_party/ncnn-android/(BUILD.md 里有命令)
# cp 模型文件到 app/src/main/assets/

export JAVA_HOME=/opt/android-studio/jbr
./gradlew installDebug    # 装 Debug APK 到连接的设备
```

布局:半屏 IME 风格,候选栏在面板顶端,画板占主要区域,撤销 / 清空在底部。
笔迹采集 → JNI 调 ncnn → top-10 实时刷新,抬笔后 150 ms 自动识别。

## Python 桌面 demo (`demo/`)

```bash
make -C demo run            # Tkinter 画板 + NCNN INT8 实时识别
make -C demo cli IMG=path/to/image.png  # 不开 GUI,跑一张图
```

## 关键设计:统一的 C 预处理 (`common/cpp/`)

训练 / Python demo / Android 三端**共享同一份 C 实现**。

```c
// common/cpp/preprocess.h
int hccr_preprocess(const uint8_t* gray, int w, int h, float* out_64x64);
//   1. bbox(像素 < 220 算前景)
//   2. 长边等比缩到 56,separable 三角核 bilinear(同 PIL.Image.BILINEAR)
//   3. 居中放到 64×64 白底
//   4. 翻转 + 归一化 → float[1, 64, 64]
```

- **Python**:`common/preprocess.py` 用 `ctypes` 加载 `libhccr_preprocess.so`(`make -C common/cpp`)
- **Android**:JNI 的 `CMakeLists.txt` 直接把 `preprocess.c` 编进 `libhccr_jni.so`

为什么这么搞:Java 端用 Android 自带的 `Bitmap.createScaledBitmap` 是简单 2x2 bilinear,
downscale ≥ 2x 时不做 box-prefilter,跟 PIL 语义有本质差异 → 模型识别率从 80%+ 掉到 12%。
追到根因后写 PIL bilinear 等价 Java 实现修好,但长期会跟 Python 端 drift,
所以最后下沉到 C,字节级对齐。

## 模型导出 / 部署路径

```
PyTorch ckpt → torch.jit.trace → PNNX → NCNN(.ncnn.param/.bin)
                                      ↓
                          ncnn2table 校准 (200-2000 张训练图)
                                      ↓
                          ncnn2int8 量化(skip 首层 conv)→ INT8 模型
                                      ↓
                              Android JNI ncnn.Net.load_param/load_model
                                      ↓
                              真机 ARM CPU + NEON / SDOT
```

- 端上推理走 **NCNN**(腾讯,移动端,Vulkan/OpenCL backend,库 ~1 MB)
- 转换走 **PNNX 直转**(ncnn 作者自家工具,跟 ncnn op 一对一映射,不用 ONNX 中间)

## 环境

- Python 3.13(`/home/tfbao/.venv`,uv 创建)
- PyTorch 2.11 + CUDA 13(GPU:RTX 2070 8 GB)
- NCNN runtime + pnnx 工具(`pip install ncnn pnnx`)
- NCNN 命令行工具(`pacman -S ncnn`,`ncnn2table` / `ncnn2int8`)
- Android SDK 35 + NDK 30.0.14904198 + CMake 3.22+

## License

[Apache 2.0](LICENSE)
