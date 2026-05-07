# HCCR Android App 构建

参考 plan 阶段四路径 A:独立 App,**非** IME。先验证完整识别 pipeline,IME 集成后续。

## 环境要求

- **Android SDK**:compileSdk 35
- **NDK**:30.0.14904198
- **CMake**:3.22.1+
- **Java**:JDK 17(用 Android Studio 自带的 JBR 也行)
- **Gradle**:8.11.1(wrapper 自动下载)

## 准备

### 1. 配置 SDK 路径

`android/local.properties`:
```
sdk.dir=/home/tfbao/Android/Sdk
```

### 2. ncnn-android 预编译

需要解压到 `android/third_party/ncnn-android/`:
```
android/third_party/ncnn-android/
├── arm64-v8a/{lib,include}/
├── armeabi-v7a/...
├── x86_64/...
└── x86/...
```

下载:
```bash
wget https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-android-vulkan.zip
unzip ncnn-20250503-android-vulkan.zip -d android/third_party/
mv android/third_party/ncnn-20250503-android-vulkan android/third_party/ncnn-android
```

### 3. 模型文件

从 export 阶段产出复制到 assets:
```bash
cp export/output/mbv2_aug.int8.ncnn.{param,bin} android/app/src/main/assets/
cp data/processed/charset.json android/app/src/main/assets/
```

## 构建

在 `android/` 目录:

```bash
export JAVA_HOME=/opt/android-studio/jbr
./gradlew assembleDebug                 # APK → app/build/outputs/apk/debug/app-debug.apk
./gradlew installDebug                  # 安装到已连接的设备
```

或用 Android Studio:File → Open → 选 `android/` 目录。

## 运行

应用启动后:
- 顶部状态栏 + 候选栏(top-10)
- 中间画板(白底,手指/触屏笔写字)
- 底部:撤销 / 清空

抬笔 200ms 后自动识别,top-1 显示在状态栏(含推理延迟 ms)。

## 模型 / 输入对齐

- Java 端预处理(MainActivity.preprocessToFloat64x64):
  - bbox 裁切 → 长边 56 等比缩 → 居中放 64×64 白底
  - 翻转 + 归一化:`(255 - gray) / 255` → float [1, 64, 64]
- 跟训练 pipeline(`common/normalize.py` + `common/dataset.py`)的语义一致

## TODO(后续)

- [ ] IME 集成(plan 路径 B):InputMethodService + Sime 风格的候选栏
- [ ] 真机数据收集 + fine-tune(plan 阶段五)
- [ ] 笔画粗细自适应屏幕密度
- [ ] 候选栏抖动控制(top-1 连续 N 次相同才上屏)
