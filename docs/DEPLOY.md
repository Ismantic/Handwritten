# NCNN 与 Android 部署

## 导出

`make -C save all RUN_NAME=<run> NAME=<model>` 依次生成 FP32、FP16、INT8 并运行
基准。量化需要系统中的 `ncnn2table` 和 `ncnn2int8`。不要只看模型能否加载；
必须比较 PyTorch 与各 NCNN 版本的准确率和预测一致率。

## Android

准备 SDK 35、JDK 17、NDK 30 和 CMake 3.22+，在 `android/local.properties`
填写本机 `sdk.dir`。把 ncnn Android 预编译包解压到
`android/third_party/ncnn-android/`，再复制模型和字符表：

```bash
make -C save download
cp models/Handwritten/ncnn/model.ncnn.param android/app/src/main/assets/mbv2_aug.int8.ncnn.param
cp models/Handwritten/ncnn/model.ncnn.bin android/app/src/main/assets/mbv2_aug.int8.ncnn.bin
cp models/Handwritten/ncnn/charset.json android/app/src/main/assets/charset.json
cd android && ./gradlew assembleDebug
```

JNI 直接编译仓库中的 `src/cpp/preprocess.c`。修改该文件后同时运行 Python
一致性测试和 Android 构建。
