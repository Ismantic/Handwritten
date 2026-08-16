# HCCR Android Runtime

这里是供 Android 应用复用的离线手写识别 runtime：

- `HCCRRecognizer.java`：模型、字表和 JNI 的 Java 接口；
- `hccr_jni.cc`：NCNN 推理、top-k 与 native 生命周期；
- `src/cpp/preprocess.c`（仓库根目录）：与 Python 共用的 64×64 预处理。

`android/app` 是独立 Demo，也直接编译这份 runtime。SimeApp 通过
`HANDWRITTEN_ROOT` 指向本仓库并使用相同源码和 Android assets；它只保留输入法的
书写画布、候选栏和键盘 UI。模型更新必须连同 `param`、`bin`、`charset.json` 一起
验证，再由下游应用重新打包。
