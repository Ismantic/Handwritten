# Android 构建

完整的依赖、模型复制和构建步骤统一维护在
[`docs/DEPLOY.md`](../docs/DEPLOY.md)。本目录只保留 Android 工程本身，避免两份
部署说明发生漂移。

快速构建：

```bash
cd android
./gradlew testDebugUnitTest assembleDebug
```

本机 SDK 路径写入 gitignored 的 `local.properties`，不要提交绝对路径。
