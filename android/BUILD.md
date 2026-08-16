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

## 正式 Release

正式 APK 使用独立的签名文件；签名信息不进仓库。默认从
`android/keystore.properties` 读取，也可通过 Gradle 属性指定其位置：

```properties
storeFile=handwritten-release.jks
storePassword=...
keyAlias=...
keyPassword=...
```

```bash
cd android
./gradlew -PreleaseKeystoreProperties=/path/to/keystore.properties assembleRelease
```

产物为 `app/build/outputs/apk/release/app-release.apk`。发布前至少完成 Android
构建，并在真机上验证书写、候选和上屏。
