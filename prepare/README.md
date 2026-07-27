# Prepare

本目录把授权获得的 GNT 文件转换成训练代码可直接读取的数值数据：

```text
data/raw/*.gnt → charset.json → images.npy + labels.npy
```

运行 `make -C prepare status` 检查产物，`make -C prepare all` 构建并抽样验证。
输出仍放在 `data/processed/` 和 `data/npy/`，以便保留已有大文件；代码与数据的
职责通过目录分开。这里允许解析文本/字符和图片格式，`src/` 不允许。
