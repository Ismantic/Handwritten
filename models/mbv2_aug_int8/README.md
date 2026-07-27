# MobileNetV2 INT8

仓库默认运行时模型，由 `runs/mbv2_phase1_aug/best.pt` 导出并用 NCNN 选择性
INT8 量化。识别范围为 3755 个 GB2312 一级汉字。

| 指标 | 数值 |
|---|---:|
| PyTorch top-1 | 95.47% |
| PyTorch top-10 | 99.58% |
| NCNN INT8 权重 | 4.1 MB |
| 预估移动端延迟 | 2–5 ms |

文件：

- `model.ncnn.param`：NCNN 网络结构及量化参数。
- `model.ncnn.bin`：模型权重。
- `charset.json`：输出索引到汉字的映射。

重新训练或量化后，运行
`make -C save package RUN_NAME=<run> NAME=<name>` 生成同样的运行时目录。
