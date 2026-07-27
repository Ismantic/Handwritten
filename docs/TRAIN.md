# 训练与数据准备

## 数据

从 CASIA 官方渠道申请 HWDB1.1，把训练和测试归档放入 `data/`。运行
`make -C data extract` 后，`make -C prepare all` 会生成 3755 字字符表以及
64×64、白底黑字的 NumPy 数组。`make -C prepare verify` 的图片检查不能省略：
编码错位通常不会让训练程序报错。

## 训练

先运行 `make -C src smoke` 验证数据加载、前向、评测和 checkpoint 写入；该命令
只跑 20 个训练和评测 batch，不代表准确率。确认流程后再运行
`make -C src train RUN_DIR=runs/mbv2`。所有随机种子和超参数写入 checkpoint 与
`config.json`。长实验应保留命令、最佳 epoch、top-1/top-5/top-10 和 GPU 型号。

已有基准为 MobileNetV2、30 epoch、数据增强：top-1 95.47%，top-10 99.58%。
改变数据划分、字符表顺序或预处理后，这些数字不再具有直接可比性。
