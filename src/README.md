# Core

`src/` 是训练和推理共享的核心层：

- `models.py`：可导出到 NCNN 的模型定义。
- `dataset.py`：只读取预编码的 NumPy 数组。
- `train.py`：训练、评测和 checkpoint。
- `normalize.py`：便于检查的 Python 参考实现。
- `cpp/preprocess.c`：桌面端与 Android 使用的部署实现。

训练入口必须从仓库根目录按模块运行：`python -m src.train ...`。不要在这里解析
GNT、下载数据或处理发布包。添加算子前先确认 PNNX/NCNN 可导出。
