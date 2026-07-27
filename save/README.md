# save/

把训练 checkpoint 转换成 NCNN，并打包成可直接上传 Hugging Face 的自包含目录。

```bash
make -C save help
make -C save status
```

完整发布流程：

```bash
make -C save export
make -C save verify
make -C save dry-run
make -C save upload
```

只修改推理代码或模型卡、不重传权重：

```bash
make -C save code
make -C save verify
make -C save dry-run CODE_ONLY=1
make -C save upload-code
```

## 三类目录

| 目录 | 内容 | 是否进 Git |
|---|---|---|
| `runs/` | 训练 checkpoint | 否 |
| `save/output/` | PNNX、FP32/FP16/INT8 转换中间产物 | 否 |
| `save/releases/` | HF 发布目录，可从源文件重建 | 否 |
| `models/` | 从 HF 下载、供 demo 使用的本地缓存 | 否 |

## 文件

```text
releases.py    发布清单：checkpoint、模型包、指标
cards.py       生成 HF 模型卡
convert.py     PyTorch checkpoint → NCNN
export.py      发布源 → save/releases/Handwritten
upload.py      发布目录 → Hugging Face
assets/        随模型发布的真实推理代码与示例
benchmark.py   PyTorch/NCNN 准确率和延迟对比
```

`test/test_save.py` 检查源文件与发布文件逐字节一致，并在发布目录内独立加载
PyTorch checkpoint、运行 NCNN 推理。上传需要先用 `hf auth login` 登录；
token 不写入仓库、配置或命令行。
