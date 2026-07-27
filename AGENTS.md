# 仓库约定

手写汉字识别完整链路：CASIA HWDB1.1 → PyTorch → NCNN INT8 → Android。
架构与指标见 [`README.md`](README.md)，这里只列日常开发约束。

## 分层

```text
data/       授权数据状态与解压
prepare/    GNT、字符、图像 → 数值数据
src/        模型、Dataset、训练、评测、共享 C 预处理
save/       NCNN 导出、量化、基准
demo/ android/ test/ docs/
```

`src/` 不解析原始 GNT；`prepare/` 不定义模型；`save/` 不改变训练口径。Python
与 Android 共用 `src/cpp/preprocess.c`，任何预处理变化都必须做跨端一致性验证。
优先扩展现有入口，不要创建功能重复的脚本。

## 常用命令

```bash
uv pip install -r requirements.txt
make -C data status
make -C prepare status
make -C src preprocess smoke
python -m unittest discover -s test -v
make -C save benchmark BENCH_N=5000
cd android && ./gradlew testDebugUnitTest assembleDebug
```

Makefile 默认使用 `python`。本机覆盖写进 gitignored 的 `local.mk`，模板见
`local.mk.example`；禁止把 `/home/...`、SDK 或虚拟环境绝对路径写进受控文件。

## 风格与测试

Python/Java 四空格，Python 使用 `snake_case`，Java 类使用 `PascalCase`，常量使用
`UPPER_SNAKE_CASE`。注释解释为什么和静默失败风险，不复述控制流。改 `src/` 或
`prepare/` 至少跑单元测试；改预处理还要跑 C parity 和 Android 构建；改模型或
量化要报告 checkpoint、命令、top-1/top-10、延迟、模型大小。

指标硬基准通过
`HCCR_REPRODUCE=1 python -m unittest test.test_reproduce -v` 运行，不能为了通过
而放宽评测代码或改变测试集。

## 数据与提交

CASIA 数据、`data/npy/`、`runs/`、`save/output/`、`save/releases/`、
`models/`、Android assets 和构建产物都
不进 Git。删除 gitignored 目录前先 `du -sh`；它们可能无法从仓库恢复。

提交使用简短、结果导向的中文标题，一次只做一件事，不添加 AI 署名。PR 说明
动机和验证结果；UI 变化附截图，模型变化附完整指标。
