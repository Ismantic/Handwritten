# Data

本目录只管理 CASIA HWDB1.1 原始归档和解压状态，不包含下载绕过或数据副本。
数据需从 CASIA 官方渠道申请；许可不允许仓库替用户重新分发。

```bash
make -C data status
make -C data extract
```

把两个官方文件放在 `data/`：`HWDB1.1trn_gnt.zip` 和
`HWDB1.1tst_gnt.zip`。解压结果写入 `data/raw/`，该目录不进 Git。
