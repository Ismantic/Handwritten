# HWDB1.1 数据处理 pipeline
#
# 入口 target:
#   make extract     解压原始 zip(测试集)和 zip→alz→gnt(训练集)
#   make charset     扫描训练集,统计字符表,过滤 GB2312 一级字 → processed/charset.json
#   make visualize   随机抽样,渲染汉字图 + 标签,人眼检查解析正确性
#   make npy         归一化到 64×64 + 写成 images.npy / labels.npy
#   make verify      抽样反查 npy,人眼复核
#   make all         extract + charset + npy
#   make clean       删 processed/ 和 npy/(保留 raw/)
#   make distclean   额外删 raw/
#   make help        列出 target

PYTHON      := /home/tfbao/.venv/bin/python

RAW_DIR     := raw
PROC_DIR    := processed
NPY_DIR     := npy
SCRIPTS     := scripts

TRAIN_ZIP   := HWDB1.1trn_gnt.zip
TEST_ZIP    := HWDB1.1tst_gnt.zip

# 解压完成的 sentinel(touch 出来做依赖追踪用,避免每次都重跑解压)
TRAIN_TAG   := $(RAW_DIR)/.train_extracted
TEST_TAG    := $(RAW_DIR)/.test_extracted

CHARSET     := $(PROC_DIR)/charset.json
VIZ_TAG     := $(PROC_DIR)/visualize/.done

TRAIN_NPY   := $(NPY_DIR)/train/images.npy
TEST_NPY    := $(NPY_DIR)/test/images.npy

# 可视化抽样数量,可在命令行覆盖:make visualize VIZ_N=128
VIZ_N       ?= 64

.PHONY: all extract charset visualize npy verify clean distclean help

all: extract charset npy

# ---------- extract ----------

extract: $(TRAIN_TAG) $(TEST_TAG)

$(TEST_TAG): $(TEST_ZIP)
	@echo "[extract] 解压测试集 zip..."
	mkdir -p $(RAW_DIR)/test_gnt
	unzip -q -o $< -d $(RAW_DIR)/test_gnt
	@touch $@
	@echo "[extract] 测试集 OK ($$(ls $(RAW_DIR)/test_gnt/*.gnt | wc -l) 个 .gnt)"

$(TRAIN_TAG): $(TRAIN_ZIP)
	@echo "[extract] 解压训练集外层 zip..."
	mkdir -p $(RAW_DIR)/.train_alz_tmp
	unzip -q -o $< -d $(RAW_DIR)/.train_alz_tmp
	@echo "[extract] 解压训练集内层 alz(unar,可能要几分钟)..."
	mkdir -p $(RAW_DIR)/train_gnt
	unar -q -f -o $(RAW_DIR)/train_gnt -D $(RAW_DIR)/.train_alz_tmp/HWDB1.1trn_gnt.alz
	rm -rf $(RAW_DIR)/.train_alz_tmp
	@touch $@
	@echo "[extract] 训练集 OK ($$(ls $(RAW_DIR)/train_gnt/*.gnt | wc -l) 个 .gnt)"

# ---------- charset ----------

charset: $(CHARSET)

$(CHARSET): $(TRAIN_TAG) $(SCRIPTS)/build_charset.py $(SCRIPTS)/gnt_parser.py
	@echo "[charset] 扫描训练集,生成字符表..."
	mkdir -p $(PROC_DIR)
	$(PYTHON) $(SCRIPTS)/build_charset.py \
	    --gnt-dir $(RAW_DIR)/train_gnt \
	    --out $@

# ---------- visualize ----------

visualize: $(VIZ_TAG)

$(VIZ_TAG): $(TRAIN_TAG) $(CHARSET) $(SCRIPTS)/visualize.py $(SCRIPTS)/gnt_parser.py
	@echo "[visualize] 随机采 $(VIZ_N) 个样本渲染到 $(PROC_DIR)/visualize/..."
	mkdir -p $(PROC_DIR)/visualize
	$(PYTHON) $(SCRIPTS)/visualize.py \
	    --gnt-dir $(RAW_DIR)/train_gnt \
	    --charset $(CHARSET) \
	    --out-dir $(PROC_DIR)/visualize \
	    --n $(VIZ_N)
	@touch $@

# ---------- npy ----------

npy: $(TRAIN_NPY) $(TEST_NPY)

$(TRAIN_NPY): $(TRAIN_TAG) $(CHARSET) $(SCRIPTS)/build_npy.py $(SCRIPTS)/gnt_parser.py $(SCRIPTS)/normalize.py
	@echo "[npy] 训练集归一化 + 写 npy..."
	$(PYTHON) $(SCRIPTS)/build_npy.py \
	    --gnt-dir $(RAW_DIR)/train_gnt \
	    --charset $(CHARSET) \
	    --out-dir $(NPY_DIR)/train

$(TEST_NPY): $(TEST_TAG) $(CHARSET) $(SCRIPTS)/build_npy.py $(SCRIPTS)/gnt_parser.py $(SCRIPTS)/normalize.py
	@echo "[npy] 测试集归一化 + 写 npy..."
	$(PYTHON) $(SCRIPTS)/build_npy.py \
	    --gnt-dir $(RAW_DIR)/test_gnt \
	    --charset $(CHARSET) \
	    --out-dir $(NPY_DIR)/test

# ---------- verify ----------

verify: $(TRAIN_NPY) $(TEST_NPY) $(SCRIPTS)/verify_npy.py
	@echo "[verify] 抽样训练 npy..."
	$(PYTHON) $(SCRIPTS)/verify_npy.py \
	    --npy-dir $(NPY_DIR)/train \
	    --charset $(CHARSET) \
	    --out $(PROC_DIR)/verify/train_grid.png \
	    --n $(VIZ_N)
	@echo "[verify] 抽样测试 npy..."
	$(PYTHON) $(SCRIPTS)/verify_npy.py \
	    --npy-dir $(NPY_DIR)/test \
	    --charset $(CHARSET) \
	    --out $(PROC_DIR)/verify/test_grid.png \
	    --n $(VIZ_N)

# ---------- clean ----------

clean:
	rm -rf $(PROC_DIR) $(NPY_DIR)

distclean: clean
	rm -rf $(RAW_DIR)

help:
	@echo "HWDB1.1 数据 pipeline targets:"
	@echo "  make extract     解压 zip / alz → raw/{train,test}_gnt/"
	@echo "  make charset     扫描训练集,生成 processed/charset.json"
	@echo "  make visualize   随机抽样可视化(VIZ_N=64)"
	@echo "  make npy         归一化 + 写成 images.npy / labels.npy"
	@echo "  make verify      抽样反查 npy,人眼复核(processed/verify/*.png)"
	@echo "  make all         extract + charset + npy"
	@echo "  make clean       删 processed/ 和 npy/"
	@echo "  make distclean   额外删 raw/"
