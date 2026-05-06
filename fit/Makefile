# 训练入口
#
# 必须从仓库根目录跑(common 包从根 import,npy/charset 路径相对根):
#   make -C fit smoke      # 5% 数据,1 epoch,验证 pipeline
#   make -C fit baseline   # 全量训练(参数等讨论后再调)
#   make -C fit clean      # 删掉 runs/

PYTHON := /home/tfbao/.venv/bin/python
ROOT   := ..

# smoke 默认输出
SMOKE_DIR    ?= runs/smoke
BASELINE_DIR ?= runs/baseline

.PHONY: smoke baseline clean help

smoke:
	cd $(ROOT) && $(PYTHON) fit/train.py \
	    --epochs 1 \
	    --subset-frac 0.05 \
	    --batch-size 128 \
	    --num-workers 4 \
	    --log-every 20 \
	    --out-dir $(SMOKE_DIR)

baseline:
	cd $(ROOT) && $(PYTHON) fit/train.py \
	    --epochs 30 \
	    --batch-size 128 \
	    --num-workers 4 \
	    --out-dir $(BASELINE_DIR)

clean:
	cd $(ROOT) && rm -rf $(SMOKE_DIR) $(BASELINE_DIR)

help:
	@echo "fit targets (must be invoked from repo root or via make -C fit):"
	@echo "  make -C fit smoke      5%数据 1 epoch,验证 pipeline"
	@echo "  make -C fit baseline   全量训练(参数待确认)"
	@echo "  make -C fit clean      删 runs/"
