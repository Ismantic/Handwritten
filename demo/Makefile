PYTHON := /home/tfbao/.venv/bin/python
ROOT   := ..

NCNN_PARAM   ?= export/output/mbv2_aug.int8.ncnn.param
NCNN_BIN     ?= export/output/mbv2_aug.int8.ncnn.bin
CHARSET      ?= data/processed/charset.json

.PHONY: run cli help

run:
	cd $(ROOT) && $(PYTHON) -m demo.app \
	    --ncnn-param $(NCNN_PARAM) \
	    --ncnn-bin $(NCNN_BIN) \
	    --charset $(CHARSET)

# CLI 推理(传图片路径,无 GUI),用于无显示环境测试
cli:
	@if [ -z "$(IMG)" ]; then echo "用法: make cli IMG=path/to/image.png"; exit 1; fi
	cd $(ROOT) && $(PYTHON) -m demo.inference \
	    --image $(IMG) \
	    --ncnn-param $(NCNN_PARAM) \
	    --ncnn-bin $(NCNN_BIN) \
	    --charset $(CHARSET)

help:
	@echo "demo targets:"
	@echo "  make run          启动 Tkinter GUI(需要图形界面)"
	@echo "  make cli IMG=...  CLI 模式,跑一张图打印 top-10"
