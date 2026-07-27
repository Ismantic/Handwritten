-include ../local.mk
PYTHON ?= python
ROOT   := ..

MODEL_DIR    ?= models/mbv2_aug_int8
NCNN_PARAM   ?= $(MODEL_DIR)/model.ncnn.param
NCNN_BIN     ?= $(MODEL_DIR)/model.ncnn.bin
CHARSET      ?= $(MODEL_DIR)/charset.json

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
