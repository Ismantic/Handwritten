-include ../local.mk
PYTHON ?= python
ROOT   := ..

MODEL_DIR    ?= models/Handwritten
NCNN_PARAM   ?= $(MODEL_DIR)/ncnn/model.ncnn.param
NCNN_BIN     ?= $(MODEL_DIR)/ncnn/model.ncnn.bin
CHARSET      ?= $(MODEL_DIR)/ncnn/charset.json

.PHONY: model run cli help

model:
	@if [ ! -f "../$(NCNN_BIN)" ]; then \
	    $(MAKE) -C ../save download; \
	fi

run: model
	cd $(ROOT) && $(PYTHON) -m demo.app \
	    --ncnn-param $(NCNN_PARAM) \
	    --ncnn-bin $(NCNN_BIN) \
	    --charset $(CHARSET)

# CLI 推理(传图片路径,无 GUI),用于无显示环境测试
cli: model
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
