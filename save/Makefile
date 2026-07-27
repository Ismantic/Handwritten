HERE := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT := $(patsubst %/,%,$(dir $(HERE)))

-include $(ROOT)/local.mk
PYTHON ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)
RUN_NAME ?= mbv2_phase1_aug
CKPT ?= ../runs/$(RUN_NAME)/best.pt
NAME ?= mbv2_aug
OUT_DIR ?= output
BENCH_N ?= 5000
CALIB_N ?= 2000
INT8_SKIP_LAYERS ?= convclip_0|convdwclip_0

BASE := $(OUT_DIR)/$(NAME)

.PHONY: status download trace fp16 int8 benchmark all export code verify dry-run upload upload-code help

status:
	@for f in $(CKPT) $(BASE).ncnn.param $(BASE).int8.ncnn.param; do \
	    if test -f "$$f"; then echo "[ok]      $$f"; else echo "[missing] $$f"; fi; \
	done
	@cd $(ROOT) && $(PYTHON) -m save.export --list

download:
	hf download Ismantic/Handwritten --local-dir $(ROOT)/models/Handwritten

trace: $(BASE).ncnn.param

$(BASE).ncnn.param: convert.py $(CKPT)
	cd $(ROOT) && $(PYTHON) -m save.convert --ckpt save/$(CKPT) \
	    --out-dir save/$(OUT_DIR) --name $(NAME)

fp16: $(BASE).fp16.ncnn.param

$(BASE).fp16.ncnn.param: convert.py $(CKPT)
	cd $(ROOT) && $(PYTHON) -m save.convert --ckpt save/$(CKPT) \
	    --out-dir save/$(OUT_DIR) --name $(NAME).fp16 --fp16

$(BASE).calib_list.txt:
	cd $(ROOT) && $(PYTHON) -m save.make_calib \
	    --out-dir save/$(OUT_DIR)/calib_$(NAME) --list save/$@ --n $(CALIB_N)

$(BASE).int8.ncnn.param: $(BASE).ncnn.param $(BASE).calib_list.txt quantize_int8.sh
	SKIP_LAYERS='$(INT8_SKIP_LAYERS)' ./quantize_int8.sh \
	    $(BASE).ncnn.param $(BASE).ncnn.bin $(BASE).calib_list.txt \
	    $(BASE).int8.ncnn.param $(BASE).int8.ncnn.bin

int8: $(BASE).int8.ncnn.param

benchmark: $(BASE).ncnn.param
	cd $(ROOT) && $(PYTHON) -m save.benchmark --ckpt save/$(CKPT) \
	    --ncnn-param save/$(BASE).ncnn.param --ncnn-bin save/$(BASE).ncnn.bin \
	    $(if $(wildcard $(BASE).fp16.ncnn.param),--ncnn-fp16-param save/$(BASE).fp16.ncnn.param --ncnn-fp16-bin save/$(BASE).fp16.ncnn.bin,) \
	    $(if $(wildcard $(BASE).int8.ncnn.param),--ncnn-int8-param save/$(BASE).int8.ncnn.param --ncnn-int8-bin save/$(BASE).int8.ncnn.bin,) \
	    --n $(BENCH_N)

all: trace fp16 int8 benchmark

# ---------------------------------------------------------------- HF 发布

NS ?= Ismantic
RELEASE ?= Handwritten
CODE_ONLY ?= 0

export:
	cd $(ROOT) && $(PYTHON) -m save.export $(RELEASE)

code:
	cd $(ROOT) && $(PYTHON) -m save.export --code-only $(RELEASE)

verify:
	cd $(ROOT) && $(PYTHON) -m unittest test.test_save -v

dry-run:
	cd $(ROOT) && $(PYTHON) -m save.upload --namespace $(NS) --dry-run \
	    $(if $(filter 1,$(CODE_ONLY)),--code-only,) $(RELEASE)

upload: verify
	cd $(ROOT) && $(PYTHON) -m save.upload --namespace $(NS) $(RELEASE)

upload-code: verify
	cd $(ROOT) && $(PYTHON) -m save.upload --namespace $(NS) --code-only $(RELEASE)

help:
	@echo "模型: make -C save download"
	@echo "NCNN: make -C save status|trace|fp16|int8|benchmark|all"
	@echo "HF:   make -C save export|code|verify|dry-run|upload|upload-code"
