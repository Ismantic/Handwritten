# 模型导出 pipeline:PyTorch ckpt → NCNN(FP32 / INT8)
#
# 入口 target:
#   make trace         PyTorch ckpt → traced.pt → pnnx → .ncnn.param/.bin (FP32)
#   make benchmark     PyTorch vs NCNN(FP32)精度 + 延迟对比
#   make all           trace + benchmark
#   make clean         删 output/

PYTHON  := /home/tfbao/.venv/bin/python
ROOT    := ..
SCRIPTS := scripts

# 默认目标:V2 + 增强版本
RUN_NAME ?= mbv2_phase1_aug
CKPT     ?= runs/$(RUN_NAME)/best.pt
NAME     ?= mbv2_aug
OUT_DIR  := output

NCNN_PARAM := $(OUT_DIR)/$(NAME).ncnn.param
NCNN_BIN   := $(OUT_DIR)/$(NAME).ncnn.bin

# 精度对比采样数(全测试集 22w 太慢)
BENCH_N ?= 5000

.PHONY: all trace benchmark clean help

all: trace benchmark

trace: output/.trace.$(NAME).done

output/.trace.$(NAME).done: $(SCRIPTS)/trace_and_pnnx.py
	@echo "[export] $(RUN_NAME) → NCNN..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/trace_and_pnnx.py \
	    --ckpt $(CKPT) \
	    --out-dir export/$(OUT_DIR) \
	    --name $(NAME)
	@touch $@

benchmark: output/.trace.$(NAME).done $(SCRIPTS)/benchmark.py
	@echo "[export] PyTorch vs NCNN(FP32)精度+延迟..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/benchmark.py \
	    --ckpt $(CKPT) \
	    --ncnn-param export/$(NCNN_PARAM) \
	    --ncnn-bin export/$(NCNN_BIN) \
	    --n $(BENCH_N)

clean:
	rm -rf $(OUT_DIR)

help:
	@echo "导出 pipeline targets:"
	@echo "  make trace         RUN_NAME=$(RUN_NAME) → NCNN FP32"
	@echo "  make benchmark     PyTorch vs NCNN 精度+延迟,样本数 BENCH_N=$(BENCH_N)"
	@echo "  make all           trace + benchmark"
	@echo "  make clean         删 output/"
	@echo ""
	@echo "切换模型:make trace RUN_NAME=plain_30ep_aug NAME=plain_aug"
