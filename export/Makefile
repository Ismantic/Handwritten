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
FP16_PARAM := $(OUT_DIR)/$(NAME).fp16.ncnn.param
FP16_BIN   := $(OUT_DIR)/$(NAME).fp16.ncnn.bin
INT8_PARAM := $(OUT_DIR)/$(NAME).int8.ncnn.param
INT8_BIN   := $(OUT_DIR)/$(NAME).int8.ncnn.bin
CALIB_DIR  := $(OUT_DIR)/calib_$(NAME)
CALIB_LIST := $(OUT_DIR)/$(NAME).calib_list.txt
CALIB_TAG  := $(OUT_DIR)/.calib.$(NAME).done

# 精度对比采样数(全测试集 22w 太慢)
BENCH_N    ?= 5000
# 校准数据张数(实测 1000→2000 让损失从 -0.20 → -0.08 点)
CALIB_N    ?= 2000

# 量化时让哪些层 fallback FP32(对首层 conv 量化敏感,跳过后精度从 -2.84 → -0.08)
# 默认按 V2 改造的 PNNX 命名跳过首层 conv 和首 DW conv。其它模型可能要改这个 pattern。
INT8_SKIP_LAYERS ?= convclip_0|convdwclip_0

.PHONY: all trace fp16 int8 benchmark clean help

all: trace fp16 int8 benchmark

trace: output/.trace.$(NAME).done

output/.trace.$(NAME).done: $(SCRIPTS)/trace_and_pnnx.py
	@echo "[export] $(RUN_NAME) → NCNN..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/trace_and_pnnx.py \
	    --ckpt $(CKPT) \
	    --out-dir export/$(OUT_DIR) \
	    --name $(NAME)
	@touch $@

# FP16:重新跑 pnnx 加 fp16 模式(模型权重砍半,推理延迟接近 INT8 但精度通常无损)
fp16: output/.trace.$(NAME).fp16.done

output/.trace.$(NAME).fp16.done: $(SCRIPTS)/trace_and_pnnx.py
	@echo "[export] $(RUN_NAME) → NCNN FP16..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/trace_and_pnnx.py \
	    --ckpt $(CKPT) \
	    --out-dir export/$(OUT_DIR) \
	    --name $(NAME).fp16 \
	    --fp16
	@touch $@

# INT8 量化:校准数据 → ncnn2table → ncnn2int8
int8: $(INT8_PARAM)

$(CALIB_TAG): $(SCRIPTS)/make_calib.py
	@echo "[export] 生成 $(CALIB_N) 张 INT8 校准 npy..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/make_calib.py \
	    --out-dir export/$(CALIB_DIR) \
	    --list export/$(CALIB_LIST) \
	    --n $(CALIB_N)
	@touch $@

$(INT8_PARAM): output/.trace.$(NAME).done $(CALIB_TAG) $(SCRIPTS)/quantize_int8.sh
	@echo "[export] ncnn2table + ncnn2int8 → INT8 模型(skip layers: $(INT8_SKIP_LAYERS))..."
	SKIP_LAYERS='$(INT8_SKIP_LAYERS)' $(SCRIPTS)/quantize_int8.sh \
	    $(NCNN_PARAM) \
	    $(NCNN_BIN) \
	    $(CALIB_LIST) \
	    $(INT8_PARAM) \
	    $(INT8_BIN)

benchmark: output/.trace.$(NAME).done $(SCRIPTS)/benchmark.py
	@echo "[export] PyTorch vs NCNN(FP32 / FP16 / INT8)精度+延迟..."
	cd $(ROOT) && $(PYTHON) export/$(SCRIPTS)/benchmark.py \
	    --ckpt $(CKPT) \
	    --ncnn-param export/$(NCNN_PARAM) \
	    --ncnn-bin export/$(NCNN_BIN) \
	    $(if $(wildcard $(FP16_PARAM)),--ncnn-fp16-param export/$(FP16_PARAM) --ncnn-fp16-bin export/$(FP16_BIN),) \
	    $(if $(wildcard $(INT8_PARAM)),--ncnn-int8-param export/$(INT8_PARAM) --ncnn-int8-bin export/$(INT8_BIN),) \
	    --n $(BENCH_N)

clean:
	rm -rf $(OUT_DIR)

help:
	@echo "导出 pipeline targets:"
	@echo "  make trace         RUN_NAME=$(RUN_NAME) → NCNN FP32"
	@echo "  make int8          ncnn2table 校准 + ncnn2int8 量化(CALIB_N=$(CALIB_N))"
	@echo "  make benchmark     PyTorch vs NCNN(FP32 / INT8 if exists)精度+延迟"
	@echo "  make all           trace + int8 + benchmark"
	@echo "  make clean         删 output/"
	@echo ""
	@echo "切换模型:make trace RUN_NAME=plain_30ep_aug NAME=plain_aug"
