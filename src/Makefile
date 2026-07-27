-include ../local.mk

PYTHON ?= python
ROOT := ..
SMOKE_DIR ?= runs/smoke
RUN_DIR ?= runs/mbv2

.PHONY: preprocess smoke train help

preprocess:
	$(MAKE) -C cpp

smoke:
	cd $(ROOT) && $(PYTHON) -m src.train --epochs 1 --subset-frac 0.05 \
	    --batch-size 128 --num-workers 4 --log-every 20 \
	    --max-train-steps 20 --max-eval-batches 20 --out-dir $(SMOKE_DIR)

train:
	cd $(ROOT) && $(PYTHON) -m src.train --model mobilenet_v2 --epochs 30 \
	    --batch-size 256 --warmup-epochs 3 --min-lr 1e-5 --weight-decay 1e-4 \
	    --augment true --out-dir $(RUN_DIR)

help:
	@echo "make -C src preprocess|smoke|train"
