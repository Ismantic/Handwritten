-include ../local.mk

PYTHON ?= python
ROOT := ..
RAW := $(ROOT)/data/raw
OUT := $(ROOT)/data
CHARSET := $(OUT)/processed/charset.json
VIZ_N ?= 64

.PHONY: status charset npy visualize verify all help

status:
	@for f in $(CHARSET) $(OUT)/npy/train/images.npy $(OUT)/npy/test/images.npy; do \
	    if test -f "$$f"; then echo "[ok]      $$f"; else echo "[missing] $$f"; fi; \
	done

charset: $(CHARSET)

$(CHARSET): build_charset.py gnt_parser.py $(RAW)/.train_extracted
	cd $(ROOT) && $(PYTHON) -m prepare.build_charset \
	    --gnt-dir data/raw/train_gnt --out data/processed/charset.json

$(OUT)/npy/train/images.npy: build_npy.py gnt_parser.py $(CHARSET) $(RAW)/.train_extracted
	cd $(ROOT) && $(PYTHON) -m prepare.build_npy \
	    --gnt-dir data/raw/train_gnt --charset data/processed/charset.json --out-dir data/npy/train

$(OUT)/npy/test/images.npy: build_npy.py gnt_parser.py $(CHARSET) $(RAW)/.test_extracted
	cd $(ROOT) && $(PYTHON) -m prepare.build_npy \
	    --gnt-dir data/raw/test_gnt --charset data/processed/charset.json --out-dir data/npy/test

npy: $(OUT)/npy/train/images.npy $(OUT)/npy/test/images.npy

visualize: $(CHARSET)
	cd $(ROOT) && $(PYTHON) -m prepare.visualize \
	    --gnt-dir data/raw/train_gnt --charset data/processed/charset.json \
	    --out-dir data/processed/visualize --n $(VIZ_N)

verify:
	cd $(ROOT) && $(PYTHON) -m prepare.verify_npy \
	    --npy-dir data/npy/train --charset data/processed/charset.json \
	    --out data/processed/verify/train_grid.png --n $(VIZ_N)
	cd $(ROOT) && $(PYTHON) -m prepare.verify_npy \
	    --npy-dir data/npy/test --charset data/processed/charset.json \
	    --out data/processed/verify/test_grid.png --n $(VIZ_N)

all: npy verify

help:
	@echo "make -C prepare status|charset|npy|visualize|verify|all"
