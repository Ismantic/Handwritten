# CASIA HWDB1.1 原始数据。数据许可不允许本仓库代为分发,需从官方申请下载。

TRAIN_ZIP := HWDB1.1trn_gnt.zip
TEST_ZIP  := HWDB1.1tst_gnt.zip
RAW_DIR   := raw
TRAIN_TAG := $(RAW_DIR)/.train_extracted
TEST_TAG  := $(RAW_DIR)/.test_extracted

.PHONY: status extract clean help

status:
	@for f in $(TRAIN_ZIP) $(TEST_ZIP); do \
	    if test -f "$$f"; then echo "[ok]      data/$$f"; else echo "[missing] data/$$f"; fi; \
	done
	@for d in $(RAW_DIR)/train_gnt $(RAW_DIR)/test_gnt; do \
	    if test -d "$$d"; then echo "[ok]      data/$$d"; else echo "[missing] data/$$d"; fi; \
	done

extract: $(TRAIN_TAG) $(TEST_TAG)

$(TEST_TAG): $(TEST_ZIP)
	mkdir -p $(RAW_DIR)/test_gnt
	unzip -q -o $< -d $(RAW_DIR)/test_gnt
	@touch $@

$(TRAIN_TAG): $(TRAIN_ZIP)
	mkdir -p $(RAW_DIR)/.train_alz_tmp $(RAW_DIR)/train_gnt
	unzip -q -o $< -d $(RAW_DIR)/.train_alz_tmp
	unar -q -f -o $(RAW_DIR)/train_gnt -D $(RAW_DIR)/.train_alz_tmp/HWDB1.1trn_gnt.alz
	rm -rf $(RAW_DIR)/.train_alz_tmp
	@touch $@

clean:
	@echo "不会自动删除 data/raw(可能有数 GB 且无法从仓库恢复)。"
	@echo "确认后请手动删除目标目录。"

help:
	@echo "make -C data status   检查授权数据是否就绪"
	@echo "make -C data extract  解压官方 HWDB1.1 压缩包"
