#!/usr/bin/env bash
# NCNN FP32 → INT8 量化(ncnn2table 校准 + ncnn2int8 转换),支持选择性跳过敏感层。
#
# 用法:
#   ./quantize_int8.sh <ncnn_param> <ncnn_bin> <calib_list> <out_param> <out_bin> [<table>]
#
# 环境变量:
#   METHOD       校准方法(kl / aciq / eq),默认 kl
#   SKIP_LAYERS  egrep 模式,匹配 layer 名的行从校准表里删掉,
#                让这些层 fallback 到 FP32(对量化敏感的首层 conv / 分类头特别有用)。
#                默认空(全模型 INT8)。
#                例:SKIP_LAYERS='convclip_0|convdwclip_0' 跳过首层 conv 和 dw conv
#                例:SKIP_LAYERS='convclip_0|convdwclip_0|linear_36' 再跳过分类头

set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "usage: $0 <ncnn_param> <ncnn_bin> <calib_list> <out_param> <out_bin> [<table>]" >&2
    exit 1
fi

PARAM=$1
BIN=$2
LIST=$3
OUT_PARAM=$4
OUT_BIN=$5
TABLE=${6:-${OUT_PARAM%.param}.table}
RAW_TABLE="${TABLE%.table}.raw.table"

METHOD=${METHOD:-kl}
SKIP_LAYERS=${SKIP_LAYERS:-}

echo "[quant] 校准:$PARAM + $LIST → $RAW_TABLE  (method=$METHOD)"
ncnn2table "$PARAM" "$BIN" "$LIST" "$RAW_TABLE" \
    shape=[64,64,1] \
    type=1 \
    method="$METHOD" \
    thread=8

if [[ -n "$SKIP_LAYERS" ]]; then
    # 删掉匹配 SKIP_LAYERS 的行(包括 _param_0 weight scale 行 + activation scale 行)
    PATTERN="^($SKIP_LAYERS)(_param_0)?( |\$)"
    echo "[quant] 跳过敏感层(fallback FP32):$SKIP_LAYERS"
    grep -vE "$PATTERN" "$RAW_TABLE" > "$TABLE"
    echo "  原 $(wc -l < "$RAW_TABLE") 行 → 过滤后 $(wc -l < "$TABLE") 行"
else
    cp "$RAW_TABLE" "$TABLE"
fi

echo "[quant] 量化:$PARAM → $OUT_PARAM (INT8)"
ncnn2int8 "$PARAM" "$BIN" "$OUT_PARAM" "$OUT_BIN" "$TABLE" > /dev/null

echo "[quant] OK"
ls -lh "$OUT_PARAM" "$OUT_BIN" "$TABLE"
