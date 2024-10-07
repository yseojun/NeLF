#!/bin/bash

if [ $# -ne 3 ]; then
    echo "사용법: $0 <exp_name> <mlp_depth> <mlp_width>"
    exit 1
fi

exp_name=$1
mlp_depth=$2
mlp_width=$3

python transform_onnx.py --exp_name "${exp_name}_${mlp_depth}_${mlp_width}" --mlp_depth $mlp_depth --mlp_width $mlp_width