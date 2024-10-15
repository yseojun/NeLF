#!/bin/bash

# 인자 확인
if [ "$#" -ne 3 ]; then
    echo "사용법: $0 <데이터셋_이름> <mlp_depth> <mlp_width>"
    exit 1
fi

# 인자 할당
dataset_name=$1
mlp_depth=$2
mlp_width=$3

# 실행할 명령어 구성
exp_name="${dataset_name}_${mlp_depth}_${mlp_width}"
data_dir="/data/hmjung/data_backup/NeuLF_rgb/dataset/stanford_half/${dataset_name}/"

# Python 스크립트 실행
python src/train_neulf.py \
    --exp_name "$exp_name" \
    --data_dir "$data_dir" \
    --gpuid 0 \
    --factor 1 \
    --mlp_depth "$mlp_depth" \
    --mlp_width "$mlp_width"

echo "학습이 완료되었습니다."
