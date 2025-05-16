#!/bin/sh

# 创建日志目录
mkdir -p ./logs

# 基础配置
model_name="etth1"
seq_len=48

# 参数组合循环（sh兼容写法）
pred_lengths="24 72"
loss_types="mse mae"

for pred_length in $pred_lengths; do
    for loss_type in $loss_types; do
        echo "正在训练：pred_length=${pred_length}, loss=${loss_type}"
        start_time=$(date +%s)
        
        log_name="ttmb_${model_name}_${seq_len}_${pred_length}_${loss_type}.log"
        
        python ttm_pretrain_sample.py \
            --context_length $seq_len \
            --forecast_length $pred_length \
            --patch_length 12 \
            --batch_size 64 \
            --num_layers 3 \
            --decoder_num_layers 3 \
            --dropout 0.2 \
            --head_dropout 0.2 \
            --early_stopping 0 \
            --enc_in 7 \
            --loss $loss_type \
            --adaptive_patching_levels 0 \
            --num_epochs 10 \
            --dataset "$model_name" \
            > "./logs/${log_name}" 2>&1
            
        echo "已完成：${log_name}"
        end_time=$(date +%s)
        echo "耗时: $((end_time - start_time)) 秒"
    done
done