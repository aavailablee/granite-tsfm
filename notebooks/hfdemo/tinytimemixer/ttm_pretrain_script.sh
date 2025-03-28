python ttm_pretrain_sample.py  --context_length 48 \
                               --forecast_length 24 \
                               --patch_length 10 \
                               --batch_size 64 \
                               --num_layers 3 \
                               --decoder_num_layers 3 \
                               --dropout 0.2 \
                               --head_dropout 0.2 \
                               --early_stopping 0 \
                               --adaptive_patching_levels 0 \
                               --num_epochs 100 \
                               --dataset etth1
                               > ./logs/ttm_pretrain_etth2.log 2>&1