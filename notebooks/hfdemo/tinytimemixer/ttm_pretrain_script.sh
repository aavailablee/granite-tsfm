python ttm_pretrain_sample.py  --context_length 48 \
                               --forecast_length 24 \
                               --patch_length 10 \
                               --batch_size 64 \
                               --num_layers 3 \
                               --decoder_num_layers 3 \
                               --dropout 0.2 \
                               --head_dropout 0.2 \
                               --early_stopping 0 \
                               --enc_in 862 \
                               --loss mae \
                               --adaptive_patching_levels 0 \
                               --num_epochs 10 \
                               --dataset traffic \
                            #    > ./logs/ttm_pretrain_etth1_250329_noScaler.log 2>&1