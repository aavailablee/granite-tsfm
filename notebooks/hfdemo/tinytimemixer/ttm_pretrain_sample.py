#!/usr/bin/env python
# coding: utf-8

import logging
import math
import os
import tempfile

import sys
sys.path.append(os.path.abspath("/opt/data/private/model_test/granite-tsfm"))

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.models.tinytimemixer import (
    TinyTimeMixerConfig,
    TinyTimeMixerForPrediction,
)
from tsfm_public.models.tinytimemixer.utils import get_ttm_args
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions


from data_provider.data_factory import data_provider


logger = logging.getLogger(__file__)

# TTM pre-training example.
# This scrips provides a toy example to pretrain a Tiny Time Mixer (TTM) model on
# the `etth1` dataset. For pre-training TTM on a much large set of datasets, please
# have a look at our paper: https://arxiv.org/pdf/2401.03955.pdf
# If you want to directly utilize the pre-trained models. Please use them from the
# Hugging Face Hub: https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1
# Have a look at the fine-tune scripts for example usecases of the pre-trained
# TTM models.

# Basic usage:
# python ttm_pretrain_sample.py --data_root_path datasets/
# See the get_ttm_args() function to know more about other TTM arguments

from torch.optim.lr_scheduler import _LRScheduler

class ExponentialEpochScheduler(_LRScheduler):
    """
    指数衰减调度器 - 每个 epoch 后更新一次学习率
    完全兼容 PyTorch 的调度器接口
    """
    def __init__(self, optimizer, gamma, steps_per_epoch, last_epoch=-1):
        """
        Args:
            optimizer: 优化器对象
            gamma: 每 epoch 的衰减系数
            steps_per_epoch: 每个 epoch 的步数 (batch 数)
            last_epoch: 最后一次 epoch 索引
        """
        self.gamma = gamma
        self.steps_per_epoch = steps_per_epoch
        self.global_step_count = 1  # 跟踪全局步数
        super(ExponentialEpochScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        返回当前学习率 - 基于当前 epoch 计算
        """
        # 计算当前 epoch
        current_epoch = self.global_step_count // self.steps_per_epoch
        
        # 应用指数衰减
        return [base_lr * (self.gamma ** current_epoch) 
                for base_lr in self.base_lrs]
    
    def step(self):
        """
        每次 batch 后调用 - 跟踪步数，在 epoch 结束时更新学习率
        """
        self.global_step_count += 1
        current_step = self.global_step_count
        
        # 检查是否完成一个 epoch
        if current_step % self.steps_per_epoch == 0:
            # 调用父类的 step() 来更新 _last_lr 和优化器的学习率
            super().step()  
            
            # 调试输出
            current_epoch = current_step // self.steps_per_epoch
            print(f"\nEpoch {current_epoch} complete!")
            print(f"Updating LR to: {self.get_lr()[0]:.6f}")
        else:
            # 确保在非 epoch 结束时也更新状态
            # 但不实际改变学习率（维持上次设置的值）
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def get_base_model(args):
    # Pre-train a `TTM` forecasting model
    config = TinyTimeMixerConfig(
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        patch_length=args.patch_length,
        num_input_channels=args.enc_in,
        patch_stride=args.patch_length,
        d_model=args.d_model,
        num_layers=args.num_layers,  # increase the number of layers if we want more complex models
        mode=args.encoder_channel,
        expansion_factor=2,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        scaling="std",
        gated_attn=True,
        adaptive_patching_levels=args.adaptive_patching_levels,
        # decoder params
        decoder_num_layers=args.decoder_num_layers,  # increase the number of layers if we want more complex models
        decoder_adaptive_patching_levels=0,
        decoder_mode=args.decoder_channel,
        # decoder_mode="common_channel",
        # decoder_mode="mix_channel",
        decoder_raw_residual=False,
        use_decoder=True,
        decoder_d_model=args.decoder_d_model,
        loss=args.loss,

        enc_in=args.enc_in,
        bsa = args.bsa,
        batch_size=args.batch_size,
    )

    model = TinyTimeMixerForPrediction(config)
    return model


def pretrain(args, model, dset_train, dset_val):
    # Find optimal learning rate
    # Use with caution: Set it manually if the suggested learning rate is not suitable

    learning_rate, model = optimal_lr_finder(
        model,
        dset_train,
        batch_size=args.batch_size,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    # learning_rate = args.learning_rate

    trainer_args = TrainingArguments(
        output_dir=os.path.join(args.save_dir, "checkpoint"),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        seed=args.random_seed,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(args.save_dir, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        disable_tqdm=True,
    )

    # Optimizer and scheduler
    # learning_rate = 0.0001
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 0.0001 * 0.9 exponential
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        pct_start=0.3,
        epochs=args.num_epochs,
        steps_per_epoch=math.ceil(len(dset_train) / args.batch_size),
        # steps_per_epoch=math.ceil(len(dset_train) / (args.batch_size * args.num_gpus)),
    )
    # 创建一个恒等学习率调度器（每一步都返回初始学习率）
    # import torch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: 1.0  # 始终返回乘数1.0，即学习率不变
    # )
    # 替换原来的 OneCycleLR
    # steps_per_epoch = math.ceil(len(dset_train) / args.batch_size)

    # scheduler = ExponentialEpochScheduler(
    #     optimizer,
    #     gamma=0.8,  # 每个 epoch 的衰减系数
    #     steps_per_epoch=steps_per_epoch
    # )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )

    # Set trainer
    if args.early_stopping:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
            callbacks=[early_stopping_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=dset_train,
            eval_dataset=dset_val,
            optimizers=(optimizer, scheduler),
        )

    # Train
    trainer.train()

    # Save the pretrained model

    model_save_path = os.path.join(args.save_dir, "ttm_pretrained")
    trainer.save_model(model_save_path)
    return model_save_path


def inference(args, model_path, dset_test):
    model = get_model(model_path=model_path)

    temp_dir = tempfile.mkdtemp()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=args.batch_size,
            seed=args.random_seed,
            report_to="none",
            disable_tqdm=True,
        ),
    )
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE output:", "+" * 20)
    output = trainer.evaluate(dset_test)
    print(output)

    # get predictions

    predictions_dict = trainer.predict(dset_test)

    predictions_np = predictions_dict.predictions[0]

    print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    backbone_embedding = predictions_dict.predictions[1]

    print(backbone_embedding.shape)

    plot_path = os.path.join(args.save_dir, "plots")
    # plot
    plot_predictions(
        model=trainer.model,
        dset=dset_test,
        plot_dir=plot_path,
        plot_prefix="test_inference",
        channel=0,
    )
    print("Plots saved in location:", plot_path)


if __name__ == "__main__":
    # Arguments
    args = get_ttm_args()
    # args.batch_size = 4096

    # Set seed
    set_seed(args.random_seed)

    logger.info(
        f"{'*' * 20} Pre-training a TTM for context len = {args.context_length}, forecast len = {args.forecast_length} {'*' * 20}"
    )

    if args.dataset == "ice":
        pass
    else:
        data, split_config, column_specifiers = data_provider(args)

        tsp = TimeSeriesPreprocessor(
            **column_specifiers,
            context_length=args.context_length,
            prediction_length=args.forecast_length,
            scaling=True,
            encode_categorical=False,
            scaler_type="standard",
        )

    if args.dataset == "ice":
        from data_provider.scaler import init_scaler
        from data_provider.data_loader import AttrMapper, BSIDMapper, Ice
        from torch.utils.data import DataLoader
        xscaler = init_scaler("standard")
        yscaler = init_scaler("standard")
        datasets = {}
        # dataloaders = {}
        transformer = AttrMapper()
        id_transformer = BSIDMapper()
        cfg = {
            "dataset": {
                "have_weather_forecast": False,
                "data_path": "/opt/data/private/model_test/PatchTST/PatchTST_supervised/dataset/all_ice/",
                
            },
            "batch_size": 64,
            "num_workers": 1,
        }
        class obj(object):
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(k, (list, tuple)):
                        setattr(
                            self,
                            k,
                            [obj(x) if isinstance(x, dict) else x for x in v],
                        )
                    else:
                        setattr(self, k, obj(v) if isinstance(v, dict) else v)
        cfg = obj(cfg)
        for category in ["train", "val", "test"]:
            datasets[category] = Ice(cfg.dataset, category, transformer, id_transformer, xscaler, yscaler)
            # if category == "test":
            #     d=datasets[category].bsid
        dset_train, dset_valid, dset_test = (
            datasets["train"],
            datasets["val"],
            datasets["test"],
        )
    else:
        dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)

    # Get model
    model = get_base_model(args)
    # print(model)
    # exit()

    # Pretrain，本来就是调用Trainer，只要dataset是getItem的都行，管你自定义什么臭x gun！
    model_save_path = pretrain(args, model, dset_train, dset_valid)
    print("=" * 20, "Pretraining Completed!", "=" * 20)
    print("Model saved in location:", model_save_path)

    # inference

    inference(args=args, model_path=model_save_path, dset_test=dset_test)

    print("inference completed..")
