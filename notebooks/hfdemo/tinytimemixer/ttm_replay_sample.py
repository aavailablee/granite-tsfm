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
from torch.optim.lr_scheduler import OneCycleLR
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


logger = logging.getLogger(__file__)


# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# TTM Model path. The default model path is Granite-R2. Below, you can choose other TTM releases.
# TTM_MODEL_PATH = "/home/xiaofuqiang/repo/granite-tsfm/notebooks/hfdemo/tinytimemixer/tmp/TTM_cl-48_fl-24_pl-10_apl-0_ne-100_es-False_bs-64_noMomentum-02/ttm_pretrained"
# TTM_MODEL_PATH = "/opt/data/private/model_test/granite-tsfm/notebooks/hfdemo/tinytimemixer/tmp/MS_MSE_decMix_20250529-231920_ice_cl-48_fl-24_pl-16_ne-10/ttm_pretrained"
TTM_MODEL_PATH = "/opt/data/private/model_test/granite-tsfm/notebooks/hfdemo/tinytimemixer/tmp/20250609-222957_etth1_cl-48_fl-24_pl-12_ne-10/ttm_pretrained"


# Context length, Or Length of the history.
# Currently supported values are: 512/1024/1536 for Granite-TTM-R2 and Research-Use-TTM-R2, and 512/1024 for Granite-TTM-R1
CONTEXT_LENGTH = 48

# Granite-TTM-R2 supports forecast length upto 720 and Granite-TTM-R1 supports forecast length upto 96
PREDICTION_LENGTH = 24

TARGET_DATASET = "etth1"
# dataset_path = "/home/xiaofuqiang/repo/granite-tsfm/notebooks/hfdemo/tinytimemixer/ETTh1.csv"


# Results dir
OUT_DIR = "./ttm_finetuned_models/"


timestamp_column = "date"
id_columns = []  # mention the ids that uniquely identify a time-series.

# target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
# split_config = {
#     "train": [0, 8640],
#     "valid": [8640, 11520],
#     "test": [
#         11520,
#         14400,
#     ],
# }
# Understanding the split config -- slides

# data = pd.read_csv(
#     dataset_path,
#     parse_dates=[timestamp_column],
# )

# column_specifiers = {
#     "timestamp_column": timestamp_column,
#     "id_columns": id_columns,
#     "target_columns": target_columns,
#     "control_columns": [],
# }


def zeroshot_eval(dataset_name, batch_size, context_length=48, forecast_length=24,
        loss="mse",):
    # Get data

    # tsp = TimeSeriesPreprocessor(
    #     **column_specifiers,
    #     context_length=context_length,
    #     prediction_length=forecast_length,
    #     scaling=True,
    #     encode_categorical=False,
    #     scaler_type="standard",
    # )

    # dset_train, dset_valid, dset_test = get_datasets(tsp, data, split_config)
    from data_provider.scaler import init_scaler
    from data_provider.data_loader import AttrMapper, BSIDMapper, Ice
    from torch.utils.data import DataLoader
    xscaler = init_scaler("standard")
    yscaler = init_scaler("standard")
    datasets = {}
    # dataloaders = {}
    transformer = AttrMapper()
    id_transformer = BSIDMapper()
    # weat_info_true: False  # 是否加载天气信息
    # weat_dim: 3  # 输入维度（天气）
    # attribute_true: True  # 是否添加导体属性信息
    # topo_true: False  # 是否添加地形类别
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

    # Load model
    zeroshot_model = get_model(TTM_MODEL_PATH, context_length=context_length, prediction_length=forecast_length)

    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            seed=SEED,
            report_to="none"
        ),
    )
    zeroshot_trainer.model.loss = loss
    # evaluate = zero-shot performance
    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print(zeroshot_output)

    # get predictions

    predictions_dict = zeroshot_trainer.predict(dset_test)

    predictions_np = predictions_dict.predictions[0]

    print(predictions_np.shape)

    # get backbone embeddings (if needed for further analysis)

    backbone_embedding = predictions_dict.predictions[1]

    print(backbone_embedding.shape)

    # plot
    plot_predictions(
        model=zeroshot_trainer.model,
        dset=dset_test,
        plot_dir=os.path.join(OUT_DIR, dataset_name),
        plot_prefix="test_zeroshotap3",
        indices=[0, 2, 4],
        channel=0,
    )

zeroshot_eval(
    dataset_name=TARGET_DATASET,
    context_length=CONTEXT_LENGTH,
    forecast_length=PREDICTION_LENGTH,
    batch_size=64,
    loss="mae",
)
