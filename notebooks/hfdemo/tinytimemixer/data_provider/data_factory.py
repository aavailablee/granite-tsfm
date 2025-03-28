from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

import pandas as pd


'''
其一，要给个splitConfig
其二，要给data，就是readcsv出来的，原的不需要增删
其三，要给conlumnSpecifier，为TSPreprocesser初始化
先这么多
'''

'''
trainOnly, 是分割数据集时，如果为1，就全部分给train，为0就0.7
Dataset_Pred, 把数据全处理成预测的
label_len, 加点数据通常时pred的1/3~1/2，可使预测误差降低20%-40%，尤其对复杂时间模式（如突变、周期混合）的数据改善显著
'''

'''
1. Specify train/valid/test indices or relative fractions，不行
{
    train: [0, 50],
    valid: [50, 70],
    test:  [70, 100]
}
end value is not inclusive
2. Specify train/test fractions:， 不行
{
    train: 0.7
    test: 0.2
}

split_config = {
    "train": [0, 8640],
    "valid": [8640, 11520],
    "test": [
        11520,
        14400,
    ],
}
or 下面这个是因为数据集不只有14400个才不一致
split_config = {
    "train": [0, 0.6],
    "valid": [0.6, 0.8],
    "test": [
        0.8,
        1,
    ],
}
'''
def data_provider(args):
    if args.dataset == 'etth1':
        timestamp_column = "date"
        id_columns = []  # mention the ids that uniquely identify a time-series. 比如自增列，最后会删去的
        target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }
        # data = pd.read_csv(args.root_path + args.data_path) # todo 这里拿绝对地址卡死
        dataset_path = "/home/xiaofuqiang/all_six_datasets/ETT-small/ETTh1.csv"
        data = pd.read_csv(
            dataset_path,
            parse_dates=[timestamp_column],
        )
        # mention the train, valid and split config.
        split_config = {
            "train": [0, 8640],
            "valid": [8640, 11520],
            "test": [
                11520,
                14400,
            ],
        }
    elif args.dataset == 'etth2':
        timestamp_column = "date"
        id_columns = []  # mention the ids that uniquely identify a time-series. 比如自增列，最后会删去的
        target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }
        # data = pd.read_csv(args.root_path + args.data_path) # todo 这里拿绝对地址卡死
        dataset_path = "/home/xiaofuqiang/all_six_datasets/ETT-small/ETTh2.csv"
        data = pd.read_csv(
            dataset_path,
            parse_dates=[timestamp_column],
        )
        # mention the train, valid and split config.
        split_config = {
            "train": [0, 8640],
            "valid": [8640, 11520],
            "test": [
                11520,
                14400,
            ],
        }
    elif args.dataset == 'ettm1':
        timestamp_column = "date"
        id_columns = []  # mention the ids that uniquely identify a time-series. 比如自增列，最后会删去的
        target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }
        # data = pd.read_csv(args.root_path + args.data_path) # todo 这里拿绝对地址卡死
        dataset_path = "/home/xiaofuqiang/all_six_datasets/ETT-small/ETTm1.csv"
        data = pd.read_csv(
            dataset_path,
            parse_dates=[timestamp_column],
        )
        # mention the train, valid and split config.
        split_config = {
            "train": [0, 12*30*24*4],
            "valid": [12*30*24*4, 16*30*24*4],
            "test": [
                16*30*24*4,
                20*30*24*4,
            ],
        }
    elif args.dataset == 'ettm2':
        timestamp_column = "date"
        id_columns = []  # mention the ids that uniquely identify a time-series. 比如自增列，最后会删去的
        target_columns = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        column_specifiers = {
            "timestamp_column": timestamp_column,
            "id_columns": id_columns,
            "target_columns": target_columns,
            "control_columns": [],
        }
        # data = pd.read_csv(args.root_path + args.data_path) # todo 这里拿绝对地址卡死
        dataset_path = "/home/xiaofuqiang/all_six_datasets/ETT-small/ETTm2.csv"
        data = pd.read_csv(
            dataset_path,
            parse_dates=[timestamp_column],
        )
        # mention the train, valid and split config.
        split_config = {
            "train": [0, 12*30*24*4],
            "valid": [12*30*24*4, 16*30*24*4],
            "test": [
                16*30*24*4,
                20*30*24*4,
            ],
        }
    else:
        raise NotImplementedError
    return data, split_config, column_specifiers
