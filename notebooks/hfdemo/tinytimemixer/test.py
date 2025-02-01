import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

class CustomDataCollator:
    def __call__(self, features):
        # features: 这是一个批次的数据列表，每个数据都是字典形式，例如：{'x': tensor, 'label': tensor}
        input_data = [f['x'] for f in features]  # 获取所有 'x'
        labels = [f['label'] for f in features]  # 获取所有 'label'
        
        input_data = torch.stack(input_data)  # 拼接成 (batch_size, contextLength, input_dim)
        labels = torch.stack(labels)  # 拼接成 (batch_size, predictLength)
        
        return {'x': input_data, 'labels': labels}  # 返回模型需要的格式，labels 将作为目标值传递给模型



# 1. 读取并预处理数据
def preprocess_data(file_path='./ETTh1.csv'):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # 提取时间特征并进行周期性编码
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday

    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_weekday'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['cos_weekday'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    df.drop(columns=['date', 'hour', 'weekday'], inplace=True)
    return df

# 2. 拆分数据集
def split_data(df, train_ratio=0.8, val_ratio=0.1):
    dataset = Dataset.from_pandas(df)
    
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
    # print(test_dataset)
    # exit()
    
    return train_dataset, val_dataset, test_dataset

# 3. 定义自定义MLP模型
class CustomMLPModel(nn.Module):
    def __init__(self, input_dim, contextLength=48, predictLength=24):
        super(CustomMLPModel, self).__init__()
        self.flatten = nn.Flatten()  # 展开输入序列
        self.fc1 = nn.Linear(contextLength * input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, predictLength)  # 输出预测的时间序列长度

    def forward(self, x):
        x = self.flatten(x) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出 (batch_size, predictLength)
        return x

# 4.
class TimeSeriesDataset(TorchDataset):
    def __init__(self, data, contextLength=48, predictLength=24, target_col='OT'):
        self.data = data
        self.contextLength = contextLength
        self.predictLength = predictLength
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.contextLength - self.predictLength

    def __getitem__(self, idx):
        # 获取输入序列和目标值
        x = self.data.iloc[idx: idx + self.contextLength].values
        y = self.data.iloc[idx + self.contextLength: idx + self.contextLength + self.predictLength][self.target_col].values
        
        ret = {
            'x': torch.tensor(x, dtype=torch.float32),  # 输入特征
            'label': torch.tensor(y, dtype=torch.float32)  # 目标值
        }
        # 返回字典，包含模型期望的输入键
        return ret


# 5. 配置训练
def setup_training(train_dataset, eval_dataset, model, output_dir='./results', num_epochs=10):
    training_args = TrainingArguments(
        output_dir=output_dir,  
        evaluation_strategy="epoch",  
        learning_rate=1e-5,  
        per_device_train_batch_size=64,  
        per_device_eval_batch_size=64,  
        num_train_epochs=num_epochs,  
        weight_decay=0.01,  
        logging_dir='./logs',  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer

# 6. 主函数执行流程
def main(file_path='./ETTh1.csv', contextLength=48, predictLength=24):
    # 数据处理
    df = preprocess_data(file_path)
    
    # 数据集拆分
    train_dataset, val_dataset, test_dataset = split_data(df)
    
    # 转换为TimeSeriesDataset
    train_data = train_dataset.to_pandas()
    val_data = val_dataset.to_pandas()
    
    train_dataset = TimeSeriesDataset(train_data, contextLength, predictLength)
    eval_dataset = TimeSeriesDataset(val_data, contextLength, predictLength)

    # 模型初始化
    input_dim = train_data.shape[1]  # 数据的列数，作为输入维度
    model = CustomMLPModel(input_dim=input_dim, contextLength=contextLength, predictLength=predictLength)

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置训练器
    trainer = setup_training(train_dataset, eval_dataset, model)

    # 训练模型
    trainer.train()

    # 评估模型
    results = trainer.evaluate(test_dataset)
    print(results)

if __name__ == '__main__':
    main()
