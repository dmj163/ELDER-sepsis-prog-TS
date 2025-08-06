import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_log_error
from src.data.SepsisDatset import TimeSeriesDataset, collate_fn
from torch.utils.data import DataLoader
from src.models.models import LSTM

from tqdm import tqdm
from src.utils.load_config import load_config
config = load_config()


# 创建数据集和数据加载器
train_dataset = TimeSeriesDataset(index_label_file=config['data']['index_label_mimic_train_path'],
    time_series_data_file=config['data']['full_mimic_data_path'],
    drop_columns=config['data']['drop_columns'])
val_dataset = TimeSeriesDataset(index_label_file=config['data']['index_label_mimic_val_path'],
    time_series_data_file=config['data']['full_mimic_data_path'],
    drop_columns=config['data']['drop_columns'])
test_dataset = TimeSeriesDataset(index_label_file=config['data']['index_label_mimic_test_path'],
    time_series_data_file=config['data']['full_mimic_data_path'],
    drop_columns=config['data']['drop_columns'])

train_loader = DataLoader(
    train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn,
    num_workers=config['training']['num_workers']
)
val_loader = DataLoader(
    val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn,
    num_workers=config['training']['num_workers']
)
test_loader = DataLoader(
    test_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn,
    num_workers=config['training']['num_workers']
)

input_size = config['training']['input_dim']
model = LSTM(
    input_size,
    config['training']['hidden_dim'],
    config['training']['output_dim'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, val_loader, criterion,  device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc='Validation', leave=False)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(val_loader)
    return avg_loss

print('Start training!')
for epoch in range(config['training']['epochs']):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

    val_loss = evaluate(model, val_loader, criterion, device)