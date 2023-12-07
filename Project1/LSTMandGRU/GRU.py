from data_process import X_test, X_train, y_test, y_train, Labels
from train_and_test import train_and_test, device
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # 添加上层目录到 sys.path


# global variables
num_workers = 0
hidden_size = 16  # 隐藏层维度
input_size = np.shape(X_train)[2]
output_size = len(Labels)
seq_len = np.shape(X_train)[1]
batch_size = 8
num_layers = 3
num_directions = 1


# data preparation
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


class MLGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, hidden):
        batch_size = input.shape[0]  # 动态获取 batch_size 的大小
        hidden = self.initHidden(batch_size)
        output, _ = self.gru(input, hidden)
        output = self.fc(output)
        output = self.dropout(output)
        return torch.squeeze(output[:, -1:, :], dim=1)  # 只返回最后一个时间步的输出

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers*num_directions, batch_size, self.hidden_size)


net = MLGRU(input_size, hidden_size, num_layers*num_directions, output_size)
print(net)
num_epochs = 100
lr, momentum, weight_decay = 0.003, 0.9, 0.001
optimizer = torch.optim.AdamW(
    net.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.CrossEntropyLoss()
init_hidden = torch.zeros(num_layers*num_directions,
                          batch_size, hidden_size).to(device)
losses, train_accs, test_accs = train_and_test(
    net, num_epochs, train_dataset, test_dataset, batch_size, optimizer, loss, init_hidden)
