# 用 LSTM 模型来处理时序序列
from data_process import X_test, X_train, y_test, y_train, Labels
from train_and_test import train_and_test, device
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
batch_size = 2
num_layers = 3
num_directions = 2

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


class MLLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MLLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, hidden):
        output, _ = self.lstm(input, hidden)
        output = self.fc(output)
        output = self.dropout(output)
        return torch.squeeze(output[:, -1:, :], dim=1)


net = MLLSTM(input_size, hidden_size, num_layers, output_size)
print(net)
num_epochs = 300
lr, momentum, weight_decay = 0.0008, 0.9, 0.001
optimizer = torch.optim.RMSprop(
    net.parameters(), lr=lr, alpha=0.9, weight_decay=weight_decay)
loss = nn.CrossEntropyLoss()
init_hidden = torch.zeros(num_layers*num_directions,
                          batch_size, hidden_size).to(device)
losses, train_accs, test_accs = train_and_test(
    net, num_epochs, train_dataset, test_dataset, batch_size, optimizer, loss, init_hidden)
