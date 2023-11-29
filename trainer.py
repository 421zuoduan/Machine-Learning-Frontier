import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import prompt_reflect, mask_slice_reflect, concate_prompt_data

def trainer(train_loader, model, tokenizer, correct_predictions, epoch_loss, 
                 total_samples, optimizer, criterion, device, train: bool = True):
    for i, batch in enumerate(train_loader):
        data = batch['data']
        label = batch['label']
        data = data.to(device)
        label = label.to(device)

        # 模型
        output = model(data, tokenizer)

        # 计算准确率
        output = output.view(output.shape[0], output.shape[-1])
        loss = criterion(output, label.long())

        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)

        if (train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(label)
        # 累积epoch损失
        epoch_loss += loss.item()
    return correct_predictions, epoch_loss, total_samples