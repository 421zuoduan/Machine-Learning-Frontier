import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import prompt_reflect, mask_slice_reflect, concate_prompt_data

def train_iter(correct_predictions, criterion, device, epoch_loss, mask_model, model, prompt_model,
                 tokenizer, total_samples, train_loader, optimizer, train: bool = False):
    for i, batch in enumerate(train_loader):
        data = batch['data']
        label = batch['label']

        data = data.to(device)
        label = label.to(device)

        shape = data.shape

        prompt_data, mask_slice = prompt_model()

        prompt_data = prompt_reflect(prompt_data, tokenizer, device)
        prompt_length = prompt_data.shape[-1]
        mask, slice = mask_slice_reflect(mask_slice, prompt_length, device)
        # print(prompt_data.shape)
        # print(data.shape)
        if (len(shape) == 2):
            prompt_data, mask_pos = concate_prompt_data(prompt_data, data, mask, slice, tokenizer, device)
        elif (len(shape) == 3):
            prompt_list = []
            mask_list = []
            for i in range(prompt_data.shape[0]):
                prt, mk = concate_prompt_data(prompt_data[i, :], data[:, i, :], mask[i], slice[i], tokenizer, device)
                prompt_list.append(prt)
                mask_list.append(mk)
            prompt_data = torch.stack(prompt_list, dim=1)
            mask_pos = torch.stack(mask_list, dim=0)

        attention_mask = torch.ones_like(prompt_data, device=device)
        torch.cuda.empty_cache()
        if (len(shape) == 2):
            with torch.no_grad():
                out = model(prompt_data, attention_mask)
            mask_data = out.last_hidden_state[:, mask_pos, :]

            output = mask_model(mask_data)
        elif (len(shape) == 3):
            mask_data_list = []

            for i in range(shape[0]):
                # print(prompt_data.shape)
                promptitem = prompt_data[i, :]
                attention_mask_item = attention_mask[i, :]
                with torch.no_grad():
                    out = model(promptitem, attention_mask_item)
                mask_item_list = []
                for j in range(shape[1]):
                    mask_item = out.last_hidden_state[j, mask_pos[j].item(), :]
                    mask_item_list.append(mask_item)
                mask_data = torch.stack(mask_item_list, dim=0)
                output = mask_model(mask_data)
                mask_data_list.append(output)
            output = torch.stack(mask_data_list, dim=0)
        # print(output.shape)
        output = output.view(output.shape[0], output.shape[-1])
        loss = criterion(output, label.long())

        # 计算准确率
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