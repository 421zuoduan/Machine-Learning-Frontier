import torch
from torch import nn
from transformers import PreTrainedTokenizer
from utils import prompt_reflect, mask_slice_reflect, concat_prompt_data
from prompt import prompt
from finetune import mlp, gru


class prompt_generate(nn.Module):
    def __init__(self, args, traindata_dimention, traindata_len, prompt_seq_length, labelmap, mask_hidden_size, device=torch.device('cuda')):
        super().__init__()
        self.prompt_num = traindata_len if traindata_dimention == 3 else 1
        self.prompt_length = prompt_seq_length
        self.labelmap = labelmap
        self.prompt = prompt()
        self.device = device
        self.to(self.device)

    def forward(self):

        prompt_model = self.prompt
        if (self.prompt_num == 2):
            mask_model = mlp(self.mask_hidden_size, self.args.mask_hidden_features, len(self.labelmap), self.args.dropout)
        elif (self.prompt_num == 3):
            mask_model = nn.Sequential(
                gru(self.mask_hidden_size, self.args.gru_hidden_state, self.args.gru_gru_layer),
                mlp(self.args.gru_gru_hidden_state * self.prompt_num, self.args.mask_hidden_features, len(self.labelmap),
                    self.args.dropout))
        
        return prompt_model, mask_model


class model(nn.Module):
    def __init__(self, args, LLM_model, traindata_dimention, traindata_len, labelmap, device=torch.device('cuda')):
        super().__init__()
        self.args = args
        self.device = device
        self.traindata_dimention = traindata_dimention
        self.LLM_model = LLM_model
        self.LLM_model.require_grad = False
        self.mask_hidden_size = LLM_model.config.hidden_size
        self.prompt_model, self.mask_model = prompt_generate(args, traindata_dimention, traindata_len, args.prompt_seq_length, labelmap, self.mask_hidden_size)
        self.to(self.device)

    def forward(self, data, tokenizer):

        shape = data.shape

        before_prompt, after_prompt, mask_pos = self.prompt_model.returnPrompt(tokenizer)

        prompt_data, mask_slice = self.prompt_model()
        prompt_data = prompt_reflect(prompt_data, tokenizer, self.device)
        prompt_length = prompt_data.shape[-1]
        mask, slice = mask_slice_reflect(mask_slice, prompt_length, self.device)

        # 生成prompt并与data拼接
        if self.traindata_dimention == 2:
            prompt_data, mask_pos = concat_prompt_data(prompt_data, data, mask, slice, tokenizer, self.device)
        elif self.traindata_dimention == 3:
            prompt_list = []
            mask_list = []
            for i in range(prompt_data.shape[0]):
                prt, mk = concat_prompt_data(prompt_data[i, :], data[:, i, :], mask[i], slice[i], tokenizer, self.device)
                prompt_list.append(prt)
                mask_list.append(mk)
            prompt_data = torch.stack(prompt_list, dim=1)
            mask_pos = torch.stack(mask_list, dim=0)

        attention_mask = torch.ones_like(prompt_data, device=self.device)
        torch.cuda.empty_cache()

        # 过LLM和后面的head
        if self.traindata_dimention == 2:
            # with torch.no_grad():
            #     out = model(prompt_data, attention_mask)
            out = self.LLM_model(prompt_data, attention_mask)
            mask_data = out.last_hidden_state[:, mask_pos, :]

            output = self.mask_model(mask_data)
        elif self.traindata_dimention == 2:
            mask_data_list = []

            for i in range(shape[0]):
                promptitem = prompt_data[i, :]
                attention_mask_item = attention_mask[i, :]
                # with torch.no_grad():
                #     out = model(promptitem, attention_mask_item)
                out = self.LLM_model(prompt_data, attention_mask)
                mask_item_list = []
                for j in range(shape[1]):
                    mask_item = out.last_hidden_state[j, mask_pos[j].item(), :]
                    mask_item_list.append(mask_item)
                mask_data = torch.stack(mask_item_list, dim=0)
                output = self.mask_model(mask_data)
                mask_data_list.append(output)
            output = torch.stack(mask_data_list, dim=0)
        output = output.view(output.shape[0], output.shape[-1])

        return output