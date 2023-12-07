import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from configs.config import *
from utils import *


class Tokenized():
    def __init__(self, model_name_or_path: str, db: database, db_cfg: promptConfig, random_state: int) -> None:
        self.model_name_or_path = model_name_or_path
        self.data, self.label, _ = db.discrete(db_cfg.slice_num)
        self.tokenizer = self.get_tokenizer(model_name_or_path)
        self.random_state = random_state

    def tokenize_function(self, sample: dict):
        '''
        dataset.map() 的参数函数, 用于将数据集中的每一个样本转换为模型能够识别的格式.

        参数:
            sample (dict): 数据集中的所有样本.
        返回:
            final_output (dict): 转换后能直接输入到大模型中的样本.
        '''
        # max_length=None => use the model max length (it's actually the default)
        keys = list(sample.keys())[2:]
        final_output = {}
        final_output['label'] = sample['label']
        input_ids_list = []
        attention_mask_list = []  # 长度为样本数
        for i in range(len(sample['label'])):
            feature_list = []  # 样本i的所有特征对应的值, 长度为特征数
            for key in keys:
                # 第 i 个样本的特征 key 对应的值, sample[key] 表示特征 key 对应的每一个样本的值, 长度为样本的数量
                feature_list.append(sample[key][i])
            output = self.tokenizer(
                feature_list, truncation=True, max_length=None)
            input_ids_list.append(output['input_ids'])
            attention_mask_list.append(output['attention_mask'])
        label_tensor = torch.Tensor(final_output['label']).view(
            len(sample['label']), -1)
        final_output['label'] = label_tensor[:, :490].long()
        input_ids_tensor = torch.Tensor(
            input_ids_list).view(len(sample['label']), -1)
        final_output['input_ids'] = input_ids_tensor[:, :490].long()
        attention_mask_tensor = torch.Tensor(
            attention_mask_list).view(len(sample['label']), -1)
        final_output['attention_mask'] = attention_mask_tensor[:, :490].long()
        return final_output

    def get_tokenized_dataset(self):
        dataset = data_to_dict(self.data, self.label, self.random_state)
        keys = featuers_without_label(dataset)
        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=keys,
        )
        tokenized_datasets = tokenized_datasets.rename_column(
            "label", "labels")
        return tokenized_datasets

    def get_tokenizer(self, model_name_or_path: str):
        '''
        给定模型名字或路径, 获取 tokenizer.

        参数: 
            model_name_or_path (str): 模型名字或路径.
        返回:
            tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): tokenizer.
        '''
        if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
        else:
            padding_side = "right"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side=padding_side)
        if getattr(tokenizer, "pad_token_id") is None:  # 检查 tokenizer 是否有 pad_token_id 属性
            tokenizer.pad_token_id = tokenizer.eos_token_id  # 若没有则赋值为 eos_token_id 属性的值

        return tokenizer
