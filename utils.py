# Utils.py 文件中封装了一些工具函数，包括：
#
import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from Database.utils import split_train_valid_test
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import LEVEL_TOKEN_FORMAT


# TODO: random_state 不太应该放在这里当成参数
def data_to_dict(data: np.ndarray, label: np.ndarray, randomstate: int) -> dict:
    """
    将数据转换为 huggingface datasets 能够识别的字典格式。
    参数：
        data (np.ndarray)：数据。
        label (np.ndarray)：标签。
    返回：
        dict：形如下面形式的字典:
        {
            'train': [dict1, dict2, ...]
            'validation': [dict1, dict2, ...]
            'test': [dict1, dict2, ...]
        }
        其中 dict1, dict2, ... 表示每个样本, 是形如下面形式的字典:
        {
            'id': 样本的标号
            'feature_1': 第一个特征对应的值, 如果是时序数据, 则是一个字符串, 每两个时间的数据用一个空格隔开
            ...
            'feature_n': 第 n 个特征对应的值
            'label': 标签的值
        }
        再使用 hugging face 中的库对 dict 进行一次类型转换.
    """

    train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(data, label, randomstate)
    sample_catorgory = ['train', 'validation', 'test']
    Samples = [train_data, valid_data, test_data]
    Labels = [train_label, valid_label, test_label]
    data_dict = {}
    for idx, (data, label) in enumerate(zip(Samples, Labels)):
        shape = data.shape
        if (len(shape) == 2):
            sample_dict_list = []
            for i in range(shape[0]):
                sample_dict = {}
                sample_dict['id'] = str(i)
                sample_dict['label'] = str(label[i])
                for j in range(shape[1]):
                    encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j])
                    sample_dict['feature_'+str(j)] = encode_str
                sample_dict_list.append(sample_dict)
            data_dict[sample_catorgory[idx]] = sample_dict_list
        elif (len(shape) == 3):
            sample_dict_list = []
            for i in range(shape[0]):
                sample_dict = {}
                sample_dict['id'] = str(i)
                sample_dict['label'] = str(label[i])
                for j in range(shape[1]):
                    seq_str=''
                    for k in range(shape[2]):
                        encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j, k])
                        time_series_str+=(encode_str+' ')
                    sample_dict['feature_'+str(j)] = seq_str
                sample_dict_list.append(sample_dict)
            data_dict[sample_catorgory[idx]] = sample_dict_list
        else:
            raise NotImplementedError(
                "len(shape)!=2 or len(shape)!=3 Not Implement!")
    return datasets.Dataset.from_dict(data_dict)
