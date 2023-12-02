# Utils.py 文件中封装了一些工具函数，包括：
#
from datasets import load_metric
from typing import List, Tuple
from configs.config import *
import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets

from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            'train': Dataset({
                features: ['id', 'feature_1', ..., 'feature_n', 'label'],
                numrows: a value
            })
            'validation': Dataset({
                features: ['id', 'feature_1', ..., 'feature_n', 'label'],
                numrows: a value
            })
            'test': Dataset({
                features: ['id', 'feature_1', ..., 'feature_n', 'label'],
                numrows: a value
            })
        }
        再使用 hugging face 中的库对 dict 进行一次类型转换.
    """

    train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(
        data, label, randomstate)
    sample_catorgory = ['train', 'validation', 'test']
    Samples = [train_data, valid_data, test_data]
    Labels = [train_label, valid_label, test_label]
    data_dict = {}
    for idx, (data, label) in enumerate(zip(Samples, Labels)):
        shape = data.shape
        dataset_dict = {}
        dataset_dict['id'] = [i for i in range(shape[0])]  # 样本的个数
        dataset_dict['label'] = label
        if (len(shape) == 2):
            for j in range(shape[1]):
                feature_list = []
                for i in range(shape[0]):
                    encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j])
                    feature_list.append(encode_str)
                dataset_dict['feature_'+str(j)] = feature_list
            data_dict[sample_catorgory[idx]
                      ] = datasets.Dataset.from_dict(dataset_dict)
        elif (len(shape) == 3):
            for j in range(shape[1]):
                feature_list = []
                for i in range(shape[0]):
                    seq_str = ''
                    for k in range(shape[2]):  # 样本 i 的时序数据形成的字符串
                        encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j, k])
                        seq_str += (encode_str+' ')
                    feature_list.append(seq_str)
                dataset_dict['feature_'+str(j)] = feature_list
            data_dict[sample_catorgory[idx]
                      ] = datasets.Dataset.from_dict(dataset_dict)
        else:
            raise NotImplementedError(
                "len(shape)!=2 or len(shape)!=3 Not Implement!")
    return datasets.DatasetDict(data_dict)


def get_tokenizer(model_name_or_path: str):
    # 根据模型的名字判断填充的方向， 这里的padding是指在序列数据的两端添加特定的元素，使序列达到指定的长度
    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side=padding_side, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id") is None:  # 检查 tokenizer 是否有 pad_token_id 属性
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 若没有则赋值为 eos_token_id 属性的值

    return tokenizer


def tokenize_function(sample: dict):
    # max_length=None => use the model max length (it's actually the default)
    # 获取字典的键, 存到一个list中
    keys = list(sample.keys())[2:]
    # print(keys)
    array = []
    final_output = {}
    final_output['label'] = sample['label']
    input_ids_list = []
    attention_mask_list = []  # 长度为样本数
    tokenizer = get_tokenizer(model_name_or_path)
    for i in range(len(sample['label'])):
        feature_list = []  # 样本i的所有特征对应的值, 长度为特征数
        for key in keys:
            # 第 i 个样本的特征 key 对应的值, sample[key] 表示特征 key 对应的每一个样本的值, 长度为样本的数量
            feature_list.append(sample[key][i])
        output = tokenizer(feature_list, truncation=True, max_length=None)
        input_ids_list.append(output['input_ids'])
        attention_mask_list.append(output['attention_mask'])
    label_tensor = torch.Tensor(final_output['label']).view(
        len(sample['label']), -1)
    # final_output['label'] = label_tensor.long()
    final_output['label'] = label_tensor[:, :490].long()
    input_ids_tensor = torch.Tensor(
        input_ids_list).view(len(sample['label']), -1)
    # final_output['input_ids'] = input_ids_tensor.long()
    final_output['input_ids'] = input_ids_tensor[:, :490].long()
    attention_mask_tensor = torch.Tensor(
        attention_mask_list).view(len(sample['label']), -1)
    # final_output['attention_mask'] = attention_mask_tensor.long()
    final_output['attention_mask'] = attention_mask_tensor[:, :490].long()
    # print(final_output['input_ids'][0][0],type(final_output['input_ids'][0][0]))
    # print(final_output['attention_mask'])
    print(final_output['input_ids'].shape)
    # print(np.shape(np.array(final_output['attention_mask'])))
    # print(np.shape(np.array(final_output['input_ids'])))
    # input_text : List[List[str]]  = array
    # print(input_text)
    # print(type(input_text))
    # input_text = " ".join(str(value) for value in sample.values())
    # print(type(input_text))
    # print(input_text[0:10])
    # print(len(input_text))
    # tokenizer=get_tokenizer(model_name_or_path)
    # # outputs = tokenizer(sample['feature_0'],sample['feature_1'],sample['feature_2'],sample['feature_3'],sample['feature_4'],sample['feature_5'],sample['feature_6'],sample['feature_7'],sample['feature_8'],sample['feature_9'],sample['feature_10'],sample['feature_11'],sample['feature_12'],sample['feature_13'],sample['feature_14'],sample['feature_15'],sample['feature_16'],sample['feature_17'],sample['feature_18'],sample['feature_19'],sample['feature_20'],sample['feature_21'],sample['feature_22'],sample['feature_23'],sample['feature_24'],sample['feature_25'],sample['feature_26'],sample['feature_27'],sample['feature_28'],sample['feature_29'],sample['feature_30'],sample['feature_31'],sample['feature_32'],sample['feature_33'],sample['feature_34'],sample['feature_35'],sample['feature_36'],sample['feature_37'],sample['feature_38'],sample['feature_39'],sample['feature_40'],sample['feature_41'],sample['feature_42'],sample['feature_43'],sample['feature_44'],sample['feature_45'],sample['feature_46'],sample['feature_47'],sample['feature_48'],sample['feature_49'],sample['feature_50'],sample['feature_51'],sample['feature_52'],sample['feature_53'],sample['feature_54'],sample['feature_55'],sample['feature_56'],sample['feature_57'],sample['feature_58'],sample['feature_59'],sample['feature_60'],sample['feature_61'],sample['feature_62'],sample['feature_63'],sample['feature_64'],sample['feature_65'],sample['feature_66'],sample['feature_67'],sample['feature_68'],sample['feature_69'],sample['feature_70'],sample['feature_71'],sample['feature_72'],sample['feature_73'],sample['feature_74'],sample['feature_75'],sample['feature_76'],sample['feature_77'],sample['feature_78'],sample['feature_79'],sample['feature_80'],sample['feature_81'],sample['feature_82'],sample['feature_83'],sample['feature_84'],sample['feature_85'],sample['feature_86'],sample['feature_87'],sample['feature_88'],sample['feature_89'],sample['feature_90'],sample['feature_91'],sample['feature_92'],sample['feature_93'],sample['feature_94'],sample['feature_95'],sample['feature_96'],sample['feature_97'],sample['feature_98'],sample['feature_99'],sample['feature_100'],sample['feature_101'],sample['feature_102'],sample['feature_103'],sample['feature_104'],sample['feature_105'],sample['feature_106'],sample['feature_107'],sample['feature_108'],sample['feature_109'],sample['feature_110'],sample['feature_111'],sample['feature_112'],sample['feature_113'],sample['feature_114'],sample['feature_115'],sample['feature_116'],sample['feature_117'],sample['feature_118'],sample['feature_119'],sample['feature_120'],sample['feature_121'],sample['feature_122'],sample['feature_123'],sample['feature_124'],sample['feature_125'],sample['feature_126'],sample['feature_127'],sample['feature_128'],sample['feature_129'],sample['feature_130'],sample['feature_131'],sample['feature_132'],sample['feature_133'],sample['feature_134'],sample['feature_135'],sample['feature_136'],sample['feature_137'],sample['feature_138'],sample['feature_139'],sample['feature_140'],sample['feature_141'],sample['feature_142'],sample['feature_143'],sample['feature_144'],sample['feature_145'],sample['feature_146'],sample['feature_147'],sample['feature_148'],sample['feature_149'],sample['feature_150'],sample['feature_151'],sample['feature_152'],sample['feature_153'],sample['feature_154'],sample['feature_155'],sample['feature_156'],sample['feature_157'],sample['feature_158'],sample['feature_159'],sample['feature_160'],sample['feature_161'],sample['feature_162'],sample['feature_163'],sample['feature_164'],sample['feature_165'],sample['feature_166'],sample['feature_167'],sample['feature_168'],sample['feature_169'],sample['feature_170'],sample['feature_171'],sample['feature_172'],sample['feature_173'],sample['feature_174'],sample['feature_175'],sample['feature_176'],sample['feature_177'],sample['feature_178'],sample['feature_179'],sample['feature_180'],sample['feature_181'],sample['feature_182'],sample['feature_183'],sample['feature_184'],sample['feature_185'], truncation=True, max_length=None)
    # outputs = tokenizer(sample['feature_0'],sample['feature_1'],sample['feature_2'], truncation=True, max_length=None)
    return final_output


def featuers_without_label(dataset: datasets.DatasetDict):
    # print(dataset['train'].features.keys())
    keys = list(dataset['train'].features.keys())
    keys.pop(1)
    return keys


accuracies = []


def compute_metrics(pred):
    global accuracies
    labels = pred.label_ids.tolist()  # 将ndarray转换为Python列表
    # labels = [3, 3, 3, 0, 0, 0, 4, 0, 2, 2, 3, 4, 3, 0, 3, 4, 4, 0, 3,
    #           2, 4, 2, 2, 3, 2, 2, 4, 4, 2, 2, 4, 0, 0, 0, 4, 3, 2, 4, 2, 0]
    preds = pred.predictions.argmax(-1).tolist()  # 将ndarray转换为Python列表
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    accuracies.append(acc)
    print(preds)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),  # 将ndarray转换为Python列表
        'precision': precision.tolist(),  # 将ndarray转换为Python列表
        'recall': recall.tolist()  # 将ndarray转换为Python列表
    }

# def compute_metrics(eval_preds):
#     metric = evaluate.load("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# 分类别计数
def count_labels(data):
    # 将二维列表转换为一维列表
    flatten_data = [item for sublist in data for item in sublist]
    print(flatten_data)

    # 使用Counter进行计数
    counter = Counter(flatten_data)

    # 输出每个元素的数量
    for element, count in counter.items():
        print(f"{element}: {count}")


# 分组重新赋予标签
def rename_labels(split, datasets):
    for i in ['train', 'test', 'validation']:
        new_label = 0
        for j in split:
            for k in j:
                datasets[i] = datasets[i].map(
                    lambda example: {'labels': [split[new_label][0]] if example['labels'] == [k] else example['labels']})
            new_label = new_label + 1
