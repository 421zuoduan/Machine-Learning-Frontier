from typing import List, Tuple
from configs.config import *
import numpy as np
import torch
import datasets
import matplotlib.pyplot as plt
import random

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Database.utils import split_train_valid_test

def setup_seed(seed) -> None:
    '''
    用各种方式设置随机数种子.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def featuers_without_label(dataset: datasets.DatasetDict):
    '''
    返回一个列表, 列表中的元素为数据集中除了 label 以外的所有特征的名称.
    '''
    keys = list(dataset['train'].features.keys())
    keys.pop(1)
    return keys


accuracies = []


def compute_metrics(pred):
    '''
    计算模型的评估指标, 用于 Trainer 中的 compute_metrics 参数.
    '''
    global accuracies
    labels = pred.label_ids.tolist()  # 将ndarray转换为Python列表
    preds = pred.predictions.argmax(-1).tolist()  # 将ndarray转换为Python列表
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    accuracies.append(acc)
    return {
        'accuracy': acc,
        'f1': f1.tolist(),  # 将ndarray转换为Python列表
        'precision': precision.tolist(),  # 将ndarray转换为Python列表
        'recall': recall.tolist()  # 将ndarray转换为Python列表
    }


def plot_acc(db_cfg: database, model_name: str, num_epochs: int, accuracies: list):
    '''
    绘制准确率折线图.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), accuracies, "r.-")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(db_cfg.name + " with " + model_name + " Validation Accuracy")
    plt.show()

def calculate_multi_dtw(sample1, sample2):
    '''
    基于多变量DTW算法计算两个多变量时间序列样本之间的距离。
    '''
    global cnt
    num_varibles = 90
    num_moments = int(sample1.shape[0]/num_varibles)
    multi_dtw = 0
    # 计算DTW距离之和
    for i in range(num_varibles):
        distance, _ = fastdtw(sample1[num_moments*i:num_moments*(i+1)],
                              sample2[num_moments*i:num_moments*(i+1)], dist=2)
        multi_dtw = multi_dtw + distance
    multi_dtw = multi_dtw / len(sample1)
    cnt += 1
    print(cnt, multi_dtw)
    return multi_dtw