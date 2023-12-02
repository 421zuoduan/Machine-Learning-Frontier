from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
import numpy as np
import torch
from utils import *
from configs.config import *
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
import evaluate
import peft
import transformers
import datasets
from typing import Tuple
config = datasets.DownloadConfig(resume_download=True, max_retries=100)

# file_path = 'dataset/FTD_90_200_fMRI.mat'
file_path = 'dataset/ADNI.mat'
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 32
slice_num = 100
slices = []
random_state = 0


def normalize(rawdata) -> dict:
    """
    对原始数据进行归一化处理。
    返回：
        dict：包含归一化后数据的字典。
    """
    newdatadict = {}
    for key, value in rawdata.items():
        valuemean = np.mean(value, axis=0)
        valuestd = np.std(value, axis=0)
        if not np.any(valuestd == 0):
            newvalue = (value - valuemean) / valuestd
        else:
            newvalue = value
        newdatadict[key] = newvalue
    return newdatadict


def rawdatatonumpy(rawdata) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return dict_tonp(rawdata)


def discrete(rawdata, slicenum=100, eps=1e-18) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    对数据进行离散化处理。

    参数：
        slicenum (int, 可选)：离散化的分段数。默认为100。
        eps (float, 可选)：避免除零的小值。默认为1e-18。

    返回：
        np.ndarray：离散化后的数据。
        np.ndarray：标签。
        np.ndarray：标签映射。
    """
    global slices
    data, label, labelmap = rawdatatonumpy(rawdata)
    if len(data.shape) == 2:
        datamin = np.min(data, axis=0)
        datamax = np.max(data, axis=0)
        datamax = datamax + eps
        assert datamin.shape[0] == data.shape[1]
        assert datamax.shape[0] == data.shape[1]

        num = data.shape[1]
        if not slices:
            generateslice(datamax, datamin, num, slicenum)
        for i in range(data.shape[1]):
            data[:, i] = np.digitize(data[:, i], slices[i, :])
        data = data.astype(int)
        return data, label, labelmap
    elif len(data.shape) == 3:
        data = data.transpose((0, 2, 1))
        datamin = np.array([np.min(data[:, :, i])
                           for i in range(data.shape[2])])
        datamax = np.array([np.max(data[:, :, i])
                           for i in range(data.shape[2])])
        datamax = datamax + eps
        assert datamin.shape[0] == data.shape[2]
        assert datamax.shape[0] == data.shape[2]

        num = data.shape[2]
        if not slices:
            generateslice(datamax, datamin, num, slicenum)
        for i in range(data.shape[2]):
            data[:, i, :] = np.digitize(data[:, i, :], slices[i, :])
        data: np.ndarray = data.astype(int)
        return data, label, labelmap
    else:
        raise NotImplementedError(
            'len(data.shape)!=2&&len(data.shape)!=3 Not Implement!')


def generateslice(datamax: np.ndarray, datamin: np.ndarray, num: int, slicenum: int):
    """
    生成离散化的切片。

    参数：
        datamax (np.ndarray)：每个特征的最大值。
        datamin (np.ndarray)：每个特征的最小值。
        num (int)：特征数量。
        slicenum (int)：离散化的分段数。
    """
    global slices
    for min, max in zip(datamin, datamax):
        label_slice = np.linspace(min, max, slicenum)
        slices.append(label_slice)
    slices = np.array(slices)
    assert slices.shape == (num, slicenum)


rawdata = load_mat(file_path)
rawdata = normalize(rawdata)
data, label, labelmap = discrete(rawdata, slice_num)
traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_train_valid_test(
    data, label, randomstate=random_state)

dataset = data_to_dict(data, label, random_state)
# print(featuers_without_label(dataset))
keys = featuers_without_label(dataset)
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=keys,
)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
