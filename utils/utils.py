import scipy.io as IO
import numpy as np
import datasets
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import *


def load_mat(filepath: str):
    data = IO.loadmat(filepath)
    data_dict = {}
    for key, value in data.items():
        if key not in ('__version__', '__globals__', '__header__'):
            value = np.ascontiguousarray(value)
            data_dict[key] = value.astype('float64')
    return data_dict


def split_train_valid_test(data: np.ndarray, label: np.ndarray, randomstate: int) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def np_to_dict(data, label, labelmap):
    rawdata = {}
    unique_labels = np.unique(label)
    for u_label in unique_labels:
        label_data = data[label == u_label]
        label_name = labelmap[u_label]
        if label_name not in rawdata:
            rawdata[label_name] = label_data.tolist()
        else:
            rawdata[label_name].extend(label_data.tolist())
    return rawdata


def dict_tonp(datadict):
    data = []
    label = []
    labelmap = []
    pointer = 0
    for key, value in datadict.items():
        labelmap.append(key)
        for item in value:
            label.append(pointer)
            data.append(item)
        pointer += 1
    return np.array(data), np.array(label), np.array(labelmap)


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
        model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:  # 检查 tokenizer 是否有 pad_token_id 属性
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 若没有则赋值为 eos_token_id 属性的值

    return tokenizer


def tokenize_function(sample: dict):
    # max_length=None => use the model max length (it's actually the default)
    # 获取字典的键, 存到一个list中
    keys = list(sample.keys())
    array = []
    for key in keys:
        array.append(sample[key])
    tokenizer = get_tokenizer(model_name_or_path)
    # print(tokenizer)
    print(array)
    print(type(array))
    outputs = tokenizer(array, truncation=True, max_length=None)
    return outputs


def featuers_without_label(dataset: datasets.DatasetDict):
    # print(dataset['train'].features.keys())
    keys = list(dataset['train'].features.keys())
    keys.pop(1)
    return keys
