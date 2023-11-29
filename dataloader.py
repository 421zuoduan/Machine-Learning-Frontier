import torch.nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import  Dataset
from utils import encode

class dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample

def split_dataset(data: np.ndarray, label: np.ndarray, randomstate: int):
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def load_dataset(args, data, label, tokenizer, random_state):

    traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_dataset(data, label, randomstate=random_state)
    prompt_num = 1 if len(traindata.shape) == 2 else traindata.shape[-2]

    train_dataset = dataset(traindata, trainlabel)
    valid_dataset = dataset(validdata, validlabel)

    train_dataset.data = encode(args, train_dataset.data, tokenizer)
    valid_dataset.data = encode(args, valid_dataset.data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1)

    return traindata, validdata, train_loader, valid_loader