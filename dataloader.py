import torch.nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import  Dataset
from utils import encode

def encode(data: torch.Tensor, tokenizer):
    shape = data.shape
    if (len(shape) == 2):
        encode_tensor_list = []
        for i in range(shape[0]):
            encode_item_list = []
            for j in range(shape[1]):
                encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j])
                encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                encode_item_list.append(encode_item)
            # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
            encode_tensor_list.append(encode_item_list)
        return torch.tensor(encode_tensor_list, dtype=torch.long)
    elif (len(shape) == 3):
        encode_tensor_tensor_list = []
        for i in range(shape[0]):
            encode_tensor_list = []
            for j in range(shape[1]):
                encode_item_list = []
                for k in range(shape[2]):
                    encode_str = LEVEL_TOKEN_FORMAT.format(data[i, j, k])
                    encode_item = tokenizer.convert_tokens_to_ids(encode_str)
                    encode_item_list.append(encode_item)
                # encode_tensor = torch.tensor(encode_item_list, dtype=torch.long)
                encode_tensor_list.append(encode_item_list)
            encode_tensor_tensor_list.append(encode_tensor_list)
        return torch.tensor(encode_tensor_tensor_list, dtype=torch.long)
    else:
        raise NotImplementedError("len(shape)!=2 or len(shape)!=3 Not Implement!")

class dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample

def split_dataset(data: np.ndarray, label: np.ndarray, randomstate: int) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    train_data, temp_data, train_label, temp_label = train_test_split(data, label, test_size=0.4,
                                                                      random_state=randomstate, stratify=label)

    valid_data, test_data, valid_label, test_label = train_test_split(temp_data, temp_label, test_size=0.5,
                                                                      random_state=randomstate, stratify=temp_label)

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def load_dataset(args, data, label, device, tokenizer, random_state, mask_hidden_size, labelmap):

    traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_dataset(data, label, randomstate=random_state)
    prompt_num = 1 if len(traindata.shape) == 2 else traindata.shape[-2]

    train_dataset = dataset(traindata, trainlabel)
    valid_dataset = dataset(validdata, validlabel)

    train_dataset.data = encode(train_dataset.data, tokenizer)
    valid_dataset.data = encode(valid_dataset.data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1)

    return traindata, validdata, train_loader, valid_loader