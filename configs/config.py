import torch
from dataclasses import dataclass
from typing import List, Tuple
from Database.database import database


LEVEL_TOKEN_FORMAT = '{:0>2}'


num_epochs = 50  # 训练轮数
lr = 1e-3
batch_size = 3
model_name = "chinese-macbert-large"  # 大模型名称
model_name_or_path = "./chinese-macbert-large"  # 大模型路径
tuning_method = "peft-p-tuning"  # 微调方法

ADNI = database('ADNI', 'ADNI')
PPMI = database('PPMI', 'PPMI')
ADNI_fMRI = database('ADNI_fMRI', 'ADNI_fMRI')
OCD_fMRI = database('OCD_fMRI', 'OCD_fMRI')
FTD_fMRI = database('FTD_fMRI', 'FTD_fMRI')


@dataclass
# class promptConfig():
#     # name: str
#     lr: float = 1e-3
#     # weight_decay: float = 5e-4
#     num_epochs: int = 20
#     # seed: int = 24
#     # dropout: float = 0.05
#     device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     batch_size: int = 32
class promptConfig():
    name: str
    lr: float = 1e-4
    weight_decay: float = 5e-4
    num_epochs: int = 50
    seed: int = 24
    dropout: float = 0.05
    device: torch.device = torch.device('cuda')
    batch_size: int = 4
    gru_gru_hidden_state: int = 64
    gru_gru_layer: int = 1
    prompt_seq_length: int = 64
    mask_str: str = "[MASK]"
    slice_num: int = 100
    mask_hidden_features: Tuple[int] = (512, 256)
    cut_hidden_features: Tuple[int] = (64, 128, 256)
    random_state: Tuple[int] = (0, 1, 2, 3, 4)
    add_terminal: bool = True
    patience: int = 10


ADNI_config = promptConfig('ADNI')
PPMI_config = promptConfig('PPMI')
ADNI_fMRI_config = promptConfig('ADNI_fMRI')
OCD_fMRI_config = promptConfig('OCD_fMRI')
FTD_fMRI_config = promptConfig('FTD_fMRI')
