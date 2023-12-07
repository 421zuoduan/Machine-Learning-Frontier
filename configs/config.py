import torch
from dataclasses import dataclass
from typing import List, Tuple
from Database.database import database

LEVEL_TOKEN_FORMAT = '{:0>2}'

ADNI = database('ADNI', 'ADNI')
PPMI = database('PPMI', 'PPMI')
ADNI_fMRI = database('ADNI_fMRI', 'ADNI_fMRI')
OCD_fMRI = database('OCD_fMRI', 'OCD_fMRI')
FTD_fMRI = database('FTD_fMRI', 'FTD_fMRI')


@dataclass
class promptConfig():
    name: str
    num_epochs = 500  # 训练轮数
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size = 8
    model_name = "roberta-large"  # 大模型名称
    model_name_or_path = "./robert"  # 大模型路径
    tuning_method = "peft-p-tuning"  # 微调方法
    dropout: float = 0.05
    device: torch.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size: int = 4
    slice_num: int = 100
    random_state: Tuple[int] = (0, 1, 2, 3, 4)


ADNI_config = promptConfig('ADNI')
PPMI_config = promptConfig('PPMI')
ADNI_fMRI_config = promptConfig('ADNI_fMRI')
OCD_fMRI_config = promptConfig('OCD_fMRI')
FTD_fMRI_config = promptConfig('FTD_fMRI')
