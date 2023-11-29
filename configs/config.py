import torch
from dataclasses import dataclass
from typing import List, Tuple
from Database.database import database



LEVEL_TOKEN_FORMAT = '{:0>2}'


num_epochs = 20
lr = 1e-3
batch_size = 32
model_name_or_path = "roberta-large"

@dataclass
class promptConfig():
    # name: str
    lr: float = 1e-3
    # weight_decay: float = 5e-4
    num_epochs: int = 20
    # seed: int = 24
    # dropout: float = 0.05
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size: int = 32