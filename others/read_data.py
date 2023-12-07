import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import os 
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self,root_dir,names_file,transform=None):
        self.root_dir = root_dir #根目录
        self.names_file = names_file #.txt文件路径
        self.transform = transform #数据预处理
        self.size = 0 #数据集大小
        self.names_list = [] #数据集路径列表
        
        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file: #循环读取.txt文件总每行数据信息
            self.names_list.append(f)
            self.size += 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        image_path = self.root_dir + self.names_list[index].split(' ')[0] #获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path) #读取图片
        label = int(self.names_list[index].split(' ')[1]) #读取标签
 
        return image,label
        
        #sample = {'image':image,'label':lable}
        #if self.transform:
        #    sample = self.transform(sample) 
            
        #return sample #返回图片及对应的标签

root_dir='mri/train'
names_file='mri/train/train.txt'

train_dataset = MyDataset(root_dir=root_dir,names_file=names_file,transform=None)
valid_dataset = MyDataset(root_dir=root_dir,names_file=names_file,transform=None)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=4,shuffle=True,num_workers=4)
valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=4,shuffle=True,num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")