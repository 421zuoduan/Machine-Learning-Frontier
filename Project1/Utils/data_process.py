# 读入.mat文件, 划分测试集与训练集

import scipy.io
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
# if len(sys.argv)!=2:
#     print('Usage: python data_process.py <dataset_path>')
#     sys.exit(1)

# file_path=sys.argv[1]
file_path='BeforeReconstruct/dataset/ADNI.mat'
# file_path='dataset/PPMI.mat'
mat_data=scipy.io.loadmat(file_path)
print(file_path)
# print(mat_data.keys())
train_ratio=0.6
valid_ratio=0.2
test_ratio=0.2




Labels=[key for key in mat_data.keys() if not key.startswith('__')]
dict_labels = {label: i for i, label in enumerate(Labels)}
print(dict_labels)
scaler=StandardScaler()


import time
import random
# random_seed=int(time.time())

def data_process(random_seed):
    X_train=[]
    X_test=[]
    y_train=[]
    y_test=[]
    X_valid=[]
    y_valid=[]
    y_train_master=[] # 在主分类器中, 将 MCI, MCIn, MCIp 合并为一类的训练集标签
    y_valid_master=[]
    y_test_master=[]
    X_train_MCI=[] # MCI, MCIn, MCIp 的训练集, 验证集及其标签
    X_valid_MCI=[]
    y_train_MCI=[]
    y_valid_MCI=[]
    X_test_MCI=[]
    y_test_MCI=[]

    print(f'random seed: {random_seed}')
    np.random.seed(random_seed)

    for label,data in mat_data.items():
        if label.startswith('__'):
            continue
        label_num=dict_labels[label]
        N=np.shape(data)[0]
        indices=np.random.permutation(N)
        train_end = int(train_ratio * N)
        valid_end = int((train_ratio + valid_ratio) * N)

        train_index = indices[:train_end]
        validation_index = indices[train_end:valid_end]
        test_index = indices[valid_end:]
        # print(indices)
        # print(train_index,validation_index,test_index)
        for i in range(N):
            if i in train_index:
                X_train.append(data[i])
                y_train.append(label_num)
                if file_path=='dataset/ADNI.mat' and label_num>=1 and label_num<=3:
                    y_train_master.append(-1)
                    X_train_MCI.append(data[i]) # MCI, MCIn, MCIp 的训练集
                    y_train_MCI.append(label_num)
                else:
                    y_train_master.append(label_num)
            if i in validation_index:
                X_valid.append(data[i])
                y_valid.append(label_num)
                if file_path=='dataset/ADNI.mat' and label_num>=1 and label_num<=3:
                    y_valid_master.append(-1)
                    X_valid_MCI.append(data[i]) # MCI, MCIn, MCIp 的验证集
                    y_valid_MCI.append(label_num)
                else:
                    y_valid_master.append(label_num)
            if i in test_index:
                X_test.append(data[i])
                y_test.append(label_num)
                if file_path=='dataset/ADNI.mat' and label_num>=1 and label_num<=3:
                    y_test_master.append(-1)
                    X_test_MCI.append(data[i]) # MCI, MCIn, MCIp 的测试集
                    y_test_MCI.append(label_num)
                else:
                    y_test_master.append(label_num)
                

    dict_labels_MCI = {label: i for i, label in enumerate(Labels[1:])}
    X_train = np.array(X_train)
    y_train=np.array(y_train)
    y_train_master=np.array(y_train_master)

    X_valid = np.array(X_valid)
    y_valid=np.array(y_valid)
    y_valid_master=np.array(y_valid_master)

    X_train_MCI = np.array(X_train_MCI)
    y_train_MCI=np.array(y_train_MCI)

    X_valid_MCI = np.array(X_valid_MCI)
    y_valid_MCI=np.array(y_valid_MCI)

    X_test = np.array(X_test)
    y_test=np.array(y_test)
    y_test_master=np.array(y_test_master)
    def normalize(X):
        if X!=[]:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            # X=(X-np.mean(X,axis=0))/np.std(X,axis=0)
        return X
    Labels_MCI=[-1,0,4]
    # X_train=normalize(X_train)
    # X_valid=normalize(X_valid)
    # X_train_MCI=normalize(X_train_MCI)
    # X_valid_MCI=normalize(X_valid_MCI)
    # X_test=normalize(X_test)
    # X_test_MCI=normalize(X_test_MCI)
    return X_train,y_train,X_valid,y_valid,X_test,y_test