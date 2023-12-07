import sys,os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加上层目录到 sys.path
# current_directory = os.path.abspath(os.path.curdir)
# parent_directory = os.path.dirname(current_directory)
# sys.path.append(parent_directory)
from data_process import X_test,X_train,X_valid,y_test_master, Labels_MCI,dict_labels
from data_process import y_test,y_train,y_train_master,y_valid_master,y_valid
from data_process import X_valid_MCI,X_train_MCI,y_valid_MCI,y_train_MCI,Labels
# from dimension_reduction import X_train,X_valid,X_test
from dimension_reduction_lda import dimension_reduction,visualization
from train_and_test import train_and_test,device
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as pyg_nn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import scipy.sparse as sp

train_num=X_train.shape[0]
val_num=X_valid.shape[0]
test_num=X_test.shape[0]

import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
def plot_graph(data):
    """
    Description: Plot the graph of the data.

    Input:
    - data: The data of the graph.
    """
    # 创建一个 PyG 的 Data 对象
    # 请根据您的数据自行创建 Data 对象
    # all_nodes = set(range(data.num_nodes))
    # connected_nodes = set(data.edge_index[0].tolist()) | set(data.edge_index[1].tolist())
    # isolated_nodes = all_nodes - connected_nodes
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().tolist())
    colors=['r','g','b','c','m']
    color_mapping = {dict_labels[Labels[i]]: colors[i] for i in range(len(Labels))}
    print(color_mapping)
    node_idx_train = [data.y[i] for i in range(train_num)]
    # print(node_idx_train,[i.item() for i in node_idx_train])
    node_colors_train=[]
    for i in node_idx_train:
        key=i.item()
        c=color_mapping[key]
        node_colors_train.append(c)
    # node_colors_train = [color_mapping[i.item()] for i in node_idx_train]
    node_idx_valid = [data.y[i] for i in range(train_num,train_num+val_num)]
    node_colors_valid=[]
    for i in node_idx_valid:
        key=i.item()
        c=color_mapping[key]
        node_colors_valid.append(c)
    # node_colors_valid = [color_mapping[i.item()] for i in node_idx_valid]
    node_colors_test = ['lightgray'] * test_num

    node_colors = node_colors_train + node_colors_valid + node_colors_test
    # 使用 NetworkX 和 Matplotlib 进行可视化
    # pos = nx.spring_layout(G)  # 选择布局算法
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(8, 8))
    print(len(node_colors),len(data.y),data.num_nodes)
    print(G)
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=30)
    plt.title("Graph Visualization")
    plt.show()


# def random_graph():


def construct_epsilon_neighborhood_graph(X, y, epsilon=0.5):
    """
    Description: Construct an ε-neighborhood graph of the data.

    Input:
    - X: Train data and test data.
    - y: The label of train data.
    - epsilon: The radius for ε-neighborhood.

    Return:
        - The ε-neighborhood graph of the data.
    """
    num_samples = len(X)
    edge_index = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            # Calculate the Euclidean distance between data points i and j
            distance = np.linalg.norm(X[i] - X[j])
            if distance <= epsilon:
                # Check if they belong to the same class for training data
                if i < train_num + val_num and j < train_num + val_num and y[i] == y[j]:
                    edge_index.append([i, j])
                # For non-training data or different classes, connect them directly
                elif i >= train_num + val_num or j >= train_num + val_num:
                    edge_index.append([i, j])

    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)

    X = X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)

    data = pyg.data.Data(x=X, edge_index=edge_index, y=y)
    return data


def construct_knn_graph(X,y, k=200, metric='unsupervised'):
    '''
    Description: Construct the knn graph of the data.
    
    Input:
    - X: Train data and test data.
    - y: The label of train data.
    - k: The number of nearest neighbors.
    
    Return:
        - the knn graph of the data.
    '''
    knn=NearestNeighbors(n_neighbors=k,metric=metric)
    knn.fit(X)

    _, indices = knn.kneighbors(X) # Indices of the nearest points in the population matrix.
    knn_c=KNeighborsClassifier(n_neighbors=k,metric=metric)
    knn_c.fit(np.concatenate((X_train,X_valid),axis=0),np.concatenate((y_train,y_valid),axis=0))
    y_pred=knn_c.predict(X_test)
    score=np.sum(y_pred==y_test)/len(y_test)
    print(f'knn score: {score}')
    adj_mat=np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(k):
            adj_mat[i][indices[i][j]]=1
            adj_mat[indices[i][j]][i]=1
    edge_index = []
    for i in range(len(X)):
        for j in range(len(X)):
            if j>i and adj_mat[i][j]==1: # 对于训练集中的数据，只有两个是同一类才连边
                # edge_index.append([i,j])
                if i>=train_num or j>=train_num: # 如果两个点有一个不是训练集，就直接根据邻接矩阵连边
                    edge_index.append([i,j])
                if i<train_num and j<train_num and y[i]==y[j]:
                    edge_index.append([i,j])
                
    edge_index = torch.tensor(edge_index).T
    edge_index = edge_index.to(device)
    X=X.astype(np.float32)
    X = torch.tensor(X).float().to(device)
    y = torch.tensor(y).long().to(device)
    # print(X.shape,y.shape)
    data = pyg.data.Data(x=X,edge_index=edge_index,y=y)
    return data

X=np.concatenate((X_train,X_valid,X_test),axis=0)
# y=np.concatenate((y_train_master,y_valid_master,y_test_master),axis=0) # for ANDI
y=np.concatenate((y_train,y_valid,y_test),axis=0)
# for ADNI, not PPMI
# y[y==1]=0
# y[y==4]=1
# y[y==-1]=2
# print(X,y)

k=100
data=construct_knn_graph(X,y,k=k,metric='euclidean')
# data=construct_epsilon_neighborhood_graph(X,y,0.3)
if k<=7:
    plot_graph(data)

train_idx = np.array(range(X_train.shape[0]))
val_idx = np.array(range(X_train.shape[0],X_train.shape[0]+X_valid.shape[0]))
test_idx = np.array(range(X_train.shape[0]+X_valid.shape[0],X_train.shape[0]+X_valid.shape[0]+X_test.shape[0]))
print(X.shape,y.shape)
all_f=np.zeros((X.shape[0],),dtype=np.bool)

all_f_tmp=all_f.copy()
all_f_tmp[train_idx]=True
train_mask=all_f_tmp

all_f_tmp=all_f.copy()
all_f_tmp[val_idx]=True
val_mask=all_f_tmp

all_f_tmp=all_f.copy()
all_f_tmp[test_idx]=True
test_mask=all_f_tmp

print(y.shape,train_idx.shape,val_idx.shape,test_idx.shape)
print(train_mask.shape,val_mask.shape,test_mask.shape)
print(y[train_mask].shape,y[val_mask].shape,y[test_mask].shape)

data.train_mask=train_mask
data.val_mask=val_mask
data.test_mask=test_mask


print(data)

import argparse
import torch_geometric.transforms as T
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

num_features=X_train.shape[1]
num_classes=5

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # GCNConv模型输入参数：输入结点特征维度，输出结点特征维度，是否cached和是否normalize；
        self.conv1 = pyg_nn.GCNConv(num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        # self.fc1=nn.Linear(16,8)
        # self.fc2=nn.Linear(8,4)
        self.conv2 = pyg_nn.GCNConv(16, num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv3 = pyg_nn.GCNConv(32, num_classes, cached=True,
        #                      normalize=not args.use_gdc)
        self.fc3 = nn.Linear(num_classes, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()
        self.fc5 = nn.Linear(num_classes, num_classes)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # dropout操作，避免过拟合
        
        # x = self.fc1(x)
        # x = F.relu(x)
        # x=F.dropout(x,training=self.training)

        # x = self.fc2(x)
        # x = F.relu(x)
        # x=F.dropout(x,training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)

        # x = self.conv3(x, edge_index, edge_weight)
        
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        # x=F.dropout(x,training=self.training)

        # x = self.fc5(x)

        return F.log_softmax(x, dim=1)  # 模型最后一层接上一个softmax和CNN类似
model,data=GCN().to(device),data.to(device)


# optimizer = torch.optim.Adadelta([
#     dict(params=model.reg_params, weight_decay=5e-3),
#     dict(params=model.non_reg_params, weight_decay=0)
# ], lr=5e-2)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01, weight_decay=5e-4)

optimizer = torch.optim.AdamW([
    dict(params=model.reg_params, weight_decay=0.01),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=1e-2)

def train():
    model.train()
    optimizer.zero_grad()
    output=model()
    loss=F.nll_loss(output[data.train_mask],data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()

def test():
    model.eval()
    log, accs= model(), []
    for _,mask in data('train_mask','val_mask','test_mask'):
        pred=log[mask].max(1)[1]
        # print(mask,data.y[mask].shape)
        acc=pred.eq(data.y[mask]).sum().item()/mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 20001):
    loss=train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    if epoch % 100 == 0:
        print(log.format(epoch,loss, train_acc, best_val_acc, test_acc))