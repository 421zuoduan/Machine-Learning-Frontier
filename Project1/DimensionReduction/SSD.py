import numpy as np
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
from sklearn.neighbors import NearestNeighbors
import scipy.linalg


def compute_projection_matrix(X_train,X_valid,X_test,y_train,y_valid,y_test,k, alpha, beta):
    '''
    Input:
        X: Matrix formed by all the data.
        X_l: Matrix formed by labelled data.
        y: All the label of labelled data.
        l: The number of labelled data.
        m: The number of all the data.
        c: The number of classes.
        k: the number of knn.
        alpha: the parameter of Laplacian.
    Output:
        A: The projection matrix that conduct dimension reduction for all the data.
    '''
    # 将 X_train, X_valid 合并, 并按照标签重排
    X_l=np.concatenate((X_train,X_valid),axis=0)
    y_l=np.concatenate((y_train,y_valid),axis=0)
    X=np.concatenate((X_l,X_test),axis=0)
    idx=np.argsort(y_l)
    X_l=X_l[idx]
    y_l=y_l[idx]
    m,d=X.shape # the number of data and dimension
    l,_=X_l.shape
    # Get the number of every class.
    c=len(Labels)
    class_num=[0 for i in range(c)]
    for label in y_l:
        class_num[label]+=1
    # Step 1: 计算KNN矩阵S
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    S = knn.kneighbors_graph(X).toarray()
    D = np.diag(np.sum(S, axis=1)) # 度数矩阵
    L = D - S
    # Step 2: 计算W矩阵
    blocks=[np.ones((class_num[i], class_num[i]))*(1/class_num[i]) for i in range(c)]
    W_l = scipy.linalg.block_diag(*blocks)
    print(m,l,m-l,W_l.shape,type(m),type(l),type(m-l))
    W = np.block([[W_l, np.zeros((l, m - l))], [np.zeros((m - l, l)), np.zeros((m - l, m - l))]])
    l=int(l)
    I_tilde = np.block([[np.identity(l),np.zeros((l, m - l))],[np.zeros((m - l, l)), np.zeros((m - l, m - l))]])
    I=np.identity(d)
    # Step 3: 计算广义特征值问题

    # 计算广义特征值问题的特征值和特征向量
    # print(X.shape,W.shape,X.T.shape,I.shape,L.shape,I_tilde.shape)
    S_b=np.dot(np.dot(X.T,W),X)
    print(I.shape,(beta*I).shape)
    S_t=np.dot(np.dot(X.T,I_tilde + alpha * L),X)+beta*I

    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.pinv(S_t),S_b))

    # 获取前c个非零特征值对应的特征向量
    eigvals_sorted = eigvals.argsort()
    eigvecs_sorted = eigvecs[:, eigvals_sorted]
    A = eigvecs_sorted[:, :c]
    print(A)
    return A,X

proj_mat,X=compute_projection_matrix(X_train,X_valid,X_test,y_train,y_valid,y_test,k=5,alpha=0.01,beta=0.01)
print(proj_mat.shape)

X_reduced=X.dot(proj_mat)
y=np.concatenate((y_train,y_valid,y_test),axis=0)
visualization(X_reduced,y,dim=2,method='LDA',class_num=2)
