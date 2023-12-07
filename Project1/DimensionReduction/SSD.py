import numpy as np
import sys
import os
from dimension_reduction_lda import dimension_reduction, visualization
from sklearn.neighbors import NearestNeighbors
import scipy.linalg


def compute_projection_matrix(X_train, X_valid, X_test, y_train, y_valid, y_test, k, alpha, beta, class_num):
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
    X_l = np.concatenate((X_train, X_valid), axis=0)
    y_l = np.concatenate((y_train, y_valid), axis=0)
    X = np.concatenate((X_l, X_test), axis=0)
    idx = np.argsort(y_l)
    X_l = X_l[idx]
    y_l = y_l[idx]
    m, d = X.shape
    l, _ = X_l.shape
    for label in y_l:
        class_num[label] += 1
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    S = knn.kneighbors_graph(X).toarray()
    D = np.diag(np.sum(S, axis=1))  # 度数矩阵
    L = D - S

    blocks = [np.ones((class_num[i], class_num[i])) *
              (1/class_num[i]) for i in range(class_num)]
    W_l = scipy.linalg.block_diag(*blocks)
    W = np.block([[W_l, np.zeros((l, m - l))],
                 [np.zeros((m - l, l)), np.zeros((m - l, m - l))]])
    l = int(l)
    I_tilde = np.block([[np.identity(l), np.zeros((l, m - l))],
                       [np.zeros((m - l, l)), np.zeros((m - l, m - l))]])
    I = np.identity(d)

    S_b = np.dot(np.dot(X.T, W), X)
    print(I.shape, (beta*I).shape)
    S_t = np.dot(np.dot(X.T, I_tilde + alpha * L), X)+beta*I

    eigvals, eigvecs = np.linalg.eig(np.dot(np.linalg.pinv(S_t), S_b))
    eigvals_sorted = eigvals.argsort()    # 获取前c个非零特征值对应的特征向量
    eigvecs_sorted = eigvecs[:, eigvals_sorted]
    A = eigvecs_sorted[:, :class_num]
    return A, X
