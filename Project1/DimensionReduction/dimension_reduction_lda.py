# 对数据进行降维
from data_process import data_process
# from data_process import X_test,X_train,y_test,y_train,Labels
# from data_process import X_test_MCI,X_train_MCI,y_test_MCI,y_train_MCI
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import Isomap
from sklearn.manifold import MDS

# 计算相关系数矩阵画热力图
# def heatmap():
#     correlation = np.corrcoef(X_train.T) # X_train的形状为(214, 186)
#     print(correlation)

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(correlation, cmap="coolwarm", annot=False)  # 使用不同的颜色映射和关闭标注

#     # 添加标签和标题
#     plt.xlabel("Features")
#     plt.ylabel("Features")
#     plt.title("Correlation Heatmap")

#     # 显示热力图
#     plt.show()

# def QQ():
#     # Q-Q图 验证是否符合正态分布, 除了极少数特殊的列, 基本符合
#     for i in range(10):
#         normal_sample = np.random.normal(size=len(X_train))

#         stats.probplot(X_train[:, i], dist="norm", plot=plt)
#         plt.title("Q-Q Plot for Sample Dimension 0")
#         plt.xlabel("Theoretical Quantiles")
#         plt.ylabel("Sample Quantiles")

#         # 显示图形
#         plt.show()

def dimension_reduction(X,y,dim=2,method='LDA'):
    print(f'before dimension reduction: {X.shape}')
    if method=='PCA':
        pca=PCA(n_components=dim)
        X=pca.fit_transform(X)
    elif method=='LDA':
        lda = LinearDiscriminantAnalysis(n_components=dim,solver='svd')
        X = lda.fit_transform(X, y)
        return X,lda.scalings_
    elif method=='t-SNE':
        tsne = TSNE(n_components=dim, learning_rate='auto', init='random', perplexity=20, early_exaggeration=15,
                    random_state=0)
        X = tsne.fit_transform(X)
    elif method=='LLE':
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=dim, method='standard')
        X = lle.fit_transform(X)
    elif method=='Laplacian':
        # embedding = SpectralEmbedding(n_components=n_components, affinity='rbf', gamma=None, random_state=2, eigen_solver=None, n_neighbors=None, n_jobs=None)
        embedding = SpectralEmbedding(n_components=dim)
        X = embedding.fit_transform(X)
    elif method=='Isomap':
        isomap = Isomap(n_neighbors=10, n_components=dim)
        X = isomap.fit_transform(X)
    elif method=='MDS':
        mds=MDS(n_components=dim)
        X=mds.fit_transform(X)
    return X

def visualization(X,y,dim=2,method='LDA',class_num=5):
    if dim==2:
        color_mapping = {-1:'k', 0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'm'}  # 颜色映射字典
        # 绘制散点图
        # i=1
        for label, color in color_mapping.items():
            # if i>class_num:
            #     break
            # i+=1
            # 获取当前类别的数据点索引
            class_indices = (y == label)
            # print(label,class_indices,y == label)
            # 绘制当前类别的数据点
            # print(X[class_indices, 0], X[class_indices, 1])
            plt.scatter(X[class_indices, 0], X[class_indices, 1], c=color, label=f'Class {label}')

            # 标注出颜色对应的类别
            plt.text(X[class_indices, 0].mean(), X[class_indices, 1].mean(), f'Class {label}',
                    color='black', fontsize=8, ha='center', va='center')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(method+' Dimensionality Reduction')
        plt.colorbar()
        plt.show()
    if dim==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color_mapping = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'm'}
        i=1
        for label, color in color_mapping.items():
            if i>class_num:
                break
            i+=1
            mask = (y == label)
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=color, label=f'Class {label}')

        ax.legend()
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.title('3D Scatter Plot of Dimension-Reduced Data')
        plt.show()

# Methods=['LDA','PCA','t-SNE','LLE','Laplacian','Isomap']
# dim=2
# class_num=3
# # for method in Methods:
# # 测试集
random_seed=0
X_train,y_train,X_valid,y_valid,X_test,y_test=data_process(random_seed)
# X_train_reduced=dimension_reduction(X_train,y_train,dim=10,method='PCA')
# X_train_reduced,_=dimension_reduction(X_train_reduced,y_train,dim=2,method='LDA')
X_train_reduced,_=dimension_reduction(X_train,y_train,dim=2,method='LDA')

# # # 训练集
# # X_test_reduced=dimension_reduction(X_test_MCI,y_test_MCI,dim=140,method='PCA')
# # X_test_reduced=dimension_reduction(X_test_reduced,y_test_MCI,dim=110,method='Laplacian')
# # X_test_reduced=dimension_reduction(X_test_reduced,y_test_MCI,dim=dim,method='LDA')


# # 可视化
# print(len(X_train_reduced))
visualization(X_train_reduced,y_train,dim=2,method='LDA+Isomap',class_num=5)
# # visulization(X_test_reduced,y_test_MCI,dim=3,method=method,class_num=4)
plt.show()