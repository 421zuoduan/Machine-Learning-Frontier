import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from configs.config import *
from Database.utils import split_train_valid_test
from utils import *
from fastdtw import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

cnt = 0

db_cfg = FTD_fMRI_config
db = database(db_cfg.name, db_cfg.name)
random_state = 1

data, label, labelmap = db.rawdatatonumpy()
train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(
    data, label, random_state)

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=2, metric=euclidean)

# 拟合模型
knn.fit(np.array(train_data).reshape(train_data.shape[0], -1),
        np.array(train_label).flatten())

# 预测
valid_pred = knn.predict(
    np.array(valid_data).reshape(valid_data.shape[0], -1))
test_pred = knn.predict(
    np.array(test_data).reshape(test_data.shape[0], -1))

# 计算准确率
valid_accuracy = accuracy_score(valid_label, valid_pred)
test_accuracy = accuracy_score(test_label, test_pred)

print("Validation Accuracy:", valid_accuracy)
print("Test Accuracy:", test_accuracy)
