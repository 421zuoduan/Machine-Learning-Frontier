import numpy as np
import matplotlib.pyplot as plt
from configs.config import *
from Database.utils import split_train_valid_test
from utils import *
from scipy.stats import shapiro

db_cfg = ADNI_fMRI_config  # 设置数据集
db = database(db_cfg.name, db_cfg.name)
random_state = 0

data, label, labelmap = db.rawdatatonumpy()
train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(
    data, label, random_state)

train_data = np.array(train_data)
normality = []

# 遍历每个sample
for i, sample in enumerate(train_data):
    # 初始化正态性指标列表
    normality_indices = []

    # 遍历每列数据
    for column in sample.T:
        # 进行正态性检验
        _, p_value = shapiro(column)

        # 判断是否为正态分布
        is_normal = p_value > 0.05

        # 添加正态性指标
        normality_indices.append(is_normal)
    normality_ratio = np.mean(normality_indices)
    # 输出正态性指标
    normality.append(normality_ratio)

# 绘制直方图
plt.hist(normality, bins=10, edgecolor='black')

# 设置标题和轴标签
plt.title("Normality Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

# 显示图形
plt.show()
