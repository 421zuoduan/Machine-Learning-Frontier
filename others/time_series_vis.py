
import numpy as np
import matplotlib.pyplot as plt
from configs.config import *
from Database.utils import split_train_valid_test
from utils import *
import os

db_cfg = FTD_fMRI_config  # 在主函数里换数据集只改这一个参数
db = database(db_cfg.name, db_cfg.name)
random_state = 0
# setup_seed(db_cfg.seed)
data, label, labelmap = db.rawdatatonumpy()
train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(
    data, label, random_state)


# train_data = np.array(train_data)
# valid_data = np.array(valid_data)
# test_data = np.array(test_data)


dirs = ['train', 'val', 'test']
data = [train_data, valid_data, test_data]
labels = [train_label, valid_label, test_label]
for di, da, la in zip(dirs, data, labels):
    cnt = [0]*4
    # 遍历每个sample
    output_folder = "FTD/"+di
    for i, sample in enumerate(da):
        # 创建一个新的图形
        plt.figure()

        # 绘制折线图
        x = np.arange(da.shape[2])  # x轴范围为0到119
        for line in sample:
            plt.plot(x, line, color='black')

        # 移除标题、坐标轴和刻度
        plt.axis('off')
        os.makedirs(output_folder+f"/{la[i]}", exist_ok=True)
        # 保存图像到文件夹
        filename = os.path.join(
            output_folder+f"/{la[i]}", "0"+f"0{cnt[la[i]]}.png"if cnt[la[i]] < 10 else "0"+f"{cnt[la[i]]}.png")
        cnt[la[i]] += 1
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

        # 关闭当前图形
        plt.close()
