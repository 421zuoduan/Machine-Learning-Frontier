# Machine Learning Frontier 2

本项目是屯子神技大学计算机科学与工程学院《机器学习前沿》课程的第二个项目。

文件路径，大语言模型路径，模型超参数在`config.py`中可以修改。


## Data Preparation

数据集的文件分布如下：

```
|-$ROOT/datasets
├── ADNI_90_120_fMRI.mat
├── ADNI.mat
├── FTD_90_200_fMRI.mat
├── OCD_90_200_fMRI.mat
├── PPMI.mat
```

对数据集的读取与处理在`dataloader.py`文件与`Database`文件夹中可以查看详细过程。



## Train and Test

进入虚拟环境中，运行以下命令：

```
python main.py
```

如果在linux操作系统中想要关闭终端但保持进程依然存在，运行以下命令：

```
nohup python main.py &
```

测试过程将在训练结束后自动进行，训练日志将被保存在`logs`中，训练结果将被保存在`checkpoint`文件夹中

