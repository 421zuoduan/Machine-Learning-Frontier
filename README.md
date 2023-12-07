# Machine-Learning-Frontier 2

项目结构如下:

```
├── configs/
│   └── config.py
├── Database/
│   ├── __init__.py
│   ├── database.py
│   ├── dataset.py
│   └── utils.py
├── models/
│   ├── GCN.py
│   └── GraphGRU.py
├── Project1/
├── .gitignore
├── main.py
├── README.md
├── Tokenized.py
├── dtwknn.py
├── normaltest.py
├── ViT.ipynb
└── utils.py
```

其中:
- `Database/` 文件夹用于处理数据, 包括但不限于加载数据, 归一化, 离散化等.
- `models/` 文件夹中是一些尝试过的模型
- `others/` 放了一些其他尝试的但不规范的代码
- `Project1/` 部分是项目 1 的有用的尝试的代码
- `Tokenized.py` 用于将文本数据映射到词表

由于本项目采用了若干大模型 (包括 RoBERTa, BERT, ELECTRA, MacBERT), 这些大模型较大, 未包括在在上述文件结构中.