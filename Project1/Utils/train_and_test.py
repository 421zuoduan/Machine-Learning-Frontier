import torch
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_and_test(net,num_epochs,train_dataset,test_dataset,batch_size,optimizer,loss,init_hidden,num_workers=0): 
    # 参数为 网络结构 训练的总epoch数目 训练数据集 测试数据集 batch大小 优化器 损失函数 
    net.to(device) # 将模型移至GPU
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    
    losses = [] # 存放训练loss的变化
    test_accs = [] # 存放测试集准确率的变化
    train_accs=[]
    n_train=len(train_dataset) # 训练集样本数目
    n_test=len(test_dataset)
    net.train()
    for epoch in range(num_epochs):
        train_l_sum, train_acc= 0.0, 0.0
        correct=0
        for X, y in train_iter: # 遍历测试集的每一个样本
            y=y.long()
            X, y = X.to(device), y.to(device) # 数据移至GPU
            output = net(X, init_hidden).to(device)
            optimizer.zero_grad() # 清空梯度
            l = loss(output, y) 
            l.backward() # 反向传播
            train_l_sum+=l.item() # 计算总的loss
            optimizer.step() # 更新模型参数
            # 计算训练准确率和测试准确率
            _,predicted=torch.max(output.data,1) # 找到最大值并返回最大值的索引
            correct+=((predicted==y).sum().item()) # 预测正确的数量
        train_acc=correct / n_train # 该 epoch 的测试准确率
        train_accs.append(train_acc)
        correct=0
        with torch.no_grad():
            for X,y in test_iter: # 测试集数据
                y=y.long()
                X, y = X.to(device), y.to(device) # 数据移至GPU
                # init_hidden=torch.zeros(num_layers*num_directions,batch_size,hidden_size).to(device)
                output=net(X,init_hidden)
                l=loss(output,y)
                _,predicted=torch.max(output.data,1)
                correct+=((predicted==y).sum().item()) # 正确的个数
            test_acc=correct/n_test
            test_accs.append(test_acc)
        losses.append(train_l_sum / n_train)
        if (epoch+1) % 50 == 0:
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / n_train, train_acc, test_acc))
        torch.cuda.empty_cache() # unleash the memory of GPU
        
    return losses, train_accs, test_accs # 返回训练过程中的所有信息