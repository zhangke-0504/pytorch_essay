import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 从sklearn包导入糖尿病数据集
from sklearn.datasets import load_diabetes
import numpy as np


class DiabetesDataset():
    def __init__(self):
        # 加载糖尿病数据集
        diabetes = load_diabetes()
        # 提取特征矩阵和标签
        X = diabetes .data  # 形状为 (N, 10) 的 NumPy 数组
        y = diabetes .target  # 形状为 (N,) 的 NumPy 一维数组
        threshold = np.median(y)  # 使用中位数作为阈值（或自定义数值如150）
        y_binary = np.where(y > threshold, 1, 0).astype(np.float32)
        self.x_data = torch.from_numpy(X).float()  # 形状 [N, 10]
        self.y_data = torch.from_numpy(y_binary).float().view(-1, 1)  # 形状 [N, 1]
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset, 
                          batch_size=32, 
                          shuffle=True,
                          )


# 自定义模型结构
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10,8)
        self.linear2 = torch.nn.Linear(8,6)
        self.linear3 = torch.nn.Linear(6,4)
        self.linear4 = torch.nn.Linear(4,1)
        # self.activate = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1、Prepare data
        inputs, labels = data
        # 2、Forward
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3、Backward
        optimizer.zero_grad()
        loss.backward()
        # 4、Update
        optimizer.step()

