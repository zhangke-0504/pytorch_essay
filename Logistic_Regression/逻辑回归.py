import torch
# 打印pytorch版本
print(torch.__version__)
# 查看gpu是否可用
print('Is cuda avaible? ', torch.cuda.is_available())
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# import torchvision
# train_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=True, download=True)
# test_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=False, download=True)


# 原始数据
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

# 将原始数据封装为Dataset
dataset = TensorDataset(x_data, y_data)  # 输入形状需匹配(batch_size, features)

# 创建DataLoader并设置小批量参数
batch_size = 2  # 每批样本数（可调超参数）
shuffle = False  # 每个epoch是否打乱数据顺序
dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=shuffle)


# 构建逻辑回归模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        '''
        调用父类的构造函数
        第一个参数是你的模型类名称
        '''
        super(LogisticRegressionModel, self).__init__()
        '''
        定义一个线性层
        输入维度是1，输出维度是1
        class nn.Linear contain two member Tensors:
        weight and bias
        '''
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        '''
        这里乘以一个sigmoid函数将线性模型的输出限制在0-1之间
        '''
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

'''
逻辑回归的损失是交叉熵
'''
criterion = torch.nn.BCELoss(reduction='sum')

'''
model.parameters() 的功能是自动收集模型中所有需要优化的参数​​，
并将它们传递给优化器（如 torch.optim.SGD）进行梯度更新
'''
'''
SGD算法可以加入动量梯度下降来进行优化，这样越新的梯度用来更新的贡献就越大
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
''' 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
# optimizer = torch.optim.Rprop(model.parameters(), lr=0.01)

for epoch in range(1000):
    # 遍历每个小批量
    for batch_x, batch_y in dataloader:
        # 前向传播（仅当前batch）
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        print(epoch, loss)
        
        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # output weight and bias
        print('w = ', model.linear.weight.item())
        print('b = ', model.linear.bias.item())

        # 测试模型
        x_test = torch.Tensor([[4.0]])
        y_test = model(x_test)
        print('y_pred = ', y_test.data)



import numpy as np
import matplotlib.pyplot as plt

# 生成200个0~10之间的随机数
x = np.linspace(0, 10, 200)
# 变换成两百行一列的矩阵
x_t = torch.Tensor(x).view(200, 1)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
