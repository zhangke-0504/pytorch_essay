import torch
# 打印pytorch版本
print(torch.__version__)
# 查看gpu是否可用
print('Is cuda avaible? ', torch.cuda.is_available())
# 原始数据
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

# 构建线性模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        '''
        调用父类的构造函数
        第一个参数是你的模型类名称
        '''
        super(LinearModel, self).__init__()
        '''
        定义一个线性层
        输入维度是1，输出维度是1
        class nn.Linear contain two member Tensors:
        weight and bias
        '''
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        '''
        pytorch中已经继承了反向传播的功能
        不需要人工实现
        除非需要自定义反向传播求梯度的方法
        那就需要更改继承的类
        '''
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# 定义损失函数
criterion = torch.nn.MSELoss(size_average=False)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    '''
    因为print函数会将loss对象转化为字符串
    所以是不会产生计算图的，不会有内存无效占用的情况
    '''
    print(epoch, loss)

    # 这里清空的是上一轮迭代的梯度
    optimizer.zero_grad()
    # 反向传播计算梯度
    loss.backward()
    # 更新权重和偏置等参数
    optimizer.step()

    # output weight and bias
    print('w = ', model.linear.weight.item())
    print('b = ', model.linear.bias.item())

    # 测试模型
    x_test = torch.Tensor([[4.0]])
    y_test = model(x_test)
    print('y_pred = ', y_test.data)


