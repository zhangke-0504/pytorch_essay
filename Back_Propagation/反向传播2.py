import torch
# 打印pytorch版本
print(torch.__version__)
# 查看gpu是否可用
print('Is cuda avaible? ', torch.cuda.is_available())

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 初始化模型的权重和偏置
w1, w2, b = [
    torch.nn.Parameter(torch.tensor([val], dtype=torch.float32))
    for val in [1.0, 1.0, 0.0]
]


# 虽然函数名是前馈，但这里实际上是定义模型
def forward(x):
    return w1 * x ** 2 + w2 * x + b

# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())


# 定义学习率
lr = 0.01
# 构建这个列表的目的是为了方便批量更新和清零操作
params = [w1, w2, b]
for epoch in range(100):

    for x, y in zip(x_data, y_data):
        # 前馈计算损失
        l = loss(x, y)
        # 反向计算梯度
        l.backward()
        
        print("\tgrad:", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        
        # 批量更新参数
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad

        # 批量清零梯度
        with torch.no_grad():
            for param in params:
                param.grad.zero_()
    
    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())