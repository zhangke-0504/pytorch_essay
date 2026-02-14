import torch
# 打印pytorch版本
print(torch.__version__)
# 查看gpu是否可用
print('Is cuda avaible? ', torch.cuda.is_available())

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 创建一个初始值为1.0的一维张量(Tensor)，作为模型的权重参数
w = torch.Tensor([1.0])
# 告知PyTorch需要计算该张良的梯度，以便后续通过反向传播自动更新权重
w.requires_grad = True

# 虽然函数名是前馈，但这里实际上是定义模型
def forward(x):
    return x *w

# 计算损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    # 定义学习率
    lr = 0.01
    for x, y in zip(x_data, y_data):
        '''
        前馈，用于计算得到概论迭代的损失,
        因为这里涉及到张量w的运算，而涉及到张量的计算过程，
        pytorch会自动构建计算图。
        即这一行运行pytorch会自动构建整个计算损失的计算图，
        以便后续基于该计算图进行反向传播
        '''
        l = loss(x, y)
        '''
        反向传播，会计算前馈过程中所有张量的梯度，
        并存储在requires_grad=True的张量中，即w。
        换句话说，也可以理解成，计算图中所有包含w的节点的梯度，
        反向传播会对这些节点的梯度进行计算，并存储在w.grad中。
        当梯度被存到了w.grad中之后，该轮迭代的计算图就会被释放
        下一轮迭代将构建新的计算图
        '''
        l.backward()
        '''
        这里的w.grad.item()表示取得标量,
        是只有大小没有方向的量
        可以理解成提取了w.grad的数值.

        同理，这里的l也是个张量，想要取出损失的值，
        就要取标量，即l.item()
        '''
        print("\tgrad:", x, y, w.grad.item())
        '''
        更新权重
        这里的梯度为什么要取到w.grad.data而不是w.grad
        原因是防止pytorch创建新的计算图，从而影响后续的反向传播。
        张量(tensor)内部就是data和grad，这里更新的是权重的data部分，
        通过取出grad部分的数值来更新w的data
        (w.grad也是一个tensor，所以数值存储在data部分)
        '''
        w.data = w.data - lr * w.grad.data

        '''
        对w张量中的梯度清零,
        如果不清零，则下一次计算梯度的时候，
        计算途中计算梯度是进行了一个梯度的累加（加上未清零的梯度）
        这样会导致梯度越来越大，从而影响训练效果
        但有些特殊场景也会需要用到梯度的累加
        所以pytorch对于梯度的清零并不是内置自动的，
        而是需要手动设置，这样可以更灵活地控制梯度
        '''
        w.grad.data.zero_()
    
    print("progress:", epoch, l.item())

print("predict (after training)", 4, forward(4).item())