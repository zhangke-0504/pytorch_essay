import numpy as np
import random
import time

# 原始数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 生成更多的随机数据，将数据集扩展到长度为 100
for _ in range(997):  # 已经有 3 个数据，所以再生成 97 个
    x = random.uniform(0.0, 10.0)  # 生成 0 到 10 之间的随机浮点数
    y = x * 2 + random.uniform(-1.0, 1.0)  # 生成与 x 线性相关的 y 值，加入一些随机噪声
    x_data.append(x)
    y_data.append(y)

# 初始权重
w = 1.0

# 线性模型
def forward(x):
    return x * w

# 计算损失
def loss(xs, ys):
    loss = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        loss += (y_pred -y) ** 2
    return loss / len(xs)

# 使用NumPy库将输入数据转换为矩阵，通过向量运算一次性计算整个批次的梯度
def gradient(xs, ys):
    return 2 * np.mean(xs * (xs * w - ys))

print('Predict (before training)', 2000, forward(2000))

# 指定学习率
lr = 0.01
# 指定每个 mini-batch 的大小
batch_size = 2  # 每个批次包含 2 个样本
x_array = np.array(x_data)  # 内存连续，切片速度比列表快5-10倍
y_array = np.array(y_data)
start_time = time.time()  # 开始时间
for epoch in range(100):
    for i in range(0, len(x_array), batch_size):
        x_batch = x_array[i:i+batch_size]
        y_batch = y_array[i:i+batch_size]

        # 计算梯度
        grad = gradient(x_batch, y_batch)

        # 更新权重
        w = w - lr * grad

        # 计算损失
        l = loss(x_batch, y_batch)
    print('progress:', epoch, "w=", w, "loss=", l)

print('Predict (before training)', 2000, forward(2000))
end_time = time.time()  # 结束时间
total_time = end_time - start_time  # 总耗时
print("Total training time: {:.2f} seconds".format(total_time))