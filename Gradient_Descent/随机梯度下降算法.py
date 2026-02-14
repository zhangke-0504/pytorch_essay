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
def loss(x, y):
    y_pred = forward(x)
    return (y_pred -y) ** 2

# 计算梯度
def gradient(x, y):
    return 2 * x * (x * w - y)

print('Predict (before training)', 2000, forward(2000))

start_time = time.time()  # 开始时间
for epoch in range(100):
    lr = 0.01
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        # 更新权重
        w = w - lr * grad
        print("\tgrad:", x, y, grad)
        # 计算损失
        l = loss(x, y)

    print('progress:', epoch, "w=", w, "loss=", l)

print('Predict (before training)', 2000, forward(2000))
end_time = time.time()  # 结束时间
total_time = end_time - start_time  # 总耗时
print("Total training time: {:.2f} seconds".format(total_time))