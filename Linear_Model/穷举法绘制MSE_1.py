import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w 

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
mse_list = []

# 生成一个从0.0开始到4.1结束，步长是0.1的数组
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        # 经过线性模型得到y的预测结果
        y_pred_val = forward(x_val)
        print('y_pred_val=', y_pred_val)
        # 经过损失函数计算损失值
        loss_val = loss(x_val, y_val)
        # 将当前样本的损失+=给总损失
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
# 保存图像
plt.savefig('2d_plot.png')  # 保存为 PNG 格式