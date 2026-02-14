import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

w_values = np.arange(0.0, 4.1, 0.1)
b_values = np.arange(0.0, 4.1, 0.1)
'''
np.meshgrid是numpy中的一个函数,用户生成网格数据
w_values和b_values是两个一维数组,分别表示w和b的取值范围
w_grid和b_grid是两个二维数组,表示w和b的所有组合.
举例：
w_grid:
 [[0 1 2]
 [0 1 2]
 [0 1 2]]
b_grid:
 [[7 7 7]
 [8 8 8]
 [9 9 9]]
 w_grid[i, j] 表示第 i 行第 j 列的 w 值。
例如,w_grid[0, 0] 是 0,对应 b[0,0], b=7
即对于 w_grid[i, j]，对应的 b 值是 b_grid[i, j]。

np.meshgrid 默认使用 indexing='xy'，
这会生成符合笛卡尔坐标系（行对应 Y 轴，列对应 X 轴）的网格矩阵。
假设输入的 w_values 长度为 N，b_values 长度为 M，
则生成的 w_grid 和 b_grid 形状为 ​**(M, N)**：
​**w_grid** 的每一行是 w_values 的复制（对应 X 轴方向）。
​**b_grid** 的每一列是 b_values 的复制（对应 Y 轴方向）。
例如，若 w_values = [1, 2, 3] 和 b_values = [4, 5, 6, 7]：
w_grid = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]  # 形状 (4, 3)
b_grid = [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]  # 形状 (4, 3)
'''
w_grid, b_grid = np.meshgrid(w_values, b_values)
'''
是 NumPy 中的一个函数，用于创建一个与输入数组形状相同但全为零的数组。
'''
loss_grid = np.zeros_like(w_grid)

'''
​**i 遍历 w_values 的索引**​（对应网格的列方向）。
​**j 遍历 b_values 的索引**​（对应网格的行方向）。

由于 w_grid 和 b_grid 的形状是 ​**(M, N)**​
（M 是 b_values 长度，N 是 w_values 长度），
loss_grid 的索引 [j, i] 对应以下逻辑：
行索引 j：表示 b_values 的第 j 个值（Y 轴方向）。
列索引 i：表示 w_values 的第 i 个值（X 轴方向）。
'''
for i in range(len(w_values)):
    for j in range(len(b_values)):
        w = w_values[i]
        b = b_values[j]
        total_loss = 0
        for x, y in zip(x_data, y_data):
            total_loss += loss(x, y, w, b)
        '''
        为什么不能使用 loss_grid[i, j]？
        若将索引顺序改为 [i, j]，会导致以下问题：

        ​形状不匹配：loss_grid 的形状是 (M, N)，
        而 i 的范围是 0~N-1，j 的范围是 0~M-1。
        若使用 loss_grid[i, j]，
        当 i >= M 或 j >= N 时会出现越界错误。

        ​坐标错位：w_grid[j, i] 的值是 w_values[i]，
        而 b_grid[j, i] 的值是 b_values[j]。
        若使用 loss_grid[i, j]，绘制 3D 图时，
        w 和 b 的坐标会与损失值错位，导致图像错误。
        '''
        loss_grid[j, i] = total_loss / 3  # MSE

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w_grid, b_grid, loss_grid, cmap='viridis')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
plt.savefig('3d_plot.png')
plt.show()