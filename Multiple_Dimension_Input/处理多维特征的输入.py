import torch
# 打印pytorch版本
print(torch.__version__)
# 查看gpu是否可用
print('Is cuda avaible? ', torch.cuda.is_available())
from torch.utils.data import TensorDataset, DataLoader

# 从sklearn包导入糖尿病数据集
from sklearn.datasets import load_diabetes
import numpy as np

# 加载糖尿病数据集
diabetes = load_diabetes()

# 提取特征矩阵和标签
X = diabetes.data  # 形状为 (N, 10) 的 NumPy 数组
y = diabetes.target  # 形状为 (N,) 的 NumPy 一维数组
'''
根据阈值二值化标签
sklearn中的糖尿病数据标签不是0/1二分类类型
所以需要二值化处理，不然用二分类交叉熵计算损失就会报错
'''
threshold = np.median(y)  # 使用中位数作为阈值（或自定义数值如150）
y_binary = np.where(y > threshold, 1, 0).astype(np.float32)

# 将 NumPy 数组转为 PyTorch 张量
x_data = torch.from_numpy(X).float()  # 形状 [N, 10]
y_data = torch.from_numpy(y_binary).float().view(-1, 1)  # 形状 [N, 1]
# 查看数据的维度
print('x_data:', x_data.shape)
print('y_data:', y_data.shape)

# 将原始数据封装为Dataset
dataset = TensorDataset(x_data, y_data)  # 输入形状需匹配(batch_size, features)
from torch.utils.data import random_split

# 定义划分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 计算各子集样本数
total_samples = len(dataset)
train_size = int(train_ratio * total_samples)
val_size = int(val_ratio * total_samples)
test_size = total_samples - train_size - val_size

# 使用random_split进行随机划分（确保可复现性）
generator = torch.Generator().manual_seed(2023)  # 固定随机种子
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=generator
)

# 创建数据加载器（训练集需打乱）
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

best_val_loss = float('inf')
early_stop_counter = 0
for epoch in range(100):
    # 训练模型
    model.train()
    train_loss = 0.0
    # 遍历每个小批量
    for batch_x, batch_y in train_loader:
        # 前向传播（仅当前batch）
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 验证集验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()

    # === 早停机制 ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best.pt')
        early_stop_counter = 0
    else:
        early_stop_counter +=1
        if early_stop_counter >= 10:  # 连续10轮无改进则停止
            break
    print('epoch:', epoch, 'batch:', 'train_loss:', train_loss, 'val_loss:', val_loss)


# 测试集评估最佳模型性能
model.load_state_dict(torch.load('best.pt', weights_only=True))
model.eval()
correct = 0

# torch.no_grad()是为了避免在测试时计算梯度，从而节省内存和计算资源
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_pred = model(batch_x)
        predicted = (y_pred > 0.5).float()
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / len(test_dataset)
print(f'Test Accuracy: {accuracy:.2f}%')




