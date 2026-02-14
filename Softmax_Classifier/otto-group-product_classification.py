# Kaggle 竞赛性质
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from joblib import dump

# 核心读取代码（自动识别表头和分隔符）
df = pd.read_csv('./otto-group-product-classification-challenge/train.csv')  # 显式指定制表符分隔[2,5]
# ================ 查看标签特征 ================
print('标签种类数量：',len(list(df['target'].unique())))
print('标签种类：', list(df['target'].unique()))
'''
原始标签为 ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
将df['target']中的标签数值改为[1,2,3,4,5,6,7,8,9]
'''
df['target'] = df['target'].map({'Class_1': 1, 'Class_2': 2, 'Class_3': 3, 'Class_4': 4, 'Class_5': 5, 'Class_6': 6, 'Class_7': 7, 'Class_8': 8, 'Class_9': 9})
# ================ 将标签和数据(去掉id列)进行切割 =============
# ================ 数据切割 =============
# 假设列顺序为 [id, target, feat_1,...,feat_93]
X_raw = df.drop(['id', 'target'], axis=1)  # 原始特征矩阵
# 在读取原始数据后保存特征列名
pd.Series(X_raw.columns.tolist()).to_csv(
    'original_feature_columns.csv',
    index=False,
    header=False
)
feature_names = X_raw.columns.tolist()
y = df['target'].values
print('原始X_shape:', X_raw.shape)

# ================ 动态提取列名 =============
feature_columns = df.columns[1:-1].tolist()  # 从第三列开始（原数据未删除列时）

# ================ 删除零方差列 ===========
'''
零方差列就是样本取值完全相同的列，对模型训练没有贡献，可以删除
'''

# 保存零方差掩码（布尔型Series，包含列名信息）
nonzero_var_mask = X_raw.std() != 0
nonzero_var_mask.to_csv('nonzero_var_mask.csv')
X_filtered = X_raw.loc[:, nonzero_var_mask].values
feature_names = [name for name, keep in zip(feature_names, nonzero_var_mask) if keep]
print('去除零方差列X_shape:', X_filtered.shape)

# 标准化处理（PCA前必需步骤）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
dump(scaler, 'scaler.joblib')  # 保存完整 scaler 对象

# ========== PCA核心计算 ==========
pca = PCA().fit(X_scaled)
loadings = pd.DataFrame(pca.components_.T, 
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                        index=feature_names)

# ========== 特征重要性计算 ==========
# L2范数计算全局重要性
feature_importance_l2 = np.linalg.norm(loadings, axis=1)

# 方差加权贡献度
explained_var_ratio = pca.explained_variance_ratio_
weighted_importance = np.abs(loadings) @ explained_var_ratio

# 综合排序
importance_df = pd.DataFrame({
    'L2_Importance': feature_importance_l2,
    'Weighted_Importance': weighted_importance
}, index=feature_names).sort_values('Weighted_Importance', ascending=False)

# ========== 动态阈值筛选 ==========
# 保留重要性前50%的特征
threshold = np.median(importance_df['Weighted_Importance'])
selected_features = importance_df[importance_df['Weighted_Importance'] > threshold].index.tolist()

# 重建数据集
X_final = X_raw[selected_features]
print(f"筛选后特征维度：{X_final.shape}")
# 保存筛选后的特征列名
pd.Series(selected_features).to_csv('selected_features.csv', 
                                    index=False, header=False)

# ============== 构建dataloader
# 将X转换成张量
X_tensor = torch.tensor(X_final.values, dtype=torch.float32)
'''
将Y转成独热编码
'''
# 调整标签范围
# 直接转换为类别索引值（无需one-hot）
'''
为什么这里不需要手动转独热编码，
因为softmax内部是隐式独热编码处理
例如：
# 模型输出（logits）形状：[batch_size, num_classes]
logits = torch.tensor([[2.1, -0.5, 3.0], [1.8, 0.2, -1.0]])
# 目标标签（类别索引）形状：[batch_size]
labels = torch.tensor([2, 0])  # 对应独热编码应为 [[0,0,1], [1,0,0]]
loss = criterion(logits, labels)  # 无需手动转换
'''
y_adjusted = y - 1  # 将1-9转换为0-8
y_tensor = torch.tensor(y_adjusted, dtype=torch.long)  # 注意必须是long类型

# 验证标签形状
print("标签张量形状：", y_tensor.shape)  # 应为 (样本数, )

# 将X和y组合成Dataset
dataset = TensorDataset(X_tensor, y_tensor)
# 定义分割比例
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,  # 打乱顺序增强泛化性
    pin_memory=True  # 启用锁页内存，加速GPU传输[5](@ref)
)

# 验证集DataLoader（无需shuffle）
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)
# ================= 训练模型 ===================
# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(46, 40)
        self.l2 = torch.nn.Linear(40, 30)
        self.l3 = torch.nn.Linear(30, 20)
        self.l4 = torch.nn.Linear(20, 10)
        self.l5 = torch.nn.Linear(10, 9)

    def forward(self, x):
        # x = x.view(-1, 28*28*1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x
model = Net()

# 定义损失
criterion = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100)
# 早停机制
class EarlyStopping:
    def __init__(self, patience=1000, delta=0.001):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best.pt')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # 每100个batch打印一次loss
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, dim=1)
            # 这里的total是样本总数，一般情况下是batch_size的整数倍
            total += target.size(0)
            # 如果预测值和标签值相等则为true，所以这里统计的是正确的数量
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    val_loss /= len(val_loader)
    return val_loss

# 初始化早停类
early_stopping = EarlyStopping(patience=5)

# 主训练循环
for epoch in range(10000):
    train(epoch)
    val_loss = validate()
    scheduler.step(val_loss)  # 更新学习率
    current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
    print(f"Epoch {epoch}, Learning Rate: {current_lr}")
    
    # 早停检测
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break