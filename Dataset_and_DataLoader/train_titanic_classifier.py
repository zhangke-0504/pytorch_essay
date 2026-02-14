'''
泰坦尼克号这份数据的特征列如下：
1、PassengerID: 乘客ID
2、Pclass: 乘客等级(如头等舱/二等舱/三等舱)
2、Name: 姓名
3、Sex: 性别
4、Age: 年龄
5、SibSp: 同行的兄弟姐妹/配偶数
6、Parch: 同行的父母/子女数
7、Ticket: 船票号码
8、Fare: 票价
10、Cabin: 客舱号
11、Embarked: 登船港口

处理这份数据的思考如下：
1、没用的特征是PassengerID、Name、Ticket
2、可以合并的特征如下:
(1)Pclass+Embarked : 代表着乘客在船上所处的一个位置， 
比如靠近出口的头等舱,另外，著名港口登船的有钱人会偏多，也存在一定的深层关联
Cabin特征缺失值太多，且对结果影响不大，所以不使用
(2)Sex+Age : 代表乘客的一个身体状态, 比如一个年轻的男性
(3)SibSp+Parch : 代表同行数
3、Fare显示的是乘客家庭的一个资源,可以反应家庭经济状况与存活率的关系
但是这个特征数据差异太大，需要做归一化

但是合并了之后，前两个合并特征是英文字符串，为了转换成数值类型，
需要用独热编码做处理，最后生成多种类型特征，再进行训练
'''

import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import joblib

# 读取数据并删除无用列
data = pd.read_csv('./titanic/train.csv').drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
)

# ================= 特征工程 =================
# 初始化标准化器
scaler_z = StandardScaler()
# 若有缺失，使用中位数填充（避免异常值影响）
data['Fare'].fillna(data['Fare'].median(), inplace=True)
# 训练集拟合并转换
data['Fare'] = scaler_z.fit_transform(data[['Fare']])

# Pclass + Embarked 处理（确保没有 NaN）
data['Embarked'] = data['Embarked'].str[0].fillna('Unknown')  # 填充缺失值
# 三者合并称为社会阶层，合并的另一个原因是这里面有两个特征都是英文字符串
data['Class_Social'] = data['Pclass'].astype(str) + '_' + data['Embarked']

# Sex + Age 处理（确保没有 NaN）
# 先填充 Age 缺失值（例如用中位数）
data['Age'] = data['Age'].fillna(data['Age'].median())
bins = [0, 12, 18, 30, 50, 100]
labels = ['Child', 'Teen', 'Young', 'Adult', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
data['Sex_Age'] = data['Sex'] + '_' + data['Age_Group'].astype(str)

# Family 处理
data['Family_Size'] = data['SibSp'] + data['Parch']
# 合并之后数值可能超过1，做一次归一化可以保证数据的稳定性
# 训练集拟合并转换
data['Family_Size'] = scaler_z.fit_transform(data[['Family_Size']])

# 清理中间列
final_data = data.drop(
    columns=['Pclass', 'Embarked', 'Sex', 'Age', 'Age_Group', 'SibSp', 'Parch']
)

# ================= 独热编码 =================
'''
独热编码​​是一种将​​分类变量​​（如性别、颜色、国家）转换为​​二进制向量​​的技术，
用于解决机器学习算法无法直接处理非数值型数据的问题。
其核心原理是：
​​唯一性映射​​：每个类别对应一个唯一的二进制向量，向量长度为类别总数，且仅有一个元素为1（表示激活状态），其余为0。
​​示例​​：颜色特征 ["红", "绿", "蓝"] 编码为：
红 → [1, 0, 0]
绿 → [0, 1, 0]
蓝 → [0, 0, 1]
​​消除数值误导​​：避免直接使用整数编码（如红=1、绿=2）导致模型误判类别间存在顺序或大小关系。
'''
# 定义要编码的列
categorical_features = ['Class_Social', 'Sex_Age']
print("Class_Social NaN数量:", final_data['Class_Social'].isna().sum())
print("Sex_Age NaN数量:", final_data['Sex_Age'].isna().sum())
final_data['Class_Social'] = final_data['Class_Social'].fillna('Unknown')  # 修正可能的拼写错误
final_data['Sex_Age'] = final_data['Sex_Age'].fillna('Unknown')

# 初始化编码器（修复后的参数）
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # 或删除 sparse_output 后加 .toarray()
encoded_features = encoder.fit_transform(final_data[categorical_features])
# 获取编码后的列名
encoded_columns = encoder.get_feature_names_out(categorical_features)

# 生成编码后的 DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

# 合并到最终数据时显式命名
final_data_encoded = pd.concat(
    [
        final_data.drop(columns=categorical_features),  # 原始非编码列（如 Family_Size）
        encoded_df                                       # 编码后的新列
    ], 
    axis=1
)

# 获取最终列名（排除目标列 Survived）
feature_columns = final_data_encoded.columns[final_data_encoded.columns != 'Survived'].tolist()
# 将编码器和特征列名​​共同保存
joblib.dump(
    {
        'encoder': encoder,
        'feature_columns': feature_columns  # 仅包含输入特征列（不含 Survived）
    }, 
    'titanic_encoder.pkl'
)
# 保存特征列名（排除标签列）
pd.DataFrame(
    final_data_encoded.columns[final_data_encoded.columns != 'Survived']
).to_csv('train_processed_columns.csv', index=False)
# 输出验证
print(final_data_encoded.head())
print(final_data_encoded.shape)


# ================= 分离特征和标签 =================
# 提取 Survived 列作为标签 y
y = final_data_encoded['Survived'].values
X = final_data_encoded.drop(columns=['Survived']).values

# ================= 分离特征和标签 =================
# 提取 Survived 列作为标签 y
y = final_data_encoded['Survived'].values
X = final_data_encoded.drop(columns=['Survived']).values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 打印验证
print("\n张量形状验证:")
print(f"X_tensor 形状: {X_tensor.shape}")
print(f"y_tensor 形状: {y_tensor.shape}")

# ====== 用这些张量构建PyTorch数据集 ======
dataset = TensorDataset(X_tensor, y_tensor)
# ================= 划分训练集/验证集 =================
# 设置随机种子保证可重复性
torch.manual_seed(42)

# 定义数据集总长度
dataset_size = len(dataset)
train_size = int(0.9 * dataset_size)
val_size = dataset_size - train_size

# 随机划分数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建对应的DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 自定义模型结构
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(22,44)
        self.linear2 = torch.nn.Linear(44,11)
        self.linear3 = torch.nn.Linear(11,1)
        self.ReLU = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.2)  # 丢失率20%

    def forward(self, x):
        x = self.ReLU(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  
        return x

model = Model()

criterion = torch.nn.BCELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


# ================= 早停机制和最优模型保存 =================
best_val_loss = float('inf')
early_stop_counter = 0
patience = 1000  # 允许连续50个epoch验证损失不改善
# 动态学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200)
# 修改后的训练循环
for epoch in range(10000):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # 前向传播
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            val_loss += loss.item()
    
    print(f'Epoch: {epoch}')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best.pt')
        early_stop_counter = 0  # 重置计数器
    else:
        early_stop_counter += 1
        print(f'EarlyStop counter: {early_stop_counter}/{patience}')
    
    # 早停检查
    if early_stop_counter >= patience:
        print(f'Early stopping triggered at epoch {epoch}!')
        break

    scheduler.step(val_loss)  # 更新学习率
    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print('lr:', current_lr)

print('Training completed. Best model saved as best.pt')