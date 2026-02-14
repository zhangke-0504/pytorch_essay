import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import joblib

# 加载测试数据
o_data = pd.read_csv("./titanic/test.csv")
data = pd.read_csv("./titanic/test.csv").drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin']
)

# =================特征工程================
# 初始化标准化器
scaler_z = StandardScaler()
# 若有缺失，使用中位数填充（避免异常值影响）
data['Fare'].fillna(data['Fare'].median(), inplace=True)
# 训练集拟合并转换
data['Fare'] = scaler_z.fit_transform(data[['Fare']])

# Pclass + Cabin 处理（确保没有 NaN）
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

# 加载编码器和特征列名（关键修复点）
encoder_data = joblib.load('titanic_encoder.pkl')
encoder = encoder_data['encoder']
train_columns = encoder_data['feature_columns']

# 对分类特征进行编码转换（新增transform步骤）
categorical_features = ['Class_Social', 'Sex_Age']
encoded_features = encoder.transform(final_data[categorical_features])

# 生成编码后的DataFrame（修正列名来源）
test_encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder.get_feature_names_out(categorical_features)
)

# 合并特征并对齐列顺序（优化reindex逻辑）
test_final = pd.concat([
    final_data.drop(columns=categorical_features), 
    test_encoded_df
], axis=1).reindex(columns=train_columns, fill_value=0)  # 自动填充缺失列为0

# 验证输出
print(test_final.head())
print(test_final.shape)  # 应输出 (418, 26) 或与训练集特征数一致

# 1. 定义模型架构（需与训练代码完全一致）
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

# 2. 加载模型权重
model = Model()
model.load_state_dict(torch.load('best.pt', weights_only=True))  # 添加安全参数
model.eval()  # 切换到评估模式

# 3. 转换为PyTorch张量
X_test = torch.tensor(test_final.values, 
                      dtype=torch.float32)

# 4. 进行预测
with torch.no_grad():
    outputs = model(X_test)
    predictions = (outputs > 0.5).int().squeeze()  # 二值化

# 1. 构建提交DataFrame
submission = pd.DataFrame({
    "PassengerId": pd.read_csv("./titanic/test.csv")["PassengerId"],
    "Survived": predictions.numpy()
})

# 2. 验证格式与示例文件一致
print(submission.head(3))
'''
   PassengerId  Survived
0          892         0
1          893         1
2          894         0
'''

# 3. 保存为CSV文件
submission.to_csv("submission.csv", index=False)