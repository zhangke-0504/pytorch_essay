import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from joblib import load  # 用于加载scaler
import torch.nn.functional as F

def preprocess_test(df):
    # 加载原始列顺序
    original_columns = pd.read_csv('original_feature_columns.csv', header=None)[0].tolist()
    df = df[original_columns]
    
    # 应用零方差过滤
    nonzero_var_mask = pd.read_csv('nonzero_var_mask.csv', index_col=0).squeeze().astype(bool)
    X_test_filtered = df.loc[:, nonzero_var_mask]
    X_test_filtered.columns = nonzero_var_mask.index[nonzero_var_mask]  # 重置列名
    
    # 标准化
    scaler = load('scaler.joblib')
    X_scaled = scaler.transform(X_test_filtered)
    
    # PCA特征筛选
    selected_features = pd.read_csv('selected_features.csv', header=None)[0].tolist()
    X_final = pd.DataFrame(X_scaled, columns=nonzero_var_mask.index[nonzero_var_mask])
    X_final = X_final.reindex(columns=selected_features, fill_value=0)  # 处理缺失列
    
    return X_final.values  # 返回numpy数组

# 加载测试数据（确保列顺序与训练一致）
test_df = pd.read_csv('./otto-group-product-classification-challenge/test.csv')

# 数据预处理（不再从checkpoint获取参数）
X_test_processed = preprocess_test(test_df)

# 转换为Tensor（使用.values保证顺序）
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
print("测试数据最终维度:", X_test_processed.shape)  # 应为 (N, 46)
assert X_test_processed.shape[1] == 46, "特征维度与模型输入不匹配！"

# 创建DataLoader（后续代码保持不变）
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    pin_memory=True
)

# 加载最佳模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(46, 40)
        self.l2 = torch.nn.Linear(40, 30)
        self.l3 = torch.nn.Linear(30, 20)
        self.l4 = torch.nn.Linear(20, 10)
        self.l5 = torch.nn.Linear(10, 9)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x

model = Net()
model.load_state_dict(torch.load('best.pt', weights_only=True))
model.eval()

# 预测过程
predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0]
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# 将预测结果转换为数值型的独热编码
preds_tensor = torch.tensor(predictions)
preds_one_hot = F.one_hot(preds_tensor, num_classes=9).numpy()

# 创建提交文件的DataFrame（包含id和独热编码）
submission = pd.DataFrame({
    'id': test_df['id'],
})
class_columns = [f'Class_{i+1}' for i in range(9)]
submission[class_columns] = preds_one_hot

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")