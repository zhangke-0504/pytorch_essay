import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit  # 分层抽样切割法
import matplotlib.pyplot as plt

# 判断是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据及特征工程
class Dataloader:
    def __init__(self, path, is_train=True):
        """
        初始化数据加载器
        :param path: 数据文件路径
        :param is_train: 是否为训练模式（决定是否包含标签）
        """
        self.df = self._load_zip(path)
        self.is_train = is_train
        
    def _load_zip(cls, path):
        """类方法加载zip数据"""
        return pd.read_csv(path, sep='\t', compression='zip')
    
    def _handle_duplicates(self):
        """处理重复数据"""
        if self.is_train:
            # 测试集没必要去重
            self.df = self.df.drop_duplicates('SentenceId')
            self.df = self.df.drop('PhraseId', axis=1)
        else:
            # 测试集需要将PhraseId另存
            self.df_test_PhraseId = self.df[['PhraseId']].astype(str)  # 注意双中括号保留列名
            self.df = self.df.drop('PhraseId', axis=1)
        return self
    
    def _handle_missing_values(self):
        """处理缺失值"""
        if self.is_train:
            self.df = self.df.dropna(subset=['Phrase'])
            # 如果根据空格拆分（分词）之后的长度为0说明该数据为空
            self.df = self.df[self.df['Phrase'].apply(
                lambda x: len(str(x).strip().split())) > 0]
        else:
            # 如果是测试集，None数据则用空格字符替换
            self.df['Phrase'] = self.df['Phrase'].fillna(' ')
        return self
    
    def _iqr_filter(self):
        """
        箱型图过滤逻辑（类方法重构）
        这方法删的有点狠了，测试过后决定换个方法
        """
        phrase_lengths = self.df['Phrase'].apply(
            lambda x: len(str(x).strip().split()))
        
        Q1 = np.percentile(phrase_lengths, 25)
        Q3 = np.percentile(phrase_lengths, 75)
        IQR = Q3 - Q1
        
        # 计算下四分卫点的长度
        lower_bound = max(Q1 - 1.5 * IQR, 0)
        # 保留长度大于下四分卫点评论数据
        self.df = self.df[phrase_lengths >= lower_bound]
        return self

    def drop_little_words(self):
        """
        删除没有意义的短评
        这里之所以设置为3主要还是根据语法来判断
        即主+谓+宾
        例如 I love it
        """
        self.df = self.df[self.df['Phrase'].apply(
            lambda x: len(str(x).strip().split())) >= 3]
        return self
    
    def feature_engineering(self):
        """主处理方法"""     
        (self._handle_duplicates()
            ._handle_missing_values()) 
        if self.is_train:
            # # 只有训练数据才需要进行箱型图上下四分卫点算法过滤
            # self._iqr_filter()
            self.drop_little_words()
            print('标签分布:', self.df['Sentiment'].value_counts().to_dict())
            return self.df.drop('Sentiment', axis=1), self.df['Sentiment']
        else:
            print('self.df(test):', self.df.shape)
            print('self.df_test_PhraseId:', self.df_test_PhraseId.shape)
            return self.df, self.df_test_PhraseId


# 将经过特征工程处理过的数据处理成训练用的张量
class Make_Tensor:
    def __init__(self, data, is_train=True):
        """
        data是的形式如下：
        (X_train, y_train)
        (X_test)
        """
        self.data = data
        self.is_train = is_train
        if is_train:
            self.X, self.y = data
        else:
            self.X = data
    
    def padding(self):
        """根据最长的评论来补齐该特征列其他行的长度"""
        # 切割为列表
        self.X['Phrase'] = self.X['Phrase'].str.split()
        # 计算最大长度
        max_length = self.X['Phrase'].apply(len).max() 
        # 后端填充补齐长度
        self.X['Phrase'] = self.X['Phrase'].apply(
            lambda x: x + [''] * (max_length - len(x))
        )
        # # 前端填充补齐长度
        # self.X['Phrase'] = self.X['Phrase'].apply(lambda x: ['']*(max_length-len(x)) + x)
        # # 检查查看所有X['Phrase']的长度是否一致
        # # 检查长度一致性
        # length_check = self.X['Phrase'].apply(len)
        
        # # 验证逻辑
        # if not (length_check == max_length).all():
        #     # 获取异常行索引
        #     invalid_indices = self.X[length_check != max_length].index.tolist()
            
        #     # 输出调试信息
        #     error_msg = (
        #         f"填充后长度不一致！\n"
        #         f"- 理论长度: {max_length}\n"
        #         f"- 异常行数: {len(invalid_indices)}\n"
        #         f"- 首条异常行内容: {self.X.loc[invalid_indices[0], 'Phrase']}"
        #     )
        #     raise ValueError(error_msg)
        return self
        
    def make_word_dict(self):
        """构建词映射字典，包含词频统计和特殊标记处理"""
        # 初始化特殊标记（兼容空字符串的填充符）
        '''
        <UNK>的目的是因为测试集中可能存在训练集中没有的词，所以需要特殊标记
        '''
        self.token2id = {'': 0, '<UNK>': 1}  # 空字符串映射为PAD
        self.id2token = {0: '', 1: '<UNK>'}  # 反向映射
        self.word_freq = {}

        # 词频统计（过滤空字符串）
        for phrase_list in self.X['Phrase']:
            for word in phrase_list:
                if word == '':  # 跳过填充符
                    continue
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # 按词频降序排序（高频词优先分配更小的ID）
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: (-x[1], x[0])  # 词频降序，相同词频按字母升序
        )

        # 动态分配ID（从2开始）
        current_id = 2
        for word, freq in sorted_words:
            self.token2id[word] = current_id
            self.id2token[current_id] = word
            current_id += 1

        # 添加词表元数据（可选）
        self.vocab_size = len(self.token2id)  # 总词表大小（含特殊标记）
        self.num_unknown = len(self.word_freq) - (current_id - 2)  # 未登录词数
        # print('self.id2token:', self.id2token)
        print('self.vocab_size:', self.vocab_size)
        print('self.num_unknown:', self.num_unknown)
        return self
    
    def make_tensor(self):
        """根据self.id2token生成Tensor张量"""
        (self.padding()
            .make_word_dict())
        '''
        在[self.token2id.get(word, 1) for word in phrase]  # 每个phrase保持为子列表
            for phrase in self.X['Phrase']外再加一层列表的目的是为了保证输入给网络的
        数据维度为[batch_size, sequence_length],这里的batch_size=1
        '''
        self.X_tensor = torch.tensor(
            [[self.token2id.get(word, 1) for word in phrase]  # 每个phrase保持为子列表
            for phrase in self.X['Phrase']], 
            dtype=torch.long
        )
        if self.is_train:
            # 生成标签张量时直接使用self.y
            self.y_tensor = torch.tensor(self.y.values, dtype=torch.long)  # 从Series转换为numpy数组再转张量
            return self.X_tensor.to(device), self.y_tensor.to(device), self.vocab_size
        else:
            return self.X_tensor.to(device)

# 定义模型     
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        # 测试发现拟合都难，所以就别考虑过拟合问题去用这个破丢弃层了
        self.dropout = nn.Dropout(0.5)
        # self.attention = nn.Linear(hidden_dim*2, 1)  # 注意力权重计算
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch_size, seq_len, hidden_dim)
        # out = self.dropout(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        return self.fc(out)

# 早停类
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter_acc = 0    # 准确率未提升的计数
        self.counter_loss = 0   # 损失未变化的计数
        self.best_acc = -np.inf # 初始化最佳准确率为负无穷
        self.best_loss = np.inf # 初始化最佳损失为正无穷
        self.early_stop = False

    def __call__(self, val_acc, val_loss, model):
        # 条件1: 验证准确率是否超过历史最佳 + delta
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter_acc = 0  # 重置准确率计数器
            self.save_checkpoint(val_acc, val_loss, model, 'acc_improved')
        else:
            self.counter_acc += 1

        # 条件2: 验证损失是否在容忍范围内未变化
        if abs(val_loss - self.best_loss) <= self.delta:
            self.counter_loss += 1
        else:
            self.best_loss = val_loss
            self.counter_loss = 0  # 重置损失计数器
            self.save_checkpoint(val_acc, val_loss, model, 'loss_improved')

        # 双条件同时触发时早停
        if self.counter_acc >= self.patience and self.counter_loss >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f'EarlyStopping triggered: Acc停滞{self.counter_acc}轮, Loss停滞{self.counter_loss}轮')

    def save_checkpoint(self, val_acc, val_loss, model, reason):
        if self.verbose:
            print(f'Checkpoint saved [{reason}] | Acc: {val_acc:.4f}, Loss: {val_loss:.4f}')
        torch.save(model.state_dict(), 'best.pt')

# 训练模型  
def train_model(model, train_data, val_data, num_epochs):
    # 初始化组件
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=20, factor=0.5
    )

    # 训练记录
    train_losses = []
    val_accuracies = []
    early_stopper = EarlyStopping(patience=10)
    
    # 数据准备
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # 定义绘图需要的数据结构
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 添加进度条
        for inputs, labels in zip(X_train.split(16), y_train.split(16)):
            # 梯度清零
            optimizer.zero_grad()
            '''
            利用最后一个时间步的隐层来计算预测输出
            Y = torch.matmul(H, self.W_hq) + self.b_q
            这里W_hq是输出门的权重
            '''
            outputs = model(inputs)
            # 前馈计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 梯度裁剪
            # 更新权重
            optimizer.step()
            total_loss += loss.item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            loss = criterion(val_outputs, y_val) 
            val_loss = loss.item()
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val).sum().item() / y_val.size(0)
        # 打印训练集损失和验证集损失
        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(X_train):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # 早停判断
        early_stopper(val_acc, val_loss, model) 
        if early_stopper.early_stop:  # 直接通过类属性判断
            break
        
        # 学习率调整
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr}")
        
        # 记录数据
        train_losses.append(total_loss / len(X_train))
        val_accuracies.append(val_acc)
        
        # 释放显存
        torch.cuda.empty_cache()

        # 动态更新数据
        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 每50个epoch保存一次图片（新增条件判断）
        if (epoch + 1) % 50 == 0:
            # 创建新图形避免覆盖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            # 坐标轴初始化
            ax1.set_xlim(0, epoch + 1)
            ax2.set_xlim(0, epoch + 1)
            # 绘制训练/验证损失
            ax1.plot(train_losses, 'b-', label='Train Loss')
            ax1.plot(val_losses, 'r-', label='Val Loss')
            ax1.set_title(f'Training Curve (Epoch {epoch+1})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # 绘制验证准确率
            ax2.plot(val_accuracies, 'g-', label='Val Acc')
            ax2.set_title(f'Validation Accuracy (Epoch {epoch+1})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            
            # 保存带epoch编号的图片（网页1的保存方式）
            plt.savefig(f'./training_curve/training_curve_epoch_{epoch+1}.png')
            plt.close(fig)  # 关闭图形释放内存
            
            print(f"已保存第 {epoch+1} 轮的训练曲线图")

    # 保存最后一轮的模型名叫last.pt
    torch.save(model.state_dict(), 'last.pt')
    return model

# 测试模型
def test_model(model, test_data, X_test_PhraseId, model_file, batch_size=64):
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()
    
    all_predictions = []
    with torch.no_grad():
        # 分批次处理测试数据
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            all_predictions.append(preds.cpu())
            
            # 及时释放中间变量
            del batch, outputs, preds
            torch.cuda.empty_cache()
            
    predictions = torch.cat(all_predictions)
    
    # 生成结果DataFrame
    results = pd.DataFrame({
        'PhraseId': X_test_PhraseId.values.flatten(), 
        'Sentiment': predictions.cpu().numpy()
    })
    
    # 保存CSV
    csv_name = model_file.split('.')[0] + '-submission.csv'
    results.to_csv(csv_name, index=False)
    print(f"预测结果已保存至 {csv_name}")


if __name__ == '__main__':
    train_loader = Dataloader('./sentiment-analysis-on-movie-reviews/train.tsv.zip')
    X_train_origin, y_train_origin = train_loader.feature_engineering()
    
    test_loader = Dataloader('./sentiment-analysis-on-movie-reviews/test.tsv.zip', is_train=False)
    X_test_origin, X_test_PhraseId = test_loader.feature_engineering()

    # 生成张量
    X_train_tensor, y_train_tensor, vocab_size = Make_Tensor((X_train_origin, y_train_origin), is_train=True).make_tensor()
    X_test_tensor = Make_Tensor((X_test_origin), is_train=False).make_tensor()
    print(X_train_tensor.shape, y_train_tensor.shape, X_test_tensor.shape)
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train_tensor, y_train_tensor, 
    #     test_size=0.2, random_state=42, 
    #     stratify=y_train_tensor.cpu().numpy()
    # )
    # 分层抽样切割法
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(X_train_tensor.cpu(), y_train_tensor.cpu()):
        X_train, X_val = X_train_tensor[train_idx], X_train_tensor[val_idx]
        y_train, y_val = y_train_tensor[train_idx], y_train_tensor[val_idx]

    # 参数初始化
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    num_classes = 5  # 情感分类的类别数
    epochs = 200
    # 实例化模型
    model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes).to(device)
    # 训练模型
    trained_model = train_model(model, (X_train, y_train), (X_val, y_val), num_epochs=epochs)

    # 测试预测
    best_model_file = 'best.pt'
    last_model_file = 'last.pt' 
    test_model(trained_model, X_test_tensor, X_test_PhraseId, model_file=best_model_file)
    test_model(trained_model, X_test_tensor, X_test_PhraseId, model_file=last_model_file)
    