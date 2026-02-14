import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
'''
很离谱的一个包，只适合拿来学习，不适合用来做项目，
load_data_jay_lyrics这函数的导出路径还是定死的一个目录结构，
离谱他妈给离谱开门，离谱到家了
最后直接把配套的学习数据下载下来放data目录下，
把d2l里的数据集加载函数复制过来改了下路径才把这周杰伦数据加载成功
'''
# import d2lzh_pytorch as d2l 
import zipfile

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

# 加载数据集（需确保数据集路径正确）
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

# 超参数设置
num_hiddens = 256
num_epochs = 160
num_steps = 35
batch_size = 32
lr = 1e-2
clipping_theta = 1e-2
pred_period = 40
pred_len = 50
prefixes = ['分开', '不分开']

class GRUModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        
        # 初始化门控参数
        # 更新门参数
        self.W_xz = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hz = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # 重置门参数
        self.W_xr = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hr = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        
        # 候选隐藏状态参数
        self.W_xh = nn.Parameter(torch.randn(vocab_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        # 输出层参数
        self.W_hq = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_q = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, inputs, state):
        """前向传播实现GRU计算逻辑"""
        # inputs形状: (seq_len, batch_size, vocab_size)
        # state形状: (batch_size, hidden_size)
        H, = state
        outputs = []
        
        for X in inputs:  # 按时间步遍历序列
            # 更新门计算
            Z = torch.sigmoid(torch.matmul(X, self.W_xz) + 
                            torch.matmul(H, self.W_hz) + self.b_z)
            
            # 重置门计算
            R = torch.sigmoid(torch.matmul(X, self.W_xr) + 
                            torch.matmul(H, self.W_hr) + self.b_r)
            
            # 候选隐藏状态
            H_tilda = torch.tanh(torch.matmul(X, self.W_xh) + 
                               torch.matmul(R * H, self.W_hh) + self.b_h)
            
            # 最终隐藏状态
            H = Z * H + (1 - Z) * H_tilda 
            
            # 输出层计算
            Y = torch.matmul(H, self.W_hq) + self.b_q
            outputs.append(Y)
        
        return torch.stack(outputs), (H,)

    def begin_state(self, batch_size):
        """初始化隐藏状态"""
        return (torch.zeros((batch_size, self.hidden_size), device=device),)

# 数据预处理
class LyricsDataset(Dataset):
    def __init__(self, indices, char_to_idx, num_steps):
        self.indices = indices  # 确保是索引列表
        self.char_to_idx = char_to_idx
        self.num_steps = num_steps
        self.vocab_size = len(char_to_idx)
        
    def __len__(self):
        return (len(self.indices) - 1) // self.num_steps  # 修正长度计算
    
    def __getitem__(self, idx):
        start = idx * self.num_steps
        end = start + self.num_steps + 1
        # 增加越界保护
        if end > len(self.indices):
            end = len(self.indices)
            start = end - self.num_steps - 1
        segment = self.indices[start:end]
        X = torch.zeros((self.num_steps, self.vocab_size))
        Y = torch.zeros((self.num_steps, self.vocab_size))
        
        for i in range(self.num_steps):
            X[i] = torch.nn.functional.one_hot(
                torch.tensor(segment[i]), self.vocab_size).float()
            Y[i] = torch.nn.functional.one_hot(
                torch.tensor(segment[i+1]), self.vocab_size).float()
        
        return X.to(device), Y.to(device)
    
def predict_rnn(model, prefix, num_chars, char_to_idx, idx_to_char, temperature=1.0):
    """生成歌词的核心预测函数
    参数:
        model: 训练好的GRU模型
        prefix: 起始字符串（需为训练集中存在的字符）
        num_chars: 需要生成的字符数
        char_to_idx: 字符到索引的字典
        idx_to_char: 索引到字符的字典
        temperature: 温度参数（控制生成随机性，越大越随机）
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        # 初始化隐藏状态（batch_size=1）
        state = model.begin_state(batch_size=1)
        
        # 处理前缀字符，生成初始隐藏状态
        outputs = [char_to_idx[prefix[0]]]
        # print('outputs_prefix:', outputs)
        # print('idx_to_char:', [idx_to_char[i] for i in outputs])
        for t in range(len(prefix)-1):
            X = torch.tensor([[outputs[-1]]], dtype=torch.long).to(device)
            X = F.one_hot(X, len(char_to_idx)).float().squeeze(0)  # 转为one-hot
            _, state = model(X.unsqueeze(0), state)
            outputs.append(char_to_idx[prefix[t+1]])
        # print('outputs:', outputs)

        # 开始生成新字符
        for _ in range(num_chars):
            X = torch.tensor([[outputs[-1]]], dtype=torch.long).to(device)
            X = F.one_hot(X, len(char_to_idx)).float().squeeze(0)
            # print('X:', X.shape)
            
            # 前向传播得到概率分布
            Y, state = model(X.unsqueeze(0), state)
            Y = Y.squeeze(0).squeeze(0) / temperature
            
            # 通过softmax转换为概率分布
            prob = F.softmax(Y, dim=0)
            
            # 从概率分布中采样
            next_char = torch.multinomial(prob, num_samples=1).item()
            outputs.append(next_char)
        print('outputs:', outputs)
        print('idx_to_char:', [idx_to_char[i] for i in outputs[len(prefix):]])

    # 将索引转换为字符（跳过前缀中已存在的字符）
    return ''.join([idx_to_char[i] for i in outputs[len(prefix):]])

# 创建数据加载器
'''
corpus:原始歌词文本数据集，通常是一个连续的字符串或字符列表。例如，周杰伦歌词数据集中可能包含所有歌词的完整文本
indices:将corpus中的每个字符映射为对应索引的列表。例如，字符“想”可能被转换为整数944
char_to_idx:字符到索引的字典映射，记录所有唯一字符及其对应的整数编号。例如：{'想': 944, '要': 233}
num_steps:每个训练样本的时间步数（序列长度）。例如，若设为6，则每个样本包含连续的6个字符
'''
# 正确调用方式（调整参数顺序）
dataset = LyricsDataset(
    indices=corpus_indices,      # 字符索引列表
    char_to_idx=char_to_idx,     # 字符到索引的字典
    num_steps=num_steps
)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
model = GRUModel(vocab_size, num_hiddens).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for X, Y in data_loader:
        # print('X.shape:', X.shape)
        # print('Y.shape:', Y.shape)
        # 转置维度为(seq_len, batch_size, vocab_size)
        X = X.permute(1, 0, 2)
        Y = Y.permute(1, 0, 2)
        
        # 初始化隐藏状态
        state = model.begin_state(X.size(1))
        
        # 前向传播
        outputs, state = model(X, state)
        
        # 计算损失
        loss = 0
        for t in range(num_steps):
            loss += criterion(outputs[t], Y[t].argmax(1))
        loss /= num_steps
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_theta)
        
        optimizer.step()
        total_loss += loss.item()
    
    # 每个epoch打印损失
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}')
    # 对每个前缀生成歌词
    prefixes = ['分开', '不分开']
    for prefix in prefixes:
        generated = predict_rnn(
            model=model,
            prefix=prefix,
            num_chars=pred_len,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            temperature=0.8  # 可调节生成随机性
        )
        print(f'前缀 "{prefix}" 生成结果:\n{prefix}{generated}\n')
