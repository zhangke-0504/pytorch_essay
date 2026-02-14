import zipfile
import torch
from torch.utils.data import Dataset, DataLoader

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集（保留原始顺序）"""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')[:10000]
    
    # 修正字符索引映射逻辑（保持顺序一致性）
    unique_chars = list(sorted(set(corpus_chars)))
    char_to_idx = {char: i for i, char in enumerate(unique_chars)}
    idx_to_char = {i: char for i, char in enumerate(unique_chars)}
    
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, len(unique_chars)

class LyricsDataset(Dataset):
    def __init__(self, corpus_indices, seq_length=35):
        self.seq_length = seq_length
        self.data = torch.LongTensor(corpus_indices)
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        inputs = self.data[idx:idx+self.seq_length]
        targets = self.data[idx+1:idx+self.seq_length+1]
        return inputs, targets

# 初始化数据集
corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
dataset = LyricsDataset(corpus_indices)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
class CharLSTM(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 嵌入层将输入维度从vocab_size映射到hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)  
        
        # 修改所有W_x参数的输入维度为hidden_size（原为vocab_size）
        self.W_xi = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.W_hi = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_i = torch.nn.Parameter(torch.zeros(hidden_size))
        
        # 遗忘门参数
        self.W_xf = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.W_hf = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_f = torch.nn.Parameter(torch.zeros(hidden_size))
        
        # 输出门参数
        self.W_xo = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.W_ho = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_o = torch.nn.Parameter(torch.zeros(hidden_size))
        
        # 候选记忆参数
        self.W_xc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.W_hc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.b_c = torch.nn.Parameter(torch.zeros(hidden_size))
        
        # 输出层
        self.W_hq = torch.nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_q = torch.nn.Parameter(torch.zeros(vocab_size)) 
        
    def forward(self, inputs, state=None):
        embedded = self.embedding(inputs)  # (batch_size, seq_len, hidden_size)
        batch_size = inputs.size(0)
        device = inputs.device
        
        # 初始化隐藏状态（参考网页3的状态初始化）
        if state is None:
            H = torch.zeros(batch_size, self.hidden_size).to(device)
            C = torch.zeros(batch_size, self.hidden_size).to(device)
        else:
            H, C = state
            
        outputs = []
        # 遍历时间步
        for X in embedded.unbind(1):  # X形状变为(batch_size, hidden_size) 
            # 输入门计算
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
            # 遗忘门计算
            F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
            # 输出门计算 
            O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
            # 候选记忆
            C_tilda = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
            # 更新记忆细胞
            C = F * C + I * C_tilda
            # 更新隐藏状态
            H = O * torch.tanh(C)
            # 输出预测
            Y = torch.matmul(H, self.W_hq) + self.b_q
            outputs.append(Y)
            
        return torch.stack(outputs, dim=1), (H, C)
    

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    model = CharLSTM(vocab_size, 256).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(160):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播（参考网页5的训练流程）
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
        generate_text(model, idx_to_char, device)  # 生成示例文本

def generate_text(model, idx_to_char, device, prefixes=['分开', '不分开'], length=50):
    model.eval()
    for start_str in prefixes:
        chars = [ch for ch in start_str]
        inputs = torch.tensor([[char_to_idx[ch] for ch in chars]], dtype=torch.long).to(device)
        
        with torch.no_grad():
            hidden = None
            for _ in range(length):
                outputs, hidden = model(inputs, hidden)
                predicted = outputs.argmax(dim=-1)[:,-1].unsqueeze(0)
                inputs = predicted
                chars.append(idx_to_char[predicted.item()])
        
        print('生成文本:', ''.join(chars))


if __name__ == "__main__":
    train_model()