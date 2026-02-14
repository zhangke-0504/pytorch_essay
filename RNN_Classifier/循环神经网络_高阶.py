import torch
from torch.utils.data import Dataset, DataLoader
import time
import gzip
import csv
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import numpy as np

# 定义一个自定义数据集类 NameDataset，用于加载和处理名字数据
class NameDataset(Dataset):
    # 初始化方法，根据是否为训练集来加载对应的数据文件
    def __init__(self, is_train_set=True):
        # 根据是否为训练集设置不同的文件路径
        filename = './data/names_train.csv.gz' if is_train_set else './data/names_test.csv.gz'
        # 使用 gzip 打开文件，读取 csv 格式数据
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)  # 将读取的数据行存储为列表
        # 提取名字和对应的国家
        self.names = [row[0] for row in rows]
        self.len = len(self.names)  # 数据集的长度（名字数量）
        self.countries = [row[1] for row in rows]
        # 获取国家列表（去重并排序）
        self.country_list = list(sorted(set(self.countries)))
        # 创建国家到索引的映射字典
        self.country_dict = self.getCountryDict()
        # 国家的数量
        self.country_num = len(self.country_list)

    # 根据索引获取一个名字及对应的国家索引
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    # 获取数据集的长度
    def __len__(self):
        return self.len
    
    # 创建国家到索引的映射字典
    def getCountryDict(self):
        country_dict = dict()
        #为 每个国家分配一个唯一的索引
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict
    
    # 根据索引获取对应的国家名称
    def idx2country(self, index):
        return self.country_list[index]
    
    # 获取国家的数量
    def getCountriesNum(self):
        return self.country_num
    
# 设置模型训练的超参数
HIDDEN_SIZE = 100  # 隐藏层大小
BATCH_SIZE = 256   # 批处理大小
N_LAYER = 2        # RNN 的层数
N_EPOCHS = 100     # 训练的轮数
N_CHARS = 128      # 字符集的大小（ASCII 字符）

# 创建训练集和测试集的数据加载器
trainset = NameDataset(is_train_set=True)  # 训练集
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 批处理训练数据，打乱顺序
testset = NameDataset(is_train_set=False)  # 测试集
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)   # 批处理测试数据，不打乱顺序

# 获取国家的数量
N_COUNTRY = trainset.getCountriesNum()

# 定义一个函数，用于将张量移动到 GPU（如果可用）
def create_tensor(tensor):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = tensor.to(device)
    return tensor

# 定义一个函数，将名字转换为字符的 ASCII 码列表及其长度
def name2list(name):
    return [ord(c) for c in name], len(name)

# 定义一个函数，将名字和国家索引转换为模型可处理的张量
def make_tensors(names, countries):
    # 将每个名字转换为字符的 ASCII 码列表及其长度
    sequences_and_lengths = [name2list(name) for name in names]
    # 分离字符序列和序列长度
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])

    # 创建一个零张量，用于存储所有名字的字符序列
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    # 填充字符序列到张量中
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 按序列长度降序排列，以便使用 pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)    
    # 根据排列后的索引重新排列序列张量和国家索引
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    # 返回序列张量、序列长度和国家索引张量
    return create_tensor(seq_tensor), \
           seq_lengths, \
           create_tensor(countries)

# 定义一个 RNN 分类器类
class RNNClassifier(torch.nn.Module):
    # 初始化方法，设置输入大小、隐藏层大小、输出大小等参数
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # 是否为双向 RNN
        self.n_directions = 2 if bidirectional else 1

        # 定义嵌入层，将字符索引转换为嵌入向量
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        # 定义 GRU 层
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, 
                                bidirectional=bidirectional)
        # 定义全连接层，用于将 GRU 的输出映射到输出大小
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    # 初始化隐藏状态
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_tensor(hidden)
    
    # 前向传播方法
    def forward(self, input, seq_lengths):
        # 转置输入张量，使其形状为 (SeqLen, BatchSize)
        input = input.t()
        batch_size = input.size(1)

        # 初始化隐藏状态
        hidden = self._init_hidden(batch_size)
        # 应用嵌入层
        embedding = self.embedding(input)

        # 使用 pack_padded_sequence 包装嵌入后的序列，以便处理变长序列
        gru_input = pack_padded_sequence(embedding, seq_lengths)

        # 通过 GRU 层
        output, hidden = self.gru(gru_input, hidden)
        # 处理双向 GRU 的隐藏状态
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        # 应用全连接层，得到最终输出
        fc_output = self.fc(hidden_cat)
        return fc_output 

# 定义一个函数，用于计算时间经过
def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return f'{int(m)}m {s:.2f}s' 

# 定义训练模型的函数
def trainModel():
    total_loss = 0  # 初始化总损失
    # 遍历训练集数据加载器
    for i, (names, countries) in enumerate(trainloader, 1):
        # 将名字和国家转换为张量
        inputs, seq_lengths, target = make_tensors(names, countries)
        # 前向传播，得到模型输出
        output = classifier(inputs, seq_lengths)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累加损失
        # 每 10 个批次打印一次训练信息
        if i % 10 == 0 :
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i}*{len(inputs)}/{len(trainset)}]', end='')
            print(f'Loss={total_loss / (i * len(inputs))}', end='')
    return total_loss

# 定义测试模型的函数
def testModel():
    correct = 0  # 正确预测的数量
    total = len(testset)  # 测试集的总大小
    print("evaluating trained model ...")
    with torch.no_grad():  # 禁用梯度计算
        # 遍历测试集数据加载器
        for i, (names, countries) in enumerate(testloader, 1):
            # 将名字和国家转换为张量
            inputs, seq_lengths, target = make_tensors(names, countries)
            # 前向传播，得到模型输出
            output = classifier(inputs, seq_lengths)
            # 获取预测结果
            pred = output.max(dim=1, keepdim=True)[1]
            # 统计正确预测的数量
            correct += pred.eq(target.view_as(pred)).sum().item()

        # 计算准确率并打印测试结果
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accurace {correct}/{total} {percent}%')

    return correct / total

# 主程序入口
if __name__ == '__main__':
    # 创建 RNN 分类器实例
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    # 检查是否有 GPU 可用，并将模型移动到相应的设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()  # 记录训练开始时间
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []  # 用于存储每个轮次的准确率

    # 训练循环
    for epoch in range(1, N_EPOCHS + 1):
        # 训练模型
        trainModel()
        # 测试模型并获取准确率
        acc = testModel()
        # 记录准确率
        acc_list.append(acc)

    # 绘制准确率曲线
    epoch = np.arange(1, len(acc_list) + 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()