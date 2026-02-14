import torch

'''
以天气数据举例
seq_len代表三天的数据
input_size代表4个天气特征
'''
batch_size = 1
seq_len = 3  # 序列长度
input_size = 4
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, input_size)
dataset = torch.randn(seq_len, batch_size, input_size)

# (batch, hidden_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print("=" * 20, idx, "=" * 20)
    print("Input size:", input.shape)

    hidden = cell(input, hidden)

    print("outputs_size:", hidden.shape)
    print(hidden)