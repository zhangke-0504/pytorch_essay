import torch
import numpy as np

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

# one_hot_lookup = [
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 0, 0, 1]
# ]
one_hot_lookup = np.eye(4)
print('one_hot_lookup:', one_hot_lookup)

x_one_hot = [one_hot_lookup[x] for x in x_data]

batch_size = 1
input_size = 4
hidden_size = 4
num_layers = 2

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
print('inputs:', inputs)
labels = torch.LongTensor(y_data)
print('labels:', labels)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers)
    def forward(self, input):
        hidden = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size
        )
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = net(inputs)
    print(output.shape)
    print(labels.shape)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    _, idx = output.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted:', ''.join([idx2char[i] for i in idx]), end='')
    print(', Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1,100, loss.item()))