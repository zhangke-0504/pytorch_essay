import torch
import torch.nn as nn

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input_channels, output_channels, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 4 * 4, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 卷积层1 + 激活 + 池化
        # 图片是（1， 28， 28）
        # 卷积层1：输出（6， 24， 24）
        # 池化层：输出（6， 12， 12）
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        # 卷积层2 + 激活 + 池化
        # 输入（6， 12， 12）
        # 卷积层2：输出（16， 8， 8）
        # 池化层：输出（16， 4， 4）
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool(x)

        # 三层全连接
        # 因为损失函数是用的交叉熵
        # 所以最后一层全连接不需要激活函数
        x = x.view(-1, 16 * 4 * 4)
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x