import torch
import torch.nn as nn

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.Dense1 = nn.Linear(256 * 5 * 5, 4096)
        self.Dense2 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, num_classes)

    def forward(self, x):
        # 输入图像尺寸为（3, 224, 224）
        # 经过第一层卷积变为（96， 54， 54）
        x = self.relu(self.conv1(x))
        # 经过第一层池化变为（96, 26, 26）
        x = self.pool1(x)
        # 经过第二层卷积(256, 26, 26)
        x = self.relu(self.conv2(x))
        # 经过第二层池化(256, 12, 12)
        x = self.pool2(x)
        # 经过第三层卷积(384, 12, 12)
        x = self.relu(self.conv3(x))
        # 经过第四层卷积(384, 12, 12)
        x = self.relu(self.conv4(x))
        # 经过第五层卷积（256, 12, 12）
        x = self.relu(self.conv5(x)) # 第五层结构和第四层一致
        # 经过第三层池化（256, 5, 5）
        x = self.pool2(x) # 第三层池化结构与第二层一致
        # 经过全连接层
        x = x.view(x.size(0), -1)
        x = self.Dropout(self.relu(self.Dense1(x)))
        x = self.Dropout(self.relu(self.Dense2(x)))
        x = self.output(x)
        return x