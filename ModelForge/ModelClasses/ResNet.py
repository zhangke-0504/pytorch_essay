import torch
import torch.nn as nn


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 自动判断是否需要1x1卷积
        self.use_1x1conv = (in_channels != out_channels) or (stride != 1)
        if self.use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        out = torch.add(x, y) # 残差连接
        out = self.relu(out)
        return out


# 定义块级单位批量连接函数
def make_layers(block, input_channels, output_channels, num_blocks, stride=1):
    layers = []
     # 首块带下采样
    layers.append(block(input_channels, output_channels, stride, use_1x1conv=True))
     # 后续块stride=1
    for _ in range(1, num_blocks):
        layers.append(block(output_channels, output_channels, stride=1))
    return nn.Sequential(*layers)
    
# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            make_layers(ResidualBlock, 64, 64, 2, stride=1),
            make_layers(ResidualBlock, 64, 128, 2, stride=2),
            make_layers(ResidualBlock, 128, 256, 2, stride=2),
            make_layers(ResidualBlock, 256, 512, 2, stride=2),
        )
        self.b3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # 之所以将最后一层全连接单独定义，是因为如果是需要加载成预训练模型的场景下
        # 统一排除预训练模型中的output层，避免标签类别不一致导致的形状错误
        self.output = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.output(x)
        return x
        
        

