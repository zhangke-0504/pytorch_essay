import torch
import torch.nn as nn

# 定义Dense块
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_convs):
        super(DenseBlock, self).__init__()
        """
        稠密块只改变通道数，不改变高和宽
        """
        net = []
        for i in range(num_convs):
            net.append(self._conv_block(in_channels, growth_rate))
            in_channels += growth_rate
        self.net = nn.Sequential(*net)
    
    def _conv_block(self, in_channels, growth_rate):
        x = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 
                      growth_rate, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False)
        )
        return x

    def forward(self, x):
        """
        不同于ResNet将输出与输入相加
        稠密块将输入与输出在通道维上连结
        """
        for block in self.net:
            out = block(x)
            x = torch.cat((x, out), dim=1)
        return x
    
# 定义过渡层
class transition_block(nn.Module):
    """
    由于每个稠密块都会带来通道数的增加
    过渡层用来空值模型复杂度
    """
    def __init__(self, in_channels, out_channels):
        super(transition_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_block(x)
    

# 定义DenseNet
class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        net = []
        in_channels = 64
        growth_rate = 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            net.append(DenseBlock(in_channels, growth_rate, num_convs))
            in_channels += num_convs * growth_rate
            # 在稠密块之间加入通道数减半的过渡层
            if i != len(num_convs_in_dense_blocks) - 1:
                net.append(transition_block(in_channels, in_channels // 2))
                in_channels = in_channels // 2
        self.b2 = nn.Sequential(*net)
        self.b3 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x