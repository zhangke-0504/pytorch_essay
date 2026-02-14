import torch
import torch.nn as nn

# 定义NiN块
class NiNBlock(nn.Module):
    def __init__ (self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            # 一个卷积层后跟两个全连接层
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            # 卷积核为1的卷积层起到变维的作用，效果等同于线性层，
            # 所以这里用kernel=1的卷积层而不用nn.Linear
            # nn.Linear还需要展平再连接，很不优雅
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 关键修正1
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), # 关键修正2
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    
# 定义NiN网络
class NiNNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(NiNNet, self).__init__()
        self.net = nn.Sequential(
            NiNBlock(1, 96, 11, 4, 0),
            nn.MaxPool2d(3, 2),
            NiNBlock(96, 256, 5, 1, 1),
            nn.MaxPool2d(3, 2),
            NiNBlock(256, 384, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            NiNBlock(384, num_classes, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 将四维转二维，方便计算交叉熵损失
            nn.Flatten()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)