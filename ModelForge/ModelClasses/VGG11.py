import torch
import torch.nn as nn

# 定义VGG块
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
    

# 定义主模型
class VGG11(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super(VGG11, self).__init__()
        # 定义卷积块序列 
        self.features = nn.Sequential(
            VGGBlock(3, 64, num_convs=1),        # Block1: 输入3通道，输出64通道
            VGGBlock(64, 128, num_convs=1),     # Block2
            VGGBlock(128, 256, num_convs=2),    # Block3
            VGGBlock(256, 512, num_convs=2),    # Block4
            VGGBlock(512, 512, num_convs=2)     # Block5
        )
        # 定义全连接层 
        self.Dense = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.output = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)        # 输入尺寸：(N, 3, 224, 224)
        x = torch.flatten(x, 1)     # 展平后尺寸：(N, 512 * 7 * 7)
        x = self.Dense(x)
        x = self.output(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)