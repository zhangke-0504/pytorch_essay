import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    # 用均值和标准差对图像进行归一化
    transforms.Normalize((0.1307,), (0.3081, ))
])

train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               download=True, 
                               transform=transform)
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_dataset = datasets.MNIST(root='./data', 
                              train=False, 
                              download=True, 
                              transform=transform)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels,
                                     kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 
                                     kernel_size=3, padding=1)
        
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
    
'''
Bottleneck残差块（ResNet-50+）
通过1x1卷积实现"降维-特征提取-升维"的三层结构，
显著减少深层网络的计算量。
当输入输出维度不一致时，
采用1x1卷积调整跳跃连接通道数。
'''
class BottleneckBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.expansion = 4  # 输出通道扩展倍数
        mid_channels = out_channels // self.expansion
        
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(mid_channels)
        self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, 
                                    kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(mid_channels)
        self.conv3 = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        
        # 通道数不匹配时使用1x1卷积调整
        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + residual)

'''
预激活残差块（ResNet-v2）
将BN和ReLU置于卷积操作之前，
形成"BN-ReLU-Conv"的预激活结构，
改善梯度流动。
此结构在超深网络（如ResNet-152）中表现更优。
'''  
class PreActBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        return x + residual
    
'''
通道注意力残差块（SE-ResNet）
引入通道注意力机制（Squeeze-and-Excitation），
通过全局平均池化和全连接层动态学习各通道权重，
增强重要特征通道的响应。
'''
class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)

'''
宽残差网络（WRN）
通过增加每层卷积核数量（宽度）而非深度来提升性能，
配合Dropout层防止过拟合。
适合小尺寸图像（如MNIST）的特征提取。
'''   
class WideResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 
                                    kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.dropout = torch.nn.Dropout2d(0.3)
        
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
    

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        # self.rblock1 = ResidualBlock(16)
        # self.rblock2 = ResidualBlock(32)
        # self.bottleneck1 = BottleneckBlock(16, 16)
        # self.bottleneck2 = BottleneckBlock(32, 32)
        # self.preactblock1 = PreActBlock(16)
        # self.preactblock2 = PreActBlock(32)
        # self.seresnet1 = SEResidualBlock(16)
        # self.seresnet2 = SEResidualBlock(32)
        self.wrn1 = WideResBlock(16, 16)
        self.wrn2 = WideResBlock(32, 32)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        # x = self.rblock1(x)
        # x = self.bottleneck1(x)
        # x = self.preactblock1(x)
        # x = self.seresnet1(x)
        x = self.wrn1(x)
        x = self.mp(F.relu(self.conv2(x)))
        # x = self.rblock2(x)
        # x = self.bottleneck2(x)
        # x = self.preactblock2(x)
        # x = self.seresnet2(x)
        x = self.wrn2(x)
        x = x.view(in_size, -1)  # 展平
        x = self.fc(x)
        return x
    

model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 定义交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        # send the inputs and targets at every step to the GPU
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, 
            running_loss/2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy on test set: %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):  # loop over the dataset multiple times
        train(epoch)
        test()
