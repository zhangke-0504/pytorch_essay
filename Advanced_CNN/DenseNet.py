import torch
import torch.nn as nn
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

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(self._make_layer(in_channels, growth_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        features = [x]  # 存储所有层级联结果
        for layer in self.block:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)  # 返回全部特征级联
 
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)
 
class DenseNet201(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet201, self).__init__()
        self.features = nn.Sequential(
            # 输入(1, 28, 28)
            # 输出计算为
            # (28+2x3-7)/2 + 1 = 14  (向下取整)
            # 所以输出是(64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            # 输入(64, 14, 14)
            # 对每个通道的特征图进行标准化, 强制数据分布符合均值为0、方差为1的正态分布
            nn.BatchNorm2d(64),
            # 激活函数
            nn.ReLU(inplace=True),
            # 输入(64, 14, 14)
            # 输出计算为
            # (14+2x1-3)/2 + 1 = 7  (向下取整)
            # 输出(64, 7, 7)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第一个DenseBlock(6层)
            # 每层增长率为32,通道变化过程如下：
            # 输入通道64，新增通道32，累计通道数64+32=96
            # 输入通道96， 新增通道32， 累计通道数96+32=128
            # 输入通道128，新增通道32，累计通道数128+32=160
            # 输入通道160，新增通道32，累计通道数160+32=192
            # 输入通道192，新增通道32，累计通道数192+32=224
            # 输入通道224，新增通道32，累计通道数224+32=256
            # 最终输出(256, 7, 7)
            DenseBlock(64, 32, 6),
            # 第一个TransitionLayer层
            # 经过一个1x1的卷积层，
            # 输入(256, 7, 7)
            # 输出(128, 7, 7)
            # 经过一个平均池化层，
            # 输入(128, 7, 7)
            # 输出(128, 3, 3)
            TransitionLayer(64 + 6*32, 128),
            # 由于mnist数据集图片太小，所以后续的网络层就不用了
            # DenseBlock(128, 32, 12),
            # TransitionLayer(128 + 12*32, 256),
            # DenseBlock(256, 32, 24),
            # TransitionLayer(256 + 24*32, 512),
            # DenseBlock(512, 32, 16),
        )
        self.classifier = nn.Linear(128*3*3, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
 
# 实例化 DenseNet-201 模型
model = DenseNet201(num_classes=10)

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