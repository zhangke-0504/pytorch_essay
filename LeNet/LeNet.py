import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_iter = DataLoader(train_data, batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size, shuffle=False)


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # input_channels, output_channels, kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 4 * 4, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
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
        x = self.linear3(x)
        return x
    
    
# 定义训练函数
def train(model_path, train_loader, test_loader, is_train=True):
    # 定义GPU计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = LeNet().to(device)
    # 定义训练超参数
    num_epochs = 20
    lr = 0.01
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 训练模型
    if is_train:
        # 将loss.item()保存进一个列表方便绘图
        loss_list = []
        for epoch in range(num_epochs): 
            total_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                # 将数据移到GPU上
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 计算总损失
                total_loss += loss.item()
            # 保存loss
            loss_list.append(total_loss)

            # 打印每个epoch的损失
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        # 保存epoch-loss曲线为jpg
        # 横坐标为epoch, 纵坐标为loss
        plt.figure(figsize=(10, 6), dpi=150)
        line = plt.plot(range(1, num_epochs+1), loss_list, 
                    color='#E41A1C', linewidth=2, 
                    marker='o', markersize=8,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    label='Training Loss')

        plt.xticks(range(1, num_epochs+1), fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(bottom=0)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.title('Training Loss Curve\nFashionMNIST - LeNet', 
                fontsize=16, pad=20)

        # 添加数据标签
        for i, v in enumerate(loss_list):
            plt.text(i+1, v+0.02, f"{v:.2f}", 
                    ha='center', va='bottom',
                    fontsize=9, color='#377EB8')

        plt.legend(loc='upper right', fontsize=12)
        plt.tight_layout()
        plt.savefig("epoch-loss.jpg", dpi=300, bbox_inches='tight')
        plt.close()  # 防止内存泄漏

        # 保存模型
        torch.save(model.state_dict(), model_path)
    else:
        # 加载模型
        model.load_state_dict(torch.load(model_path, weights_only=True))
        # 测试模型
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f"Accuracy of the model on the test images: {100 * correct / total}%")


if __name__ == '__main__':
    # 设置模型保存及加载路径
    model_path = 'last.pt'
    # 调用train函数
    train(model_path, train_iter, test_iter, is_train=False)

        
