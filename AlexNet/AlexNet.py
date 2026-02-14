import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 加载数据
# 将图像resize成AlexNet的输入大小（3, 224, 224）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整尺寸[7](@ref)
    transforms.Grayscale(num_output_channels=3),  # 将灰度图转为三通道伪彩图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


batch_size = 128
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)



# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self):
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
        self.Dense3 = nn.Linear(4096, 10)

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
        x = self.Dense3(x)
        return x
    
    
# 定义训练函数
def train(train_params):
    # 定义GPU计算
    device = train_params['device']
    # 实例化模型
    model = train_params['model']
    # 定义训练超参数
    num_epochs = train_params['num_epochs']
    # 定义损失函数和优化器
    criterion = train_params['criterion']
    optimizer = train_params['optimizer']

    # 函数调用状态-->训练or测试
    is_train = train_params['is_train']
    # 模型保存&加载路径
    model_path = train_params['model_path']
    # 训练集
    train_loader = train_params['train_loader']
    # 测试集
    test_loader = train_params['test_loader']

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
        plt.title('Training Loss Curve\nFashionMNIST - AlexNet', 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    train_params = {
        'model_path': 'last.pt',
        'train_loader': train_loader,
        'test_loader': test_loader,
        'is_train': False,
        'device': device,
        'model': model,
        'num_epochs': 10,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    }
    # 调用train函数
    train(train_params)

        
