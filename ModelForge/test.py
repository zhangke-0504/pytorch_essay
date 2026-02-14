import torch
import os
from DataProcess import FashionMNIST, CIFAR10, CIFAR100
from ForgeTools import LoadYaml
from ModelClasses import VGG11, DenseNet, LeNet, AlexNet, NiNNet, GoogLeNet, ResNet,\
DenseNet

# 训练&验证&测试数据加载函数
def DataLoader(config_data):
    dataset2data_params = {
        'FashionMNIST': FashionMNIST.FashionMNIST(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders(),
        'CIFAR10': CIFAR10.CIFAR10(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders(),
        'CIFAR100': CIFAR100.CIFAR100(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders()
    }
    data_params = dataset2data_params[config_data['dataset']]
    return data_params

# 加载测试配置项
class Load_TextConfig:
    def __init__(self, config_file):
        self.config_data = self.load_yaml(config_file)
        self.model_save_path = self.config_data['model_save_path']
        self.num_classes = self.config_data['num_classes']
        self.device = self.config_data['device']
        self.model_class = self.config_data['model_class']
        self.model_class2model = {
            'VGG11': VGG11.VGG11(self.num_classes).to(self.device),
            'LeNet': LeNet.LeNet(self.num_classes).to(self.device),
            'AlexNet': AlexNet.AlexNet(self.num_classes).to(self.device),
            'NiNNet': NiNNet.NiNNet(self.num_classes).to(self.device),
            'GoogLeNet': GoogLeNet.GoogLeNet(self.num_classes).to(self.device),
            'ResNet': ResNet.ResNet(self.num_classes).to(self.device),
            'DenseNet': DenseNet.DenseNet(self.num_classes).to(self.device),
        }
        self.model = self.load_model()
        self.num_epochs = self.config_data['num_epochs']
        self.criterion = self.config_data['criterion']
        self.lr = self.config_data['lr']
        self.optimizer = self.config_data['optimizer']
    
    def load_yaml(self, config_file):
        """加载配置文件"""
        config_data = LoadYaml.load_yaml(config_file)
        return config_data

    def load_model(self):
        """加载模型"""
        self.model = self.model_class2model[self.model_class]
        return self.model
        

def test(model, test_loader, model_path, device):
    """测试函数"""
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
    # 定义配置文件路径
    config_file = './Configs/ResNet.yaml'
    # 实例化配置读取类
    TextConfig = Load_TextConfig(config_file)
    # 加载配置文件内容，方便全局使用
    config_data = TextConfig.load_yaml(config_file)
    # 设置GPU测试还是CPU测试
    device = config_data['device']
    # 设置模型加载路径
    model_path = os.path.join(config_data['model_save_path'], 'best.pt')
    # 加载数据
    data_params = DataLoader(config_data)
    # 取出测试集数据
    test_loader = data_params['test_loader']
    # 加载模型
    model = TextConfig.load_model()
    # 开始测试
    test(model, test_loader, model_path, device)