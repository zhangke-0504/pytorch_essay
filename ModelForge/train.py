import torch
import os
from DataProcess import FashionMNIST, CIFAR10, CIFAR100
from ForgeTools import LoadYaml, EarlyStopping, TrainingVisualizer, LoadPretrained
from ModelClasses import VGG11, DenseNet, LeNet, AlexNet, NiNNet, GoogLeNet, ResNet,\
DenseNet

def DataLoader(config_data):
    dataset2data_params = {
        'FashionMNIST': FashionMNIST.FashionMNIST(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders(),
        'CIFAR10': CIFAR10.CIFAR10(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders(),
        'CIFAR100': CIFAR100.CIFAR100(batch_size=config_data['batch_size'], val_ratio=config_data['val_ratio']).get_loaders()
    }
    data_params = dataset2data_params[config_data['dataset']]
    return data_params

# 加载训练配置项
class Load_TrainConfig:
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
        if self.config_data['pretrained']['is_pretrained']:
            self.model = LoadPretrained.load_pretrained(
                    self.model_class2model[self.model_class], 
                    pretrained_path=self.config_data['pretrained']['path'],
                    device=self.device
                )
        else:
            self.model = self.model_class2model[self.model_class]
        return self.model
        
        
    def load_criterion(self):
        """加载损失函数"""
        if self.criterion == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()
            return self.criterion
        else:
            raise ValueError('Invalid criterion')
        
    def load_optimizer(self):
        """加载优化器"""
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            return self.optimizer
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
            return self.optimizer
        else:
            raise ValueError('Invalid optimizer')
    
    def load_hyperparameter(self):
        """加载超参数"""
        hyperparameter_params = {
            'num_epochs': self.num_epochs,
            'device': self.device,
            'model_save_path': self.model_save_path
        }
        return hyperparameter_params
    
# 训练函数
def train(model, data_params, criterion, optimizer, hyperparameter_params, config_data):
    """训练函数"""
    # 获取数据
    train_loader = data_params['train_loader']
    val_loader = data_params['val_loader'] if data_params['val_loader'] != None else data_params['test_loader']
    print('len(train_loader.dataset):',len(train_loader.dataset))
    print('len(val_loader.dataset):',len(val_loader.dataset))
    # test_loader = data_params['test_loader']
    # 获取超参数
    num_epochs = hyperparameter_params['num_epochs']
    device = hyperparameter_params['device']
    model_save_path = hyperparameter_params['model_save_path']
    # 获取早停参数
    early_stopping_params = config_data['early_stopping']
    early_stopping = EarlyStopping.EarlyStopping(
        patience=early_stopping_params['patience'],
        delta=early_stopping_params['delta'],
        save_path=early_stopping_params['save_path'],
        verbose=early_stopping_params['verbose']
    )
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=0, # 禁用内置等待机制
        threshold=1e-3,  # 降低触发阈值
        threshold_mode='abs',  # 使用绝对变化判断
        min_lr=1e-6 # 防止学习率无限衰减
    )
    # 初始化可视化对象
    visualizer = TrainingVisualizer.TrainingVisualizer()
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        # 利用训练集训练模型并计算训练集损失
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前馈
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            # 累计训练准确率
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            # 显式释放缓存，避免显存紧张的情况下触发OOM
            del outputs, loss
            torch.cuda.empty_cache() 
        # 计算训练指标
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        # 利用验证集计算验证集损失
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                # 累计验证准确率
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                # 显式释放缓存，避免显存紧张的情况下触发OOM
                del outputs, loss
                torch.cuda.empty_cache() 
        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        # 打印迭代轮数\n 训练集验证集损失\n 训练集验证集准确率
        print(f'Epoch [{epoch}/{num_epochs}]\n')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n')
        print(f'Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n')
        # 更新可视化数据
        visualizer.update(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )
        # 保存一版最新的模型last.pt
        torch.save(model.state_dict(), os.path.join(model_save_path, 'last.pt'))
        # 通过早停类判断是否停止训练并保存最佳模型
        should_stop, should_scheduler_step = early_stopping(
            train_loss=avg_train_loss, 
            val_loss=avg_val_loss,
            model=model,
            epoch=epoch
        )
        if should_scheduler_step:
            scheduler.step(avg_val_loss)
            # 打印更新后的学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate Updated to: {current_lr}")
        if should_stop:
            break

        

if __name__ == '__main__':
    # 定义配置文件路径
    config_file = './Configs/ResNet.yaml'
    # 实例化配置读取类
    TrainConfig = Load_TrainConfig(config_file)
    # 加载配置文件内容，方便全局使用
    config_data = TrainConfig.load_yaml(config_file)
    # 加载数据
    data_params = DataLoader(config_data)
    # 加载模型
    model = TrainConfig.load_model()
    # 加载损失函数
    criterion = TrainConfig.load_criterion()
    # 加载优化器
    optimizer = TrainConfig.load_optimizer()
    # 加载超参数
    hyperparameter_params = TrainConfig.load_hyperparameter()
    # 开始训练
    train(model, data_params, criterion, optimizer, hyperparameter_params, config_data)

    

