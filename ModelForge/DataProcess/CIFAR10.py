import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from collections import defaultdict

class CIFAR10:
    def __init__(self, root='./Dataset', batch_size=64, val_ratio=0.1,
                 transform=None, seed=42):
        """
        CIFAR-10数据加载器，支持分层验证集划分
        参数说明：
        root: 数据集存储路径
        batch_size: 数据加载批次大小
        val_ratio: 验证集划分比例（默认0.1即10%）
        transform: 自定义数据预处理流程（默认含标准化）
        seed: 随机划分种子
        num_workers: 数据加载线程数
        """
        # 默认预处理流程（参考CIFAR论文建议）
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 数据增强
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR10专用均值方差
                                 (0.2470, 0.2435, 0.2616))
        ])
        
        # 测试集使用非增强预处理
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        
        # 加载原始数据集
        full_train = datasets.CIFAR10(
            root=root, train=True, download=True, transform=self.transform)
        self.test_data = datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform)
        
        # 分层划分逻辑（保持类别分布一致性）
        if val_ratio > 0:
            class_indices = defaultdict(list)
            for idx, (_, label) in enumerate(full_train):
                class_indices[label].append(idx)
            
            train_subsets = []
            val_subsets = []
            generator = torch.Generator().manual_seed(seed)
            
            # 按类别划分训练/验证集
            for cls, indices in class_indices.items():
                perm = torch.randperm(len(indices), generator=generator).tolist()
                split = int(len(indices) * (1 - val_ratio))
                train_subsets.append(Subset(full_train, [indices[i] for i in perm[:split]]))
                val_subsets.append(Subset(full_train, [indices[i] for i in perm[split:]]))
            
            self.train_subset = ConcatDataset(train_subsets)
            self.val_subset = ConcatDataset(val_subsets)
            
            # 创建数据加载器（pin_memory加速GPU传输）
            self.train_loader = DataLoader(
                self.train_subset, batch_size, shuffle=True, 
                pin_memory=True)
            self.val_loader = DataLoader(
                self.val_subset, batch_size, shuffle=False, 
                pin_memory=True)
        else:
            self.val_loader = None
            self.train_loader = DataLoader(
                full_train, batch_size, shuffle=True, 
                pin_memory=True)
        
        # 测试集加载器（固定顺序）
        self.test_loader = DataLoader(
            self.test_data, batch_size, shuffle=False, 
            pin_memory=True)

    def get_loaders(self):
        """返回数据加载器字典"""
        return {
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'test_loader': self.test_loader
        }