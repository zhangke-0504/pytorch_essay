import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from collections import defaultdict

class FashionMNIST:
    def __init__(self, root='./Dataset', batch_size=64, val_ratio=0.2, 
                 transform=None, seed=42):
        """
        参数说明：
        root: 数据集存储路径
        batch_size: 数据加载批次大小
        val_ratio: 验证集划分比例
        transform: 自定义数据预处理流程
        seed: 随机划分种子(保证可重复性)
        """
        # 设置默认预处理流程
        self.transform = transform or transforms.Compose([
            # 想跑更大的网络就把这两行加上，不然(1, 28, 28)跑不了深层网络只能跑LeNet
            transforms.Resize((96, 96)),
            # transforms.Grayscale(num_output_channels=3),  # 将灰度图转为三通道伪彩图

            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])
        
       # 加载原始数据集
        full_train = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=self.transform)
        self.test_data = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=self.transform)
        
        # 分层划分逻辑
        if val_ratio > 0:
            # 按类别收集索引
            class_indices = defaultdict(list)
            for idx, (_, label) in enumerate(full_train):
                class_indices[label].append(idx)
            
            # 分层划分
            train_subsets = []
            val_subsets = []
            generator = torch.Generator().manual_seed(seed)
            
            for cls, indices in class_indices.items():
                # 随机打乱类内索引
                perm = torch.randperm(len(indices), generator=generator).tolist()
                split = int(len(indices) * (1 - val_ratio))
                
                # 添加子集
                train_subsets.append(Subset(full_train, [indices[i] for i in perm[:split]]))
                val_subsets.append(Subset(full_train, [indices[i] for i in perm[split:]]))
            
            # 合并所有类别的子集
            self.train_subset = ConcatDataset(train_subsets)
            self.val_subset = ConcatDataset(val_subsets)
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                self.train_subset, batch_size, shuffle=True)
            self.val_loader = DataLoader(
                self.val_subset, batch_size, shuffle=False)
        else:
            self.val_loader = None
            self.train_loader = DataLoader(
                full_train, batch_size, shuffle=True)
            
        # 测试集加载器
        self.test_loader = DataLoader(
                self.test_data, batch_size, shuffle=False)

    def get_loaders(self):
        """返回所有数据加载器"""
        return {
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'test_loader': self.test_loader
        }