import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from collections import defaultdict

class CIFAR100:
    def __init__(self, root='./Dataset', batch_size=64, val_ratio=0.1,
                 transform=None, seed=42):
        """
        CIFAR-100数据加载器（支持分层验证集划分）
        参数说明：
        root: 数据集存储路径
        batch_size: 数据加载批次大小
        val_ratio: 验证集划分比例（默认0.1即10%）
        transform: 自定义数据预处理流程（默认含标准化）
        seed: 随机划分种子
        """
        # 关键修改点1：使用CIFAR-100专用归一化参数（参考论文建议）
        self.transform = transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 保持相同数据增强策略
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR100均值
                                 std=[0.2675, 0.2565, 0.2761])  # CIFAR100方差[3](@ref)
        ])
        
        # 测试集使用非增强预处理（保持与训练集相同的归一化参数）
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
        
        # 关键修改点2：加载CIFAR100数据集
        full_train = datasets.CIFAR100(
            root=root, train=True, download=True, transform=self.transform)
        self.test_data = datasets.CIFAR100(
            root=root, train=False, download=True, transform=test_transform)
        
        # 保持分层划分逻辑（100个类别处理方式相同）
        if val_ratio > 0:
            class_indices = defaultdict(list)
            # 关键修改点3：CIFAR100返回的标签是细粒度标签（fine_labels）
            for idx, (_, label) in enumerate(full_train):
                class_indices[label].append(idx)
            
            train_subsets = []
            val_subsets = []
            generator = torch.Generator().manual_seed(seed)
            
            # 分层抽样逻辑保持不变（每个类独立划分）
            for cls, indices in class_indices.items():
                perm = torch.randperm(len(indices), generator=generator).tolist()
                split = int(len(indices) * (1 - val_ratio))
                train_subsets.append(Subset(full_train, [indices[i] for i in perm[:split]]))
                val_subsets.append(Subset(full_train, [indices[i] for i in perm[split:]]))
            
            self.train_subset = ConcatDataset(train_subsets)
            self.val_subset = ConcatDataset(val_subsets)
            
            # 数据加载参数保持相同配置
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
        
        self.test_loader = DataLoader(
            self.test_data, batch_size, shuffle=False,
            pin_memory=True)

    def get_loaders(self):
        """返回数据加载器字典（接口与CIFAR10版本一致）"""
        return {
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'test_loader': self.test_loader
        }