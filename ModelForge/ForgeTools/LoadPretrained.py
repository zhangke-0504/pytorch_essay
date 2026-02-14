import torch

def load_pretrained(model, pretrained_path, device='cuda'):
    """
    加载CIFAR100预训练权重到CIFAR10模型（保留除输出层外的所有权重）
    
    参数说明：
    model: 新初始化的CIFAR10模型实例（需与预训练模型结构一致）
    pretrained_path: 预训练权重路径（如'PretrainedModels/CIFAR100-ResNet.pt'）
    device: 目标计算设备
    """
    # 加载预训练权重文件
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=True)
    
    # 解包权重字典（兼容不同保存方式）
    if 'state_dict' in checkpoint:  # 检查是否包含完整检查点
        pretrained_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:  # 兼容多层级保存结构
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint
    
    # 获取当前模型的状态字典
    model_dict = model.state_dict()
    
    # 过滤匹配的权重（排除输出层）
    # 假设输出层名为'fc'（ResNet常见命名，需根据实际模型调整）
    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict 
        and not k.startswith('output')  # 排除全连接层
        and v.shape == model_dict[k].shape  # 形状校验
    }
    
    # 更新模型字典并加载
    model_dict.update(filtered_dict)
    # load_status = model.load_state_dict(model_dict, strict=False)
    
    # 打印加载状态
    # print(f"成功加载 {len(filtered_dict)}/{len(pretrained_dict)} 层参数")
    # print("缺失参数:", load_status.missing_keys)  # 应包含fc层参数
    # print("冗余参数:", load_status.unexpected_keys)  # 应为空列表
    
    return model