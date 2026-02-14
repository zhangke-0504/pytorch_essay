from DataProcess import Yelan
from ForgeTools import DataAugmentation, Anchors, VisualizeBoxes, DetectionTrainingVisualizer
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
from ModelClasses.SSD import SSD, SSDLoss
import pickle

# 自定义数据集类
class DetectionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 验证关键数据是否存在
        if not all(key in item for key in ['img', 'anchors', 'anchor_labels', 'anchor_offsets', 'positive_mask']):
            print(f"警告：索引 {idx} 的数据项缺失关键字段")
            # 尝试获取默认项
            idx = idx - 1 if idx > 0 else idx + 1
            return self.__getitem__(idx)
        # 转换为Tensor
        image = torch.from_numpy(item['img']).permute(2, 0, 1).float() / 255.0  # [C, H, W]
        anchors = torch.from_numpy(item['anchors']).float()
        labels = torch.from_numpy(item['anchor_labels']).long()
        offsets = torch.from_numpy(item['anchor_offsets']).float()
        pos_mask = torch.from_numpy(item['positive_mask']).bool()
        
        return {
            'image': image,        # 图像 [3,640,640]
            'anchors': anchors,    # 锚框坐标 [N,4]
            'labels': labels,      # 分类标签 [N]
            'offsets': offsets,    # 回归偏移量 [N,4]
            'pos_mask': pos_mask   # 正样本掩码 [N]
        }



def stratified_split_by_objects(augmented_data, test_size=0.1, val_size=0.2, min_samples=1, seed=42):
    """
    基于目标实例类别的分层分割（优化版）
    参数：
        test_size: 测试集目标实例占比
        val_size: 验证集在非测试集中的目标实例占比
        min_samples: 每个类在测试/验证集中的最小目标数
    """
    np.random.seed(seed)
    
    # 步骤1：构建图像组（原始图像与其增强版本）
    source_groups = defaultdict(list)
    for obj in augmented_data:
        source_groups[obj['source_image_id']].append(obj)
    
    # 步骤2：统计每个类别的图像分布

    class_info = {}
    for source_id, objs in source_groups.items():
        cls_counter = {}
        for obj in objs:
            for box in obj['Boxes']:
                cls = box['class_name']
                cls_counter[cls] = cls_counter.get(cls, 0) + 1
                
        for cls, cnt in cls_counter.items():
            if cls not in class_info:
                class_info[cls] = {'sources': [], 'count': 0}
            class_info[cls]['sources'].append((source_id, cnt))
            class_info[cls]['count'] += cnt

    # 步骤3：初始化分配记录
    split_candidates = defaultdict(set)
    for cls in class_info:
        np.random.shuffle(class_info[cls]['sources'])  # 随机打乱顺序
        
        # 计算所需数量
        total = class_info[cls]['count']
        required_test = max(min_samples, int(total * test_size))
        required_val = max(min_samples, int(total * val_size))
        
        # 分配测试集
        test_count = 0
        for source_id, cnt in class_info[cls]['sources']:
            if test_count >= required_test:
                break
            split_candidates[source_id].add('test')
            test_count += cnt
        
        # 分配验证集
        val_count = 0
        for source_id, cnt in class_info[cls]['sources']:
            if source_id in split_candidates:  # 跳过已分配的
                continue
            if val_count >= required_val:
                break
            split_candidates[source_id].add('val')
            val_count += cnt

    # 步骤4：确定最终分配（优先级：test > val > train）
    final_split = {}
    for source_id in source_groups:
        if 'test' in split_candidates[source_id]:
            final_split[source_id] = 'test'
        elif 'val' in split_candidates[source_id]:
            final_split[source_id] = 'val'
        else:
            final_split[source_id] = 'train'

    # 步骤5：收集结果并统计
    # split_plan = defaultdict(lambda: defaultdict(int))
    split_plan = {
        'train': {},
        'val': {},
        'test': {}
    }
    train_data, val_data, test_data = [], [], []

    for split_name in ['train', 'val', 'test']:
        split_plan[split_name] = {}
        
    for source_id, split in final_split.items():
        group = source_groups[source_id]
        # 记录分组数据
        if split == 'train':
            train_data.extend(group)
        elif split == 'val':
            val_data.extend(group)
        else:
            test_data.extend(group)
        
        # 统计类别分布
        for obj in group:
            for box in obj['Boxes']:
                cls = box['class_name']
                split_plan[split][cls] = split_plan[split].get(cls, 0) + 1
    
    # 将count转换为数字
    for cls in class_info:
        class_info[cls]['count'] = int(class_info[cls]['count'])
        
    return train_data, val_data, test_data, split_plan, class_info

# 计算精确率
def calculate_accuracy(model, dataloader, device, num_classes):
    model.eval()
    total_pos_correct = 0
    total_neg_correct = 0
    total_pos = 0
    total_neg = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            batch_labels = torch.stack(batch['labels'], dim=0).to(device)
            batch_pos_mask = torch.stack(batch['pos_mask'], dim=0).to(device)
            
            # 前向传播
            _, conf_preds = model(images)
            pred_classes = torch.argmax(conf_preds, dim=2)
            
            # 分离正负样本
            pos_mask = batch_pos_mask
            neg_mask = ~pos_mask
            
            # 正样本统计
            pos_preds = pred_classes[pos_mask]
            pos_labels = batch_labels[pos_mask]
            pos_correct = (pos_preds == pos_labels).sum().item()
            total_pos_correct += pos_correct
            total_pos += pos_mask.sum().item()
            
            # 负样本统计（背景类=0）
            neg_preds = pred_classes[neg_mask]
            neg_correct = (neg_preds == 0).sum().item()  # 背景类索引为0
            total_neg_correct += neg_correct
            total_neg += neg_mask.sum().item()
    
    # 计算各类准确率
    pos_acc = total_pos_correct / total_pos if total_pos > 0 else 0
    neg_acc = total_neg_correct / total_neg if total_neg > 0 else 0
    total_acc = (pos_acc + neg_acc) / 2
    
    print(f"正样本准确率: {pos_acc:.4f} ({total_pos_correct}/{total_pos})")
    print(f"负样本准确率: {neg_acc:.4f} ({total_neg_correct}/{total_neg})")
    print(f"整体准确率: {total_acc:.4f}")
    
    return total_acc, pos_acc, neg_acc



# 将训练和验证过程封装成函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    class_loss = 0.0
    loc_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        # 检查是否完全没有正样本
        total_pos_in_batch = torch.cat([m for m in batch['pos_mask']]).sum().item()
        if total_pos_in_batch == 0:
            print("警告: 当前批次没有任何正样本!")
            
            # 跳过训练此批次
            continue
        
        # 前向传播
        loc_preds, conf_preds = model(images)
        
        # 准备目标值
        batch_labels = torch.stack(batch['labels'], dim=0).to(device)
        batch_offsets = torch.stack(batch['offsets'], dim=0).to(device)
        batch_pos_mask = torch.stack(batch['pos_mask'], dim=0).to(device)

        
        # 计算损失
        total_batch_loss, class_batch_loss, loc_batch_loss = criterion(
            loc_preds, 
            batch_offsets,
            conf_preds,
            batch_labels,
            batch_pos_mask
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_batch_loss.backward()

        # === 新增：正样本梯度强化 ===
        if class_loss > 10 * loc_loss:
            # 获取分类头参数
            for param in model.conf_preds.parameters():  # 假设模型有conf_preds模块
                if param.grad is not None:
                    param.grad *= 2.0  # 梯度放大

        optimizer.step()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # 累计损失
        total_loss += total_batch_loss.item()
        class_loss += class_batch_loss.item()
        loc_loss += loc_batch_loss.item()
    
    # 计算平均损失
    num_batches = len(dataloader)
    total_loss /= num_batches
    class_loss /= num_batches
    loc_loss /= num_batches
    
    return total_loss, class_loss, loc_loss

def validate(model, dataloader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    class_loss = 0.0
    loc_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            
            # 前向传播
            loc_preds, conf_preds = model(images)
            
            # 准备目标值
            batch_labels = torch.stack(batch['labels'], dim=0).to(device)
            batch_offsets = torch.stack(batch['offsets'], dim=0).to(device)
            batch_pos_mask = torch.stack(batch['pos_mask'], dim=0).to(device)
            
            # 计算损失
            total_batch_loss, class_batch_loss, loc_batch_loss = criterion(
                loc_preds, 
                batch_offsets,
                conf_preds,
                batch_labels,
                batch_pos_mask
            )
            
            total_loss += total_batch_loss.item()
            class_loss += class_batch_loss.item()
            loc_loss += loc_batch_loss.item()
    
    # 计算平均损失
    num_batches = len(dataloader)
    total_loss /= num_batches
    class_loss /= num_batches
    loc_loss /= num_batches
    
    return total_loss, class_loss, loc_loss

def train_model(
        model, 
        num_classes,
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        device, 
        num_epochs=100, 
        save_path='./SaveModels/best_ssd_model.pt',
        # 新增锚框配置参数
        clustered_anchors=None,  # 聚类生成的锚框尺寸
        num_anchors_per_level=None  # 每层锚框数量
    ):
    """训练模型"""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    # 添加训练过程记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_class_loss': [],
        'val_class_loss': [],
        'train_loc_loss': [],
        'val_loc_loss': [],
        'learning_rate': [],
        'train_acc': [],  # 记录训练精确率
        'val_acc': []    # 记录验证精确率
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss, train_class_loss, train_loc_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # 计算训练精确率
        train_acc, train_pos_acc, train_neg_acc = calculate_accuracy(model, train_loader, device, num_classes)
        
        # 验证
        val_loss, val_class_loss, val_loc_loss = validate(
            model, val_loader, criterion, device
        )
        # 计算验证精确率
        val_acc, val_pos_acc, val_neg_acc = calculate_accuracy(model, val_loader, device, num_classes)
        
        # 更新学习率
        # scheduler.step(val_loss)
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_class_loss'].append(train_class_loss)
        history['val_class_loss'].append(val_class_loss)
        history['train_loc_loss'].append(train_loc_loss)
        history['val_loc_loss'].append(val_loc_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印训练进度
        print(f"  Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Loc: {train_loc_loss:.4f})")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Loc: {val_loc_loss:.4f})")
        print(f"  Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型（基于验证损失和精确率双指标）
        if val_loss < best_val_loss or val_acc> best_val_acc:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
            # 保存关键配置信息
            torch.save({
                'model_state_dict': model.state_dict(),
                'clustered_anchors': clustered_anchors,  # 聚类锚框尺寸
                'num_anchors_per_level': num_anchors_per_level,  # 每层锚框数
                'num_classes': num_classes,
                'feature_map_sizes': feature_map_sizes  # 特征图尺寸
            }, save_path)
            print(f"  Saved best model with val loss: {val_loss:.4f} and val acc: {val_acc:.4f}")

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # 添加学习率变化日志
        print(f"  当前学习率: {current_lr:.6f}")
        # 绘制精确率&损失曲线
        DetectionTrainingVisualizer.plot_acc_loss_curve(history)
    
    print("训练完成!")
    return history


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yelan = Yelan.Yelan()
    img2object_list = yelan.parse_xml()
    # 该列表用来采集填充缩放之后，以及图像增广之后的边界框和图像数据
    Aug_img2object_list = []
    print('开始进行图像填充及图像增广...')
    bar = tqdm(total=len(img2object_list))
    for object in img2object_list:
        # 检查原始图像有效性
        if object['img'] is None:
            print(f"警告：跳过无效的原始图像 {object['source_image_id']}")
            continue
        # 先将图像进行填充缩放，真实边界框同步缩放
        try:
            padded_resize_object = DataAugmentation.padded_resize(
                object['img'], 
                object['Boxes'], 
                (640, 640)
            )
        except AttributeError as e:
            print(f"错误：处理图像 {object['source_image_id']} 时发生异常")
            print("可能原因：输入图像尺寸异常，详情：", e)
            continue
        padded_resize_object['source_image_id'] = object['source_image_id']
        # print("padded_resize_object['source_image_id']:", padded_resize_object['source_image_id'])
        Aug_img2object_list.append(padded_resize_object)  # 将缩放后的图像和真实边界框保存
        # # 查看填充后的图像及真实边界框
        # VisualizeBoxes.visualize_bboxes(padded_resize_object['img'], padded_resize_object['Boxes'])
        # 对图像进行左右翻转，真实边界框同步翻转
        flip_object = DataAugmentation.img_flip(padded_resize_object, 0)  # 左右翻转
        Aug_img2object_list.append(flip_object)  # 将翻转后的图像和真实边界框保存
        # # 查看翻转后的图像及真实边界框
        # VisualizeBoxes.visualize_bboxes(flip_object['img'], flip_object['Boxes'])
        bar.update(1)
    # print('len(Aug_img2object_list):', len(Aug_img2object_list))

    # 生成多尺度锚框、匹配真实边界框、计算偏移量
    print('开始生成多尺度锚框、匹配真实边界框、计算偏移量...')
    bar = tqdm(total=len(Aug_img2object_list))
    # 训练前初始化（只需执行一次）
    # 提取所有真实框宽高
    all_gt_wh = Anchors.extract_gt_wh(Aug_img2object_list)

    # 聚类生成锚框尺寸（使用训练集数据）
    clustered_anchors = Anchors.cluster_gt_boxes(all_gt_wh, num_anchors=18)
    # print('clustered_anchors:', clustered_anchors)

    # 5. 在数据加载时使用聚类生成的锚框
    feature_map_sizes = [(40,40), (20,20), (10,10), (5,5), (3,3), (1,1)]
    img_size = (640, 640)

    for object in Aug_img2object_list:
        img_size = object['img'].shape[:2]
        # 生成锚框
        anchors = Anchors.generate_anchors(
            img_size=img_size,
            feature_map_sizes=feature_map_sizes,
            clustered_anchors=clustered_anchors,
            device=device
        )
        # 转换颜色通道
        rgb_img = cv2.cvtColor(object['img'], cv2.COLOR_BGR2RGB)
        # print(f"生成锚框数量：{len(anchors)}")
        # # 可视化锚框分布
        # VisualizeBoxes.visualize_anchors(rgb_img , anchors)
        # 计算IOU并构建真实边界框-锚框匹配矩阵
        # 提取真实框信息
        gt_boxes = [box['xyxy'] for box in object['Boxes']]
        gt_labels = [yelan.class_name_to_id[box['class_name']] for box in object['Boxes']]
        
        # 转换为张量
        gt_boxes_tensor = torch.tensor(gt_boxes, device=device)
        gt_labels_tensor = torch.tensor(gt_labels, device=device, dtype=torch.long)
        
        # 匹配锚框与真实框
        anchor_labels, anchor_offsets, positive_mask = Anchors.match_anchors(
            anchors, gt_boxes_tensor, gt_labels_tensor, iou_threshold=0.5
        )
        # # 验证图像匹配到的正样本锚框数量
        # n_pos = positive_mask.sum().item()
        # print(f"图像 {object['source_image_id']} 匹配到 {n_pos} 个正样本锚框")
        
        # 将结果存入对象（转换为numpy用于可视化）
        object['anchors'] = anchors.cpu().numpy()  
        object['anchor_labels'] = anchor_labels.cpu().numpy()
        object['anchor_offsets'] = anchor_offsets.cpu().numpy()
        object['positive_mask'] = positive_mask.cpu().numpy()
        bar.update(1)
        
        # # 可视化匹配结果
        # VisualizeBoxes.plot_matched_anchors(object['img'], anchors, object['Boxes'], positive_mask)
    
    # 切割数据集
    train_data, val_data, test_data, split_plan, class_info = stratified_split_by_objects(
        Aug_img2object_list,
        test_size=0.1,
        val_size=0.2,
        min_samples=2,
        seed=42
    )

    # 保存数据集划分
    data_split_info = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'class_info': class_info
    }
    # 保存到文件
    with open('./Dataset/Yelan/data_splits.pkl', 'wb') as f:
        pickle.dump(data_split_info, f)
        
    print("数据集划分已保存至 './Dataset/Yelan/data_splits.pkl'")

    # 打印分割结果
    print("===== 分类分布统计 =====")
    for cls in sorted(class_info.keys()):
        total = class_info[cls]['count']
        train_p = split_plan['train'][cls]/total*100
        val_p = split_plan['val'][cls]/total*100
        test_p = split_plan['test'][cls]/total*100
        print(f"{cls}: 训练 {split_plan['train'][cls]} ({train_p:.1f}%) | "
            f"验证 {split_plan['val'][cls]} ({val_p:.1f}%) | "
            f"测试 {split_plan['test'][cls]} ({test_p:.1f}%)")
        
    
    # 创建PyTorch数据集
    train_dataset = DetectionDataset(train_data)
    val_dataset = DetectionDataset(val_data)
    test_dataset = DetectionDataset(test_data)
    
    # 数据加载器（处理锚框数量不均衡问题）
    def collate_fn(batch):
        # 确保每个字段都是张量列表
        return {
            'image': torch.stack([x['image'] for x in batch], dim=0),
            'anchors': [x['anchors'] for x in batch],    # 不同图像的锚框数量不同
            'labels': [x['labels'] for x in batch],
            'offsets': [x['offsets'] for x in batch],
            'pos_mask': [x['pos_mask'] for x in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
    # 打印数据集基本信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 打印数据加载器配置
    print(f"\n训练集批次大小: {train_loader.batch_size}")
    print(f"验证集批次大小: {val_loader.batch_size}")

    # 打印总批次数量
    print(f"\n训练集总批次: {len(train_loader)}")
    print(f"验证集总批次: {len(val_loader)}")

    # 获取第一个批次的数据形状示例
    sample_batch = next(iter(train_loader))
    print("\n第一个训练批次的形状:")
    print(f"图像: {sample_batch['image'].shape} [Batch, Channel, Height, Width]")
    print(f"锚框数量: {len(sample_batch['anchors'][0])} ")
    
    # 初始化模型
    num_classes = len(yelan.class_name_to_id)
    num_anchors_per_level = [3] * 6  # 6个特征层，每层3个锚框
    model = SSD(num_classes=num_classes,
                num_anchors_per_level=num_anchors_per_level).to(device)
    criterion = SSDLoss(num_classes=num_classes)
    
    # 优化器， 学习率默认1e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学习率调度器, 采用warmup策略避免初期震荡
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=len(train_loader), 
        epochs=10
    )
    
    # 训练模型
    history = train_model(
        model=model,
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        clustered_anchors=clustered_anchors,  # 传递聚类结果
        num_anchors_per_level=num_anchors_per_level  # 传递锚框分配
    )
    
    print("训练完成!")

        