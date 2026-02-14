# ModelClasses/SSD.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 在VGGBase的conv4后添加CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        channel_att = self.channel_att(x)
        x_channel = x * channel_att
        
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        
        return x_channel * spatial_att

class VGGBase(nn.Module):
    """VGG基础网络，用于特征提取"""
    def __init__(self):
        super(VGGBase, self).__init__()
        # 输入图像: [3, 640, 640]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [64, 320, 320]
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [128, 160, 160]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [256, 80, 80]
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [512, 40, 40]
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # [512, 40, 40]
        )
        # 额外的卷积层用于特征提取
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),  # 空洞卷积扩大感受野
            nn.ReLU(inplace=True)  # [1024, 40, 40]
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)  # [1024, 40, 40]
        )
        self.cbam4 = CBAM(512)  # 在conv4后
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.cbam4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x

class AuxiliaryConvs(nn.Module):
    """辅助卷积层，用于多尺度特征提取"""
    def __init__(self):
        super(AuxiliaryConvs, self).__init__()
        # 输入: [1024, 40, 40]
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # [512, 20, 20]
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [256, 10, 10]
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [256, 5, 5]
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [256, 3, 3]
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),  # [256, 1, 1]
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 从conv7的输出开始
        conv8 = self.conv8(x)      # [512, 20, 20]
        conv9 = self.conv9(conv8)  # [256, 10, 10]
        conv10 = self.conv10(conv9) # [256, 5, 5]
        conv11 = self.conv11(conv10) # [256, 3, 3]
        conv12 = self.conv12(conv11) # [256, 1, 1]
        
        return [x, conv8, conv9, conv10, conv11, conv12]

    
class PredictionConvs(nn.Module):
    """预测卷积层 - 修改为直接接收锚框数量"""
    def __init__(self, num_classes, num_anchors_per_level):
        super(PredictionConvs, self).__init__()
        self.num_classes = num_classes
        self.num_anchors_per_level = num_anchors_per_level  # 直接传入每层锚框数量
        
        # 特征图通道数 (根据AuxiliaryConvs的输出)
        in_channels = [1024, 512, 256, 256, 256, 256]
        
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        
        for i, (num_anchor, in_channel) in enumerate(zip(num_anchors_per_level, in_channels)):
            # 定位预测: 每个锚框4个值
            self.loc_layers.append(
                nn.Conv2d(in_channel, num_anchor * 4, kernel_size=3, padding=1)
            )
            # 分类预测: 每个锚框(num_classes + 1)个值
            self.conf_layers.append(
                nn.Conv2d(in_channel, num_anchor * (num_classes + 1), kernel_size=3, padding=1)
            )
    
    def forward(self, features):
        """
        前向传播
        features: 来自AuxiliaryConvs的6个特征图列表
        """
        loc_preds = []
        conf_preds = []
        
        for i, feat in enumerate(features):
            # 定位预测
            loc_pred = self.loc_layers[i](feat)
            # 调整形状: [batch, num_anchors*4, H, W] -> [batch, H, W, num_anchors*4] -> [batch, H*W*num_anchors, 4]
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)
            loc_preds.append(loc_pred)
            
            # 分类预测
            conf_pred = self.conf_layers[i](feat)
            # 调整形状: [batch, num_anchors*(num_classes+1), H, W] -> [batch, H, W, num_anchors*(num_classes+1)] -> [batch, H*W*num_anchors, num_classes+1]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            conf_pred = conf_pred.view(conf_pred.size(0), -1, self.num_classes + 1)
            conf_preds.append(conf_pred)
        
        # 将所有特征图的预测拼接起来
        loc_preds = torch.cat(loc_preds, dim=1)  # [batch, total_anchors, 4]
        conf_preds = torch.cat(conf_preds, dim=1)  # [batch, total_anchors, num_classes+1]
        
        return loc_preds, conf_preds
    

class SSD(nn.Module):
    """SSD模型"""
    def __init__(self, num_classes, num_anchors_per_level):
        super(SSD, self).__init__()
        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvs()
        self.pred_convs = PredictionConvs(num_classes, num_anchors_per_level)
        # self.dropout = nn.Dropout(0.5)  # 添加Dropout层
    
    def forward(self, x):
        # 基础网络提取特征
        base_features = self.base(x)
        # 多尺度特征提取
        features = self.aux_convs(base_features)
        # 预测
        loc_preds, conf_preds = self.pred_convs(features)
        # conf_preds = self.dropout(conf_preds)  # 在分类头前应用
        return loc_preds, conf_preds


# class SSDLoss(nn.Module):
#     """SSD损失函数"""
#     def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0):
#         super(SSDLoss, self).__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.neg_pos_ratio = neg_pos_ratio
 
#     def forward(self, loc_preds, loc_targets, conf_preds, conf_targets, pos_mask):
#         batch_size = loc_preds.size(0)
#         num_anchors = loc_preds.size(1)
        
#         # 1. 定位损失 (仅正样本)
#         # 获取扩展后的pos_mask用于定位损失计算
#         pos_mask_expanded = pos_mask.unsqueeze(2).expand_as(loc_preds)
#         if pos_mask_expanded.sum() == 0:
#             loc_loss = torch.tensor(0.0, device=loc_preds.device)
#         else:
#             loc_loss = F.smooth_l1_loss(
#                 loc_preds[pos_mask_expanded], 
#                 loc_targets[pos_mask_expanded], 
#                 reduction='sum'
#             )
        
#         # 负样本过多的情况下需要降低负样本的权重
#         num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
#         num_neg = (self.neg_pos_ratio * num_pos).clamp(max=num_anchors-1)
#         # 计算批次平均权重（标量）
#         if num_neg.sum() > num_pos.sum():
#             neg_weight = (num_pos.sum() / num_neg.sum()).item()
#         else:
#             neg_weight = 1.0
#         # 2. 分类损失计算
#         conf_loss = FocalLoss(alpha=0.25, gamma=2)(conf_preds, conf_targets)
#         # conf_loss = F.cross_entropy(
#         #     conf_preds.view(-1, self.num_classes + 1),
#         #     conf_targets.view(-1),
#         #     reduction='none',
#         #     weight=torch.tensor([neg_weight] + [1.0]*self.num_classes, device=conf_preds.device)  # 背景类权重降低
#         # ).view(batch_size, num_anchors)

#         # 3. 难负样本挖掘（优化实现）        
#         # 按损失值排序取最难负样本
#         _, neg_idx = conf_loss.sort(dim=1, descending=True)
#         neg_rank = torch.zeros_like(neg_idx)
#         for i in range(batch_size):
#             neg_rank[i, neg_idx[i]] = torch.arange(num_anchors, device=loc_preds.device)
        
#         neg_mask = neg_rank < num_neg
        
#         # 4. 最终分类损失（仅计算正样本+精选负样本）
#         conf_loss_pos = conf_loss[pos_mask].sum()
#         conf_loss_neg = conf_loss[neg_mask & ~pos_mask].sum()
#         conf_loss_total = conf_loss_pos + conf_loss_neg
        
#         # 5. 总损失
#         total_loss = (conf_loss_total + self.alpha * loc_loss) / num_pos.sum().clamp(min=1)
        
#         return total_loss, conf_loss_total / num_pos.sum().clamp(min=1), loc_loss / num_pos.sum().clamp(min=1)
    

class SSDLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0, neg_pos_ratio=3.0):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, loc_preds, loc_targets, conf_preds, conf_targets, pos_mask):
        # 定位损失
        pos_mask_expanded = pos_mask.unsqueeze(2).expand_as(loc_preds)
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_mask_expanded], 
            loc_targets[pos_mask_expanded], 
            reduction='sum'
        )
        # === 定位损失稳定性约束 ===
        loc_loss = torch.where(
            torch.isnan(loc_loss), 
            torch.zeros_like(loc_loss), 
            loc_loss
        )
        
        # 重构FocalLoss计算（确保二维输出）
        conf_preds_flat = conf_preds.view(-1, self.num_classes+1)
        conf_targets_flat = conf_targets.view(-1)
        
        ce_loss = F.cross_entropy(
            conf_preds_flat, 
            conf_targets_flat, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 重塑为二维: [batch, num_anchors]
        conf_loss_per_anchor = focal_loss.view(pos_mask.size(0), pos_mask.size(1))
        
        # 安全处理：确保有锚框存在
        if conf_loss_per_anchor.numel() == 0:
            num_pos = torch.tensor(1.0, device=loc_preds.device)  # 避免除零
            return torch.tensor(0.0, device=loc_preds.device), torch.tensor(0.0), loc_loss / num_pos
        
        # 困难负样本挖掘（维度安全版）
        with torch.no_grad():
            neg_conf = conf_loss_per_anchor.clone()
            neg_conf[pos_mask] = 0  # 正样本位置归零
            
            # 按损失值排序取最难负样本
            values, neg_idx = neg_conf.sort(dim=1, descending=True)
            
            # 关键修复：处理单样本批次
            if neg_idx.dim() == 1:
                neg_idx = neg_idx.unsqueeze(0)  # 添加批次维度
                
            # 生成排名矩阵
            batch_size, num_anchors = neg_idx.shape
            ranks = torch.arange(num_anchors, device=neg_idx.device).expand(batch_size, -1)
            neg_rank = torch.zeros_like(neg_idx).scatter_(1, neg_idx, ranks)
            
            # 生成负样本掩码
            num_pos = pos_mask.sum(dim=1).clamp(min=1)
            num_neg = (self.neg_pos_ratio * num_pos).clamp(max=num_anchors-1)
            neg_mask = neg_rank < num_neg.unsqueeze(1)
        
        if neg_mask.sum() == 0:  # 无负样本时紧急处理
            # 选择损失最小的k个负样本 (最易错负样本)
            _, easy_neg_idx = neg_conf.sort(dim=1, descending=False)
            neg_mask = torch.zeros_like(pos_mask)
            for i in range(batch_size):
                if num_neg[i] > 0:
                    neg_mask[i, easy_neg_idx[i, :num_neg[i]]] = 1
        
        # # === 正样本权重调整 ===
        # if pos_mask.sum() > 0:
        #     # 确保分母不为零
        #     neg_count = max(neg_mask.sum(), 1)  # 至少取1防止除零
        #     # 动态权重计算（限制最大权重）
        #     pos_weight = torch.clamp(1 * neg_count / pos_mask.sum(), max=100.0)
        #     # 仅增强正样本，不归零
        #     conf_loss_per_anchor[pos_mask] *= pos_weight
        
        # 计算分类损失（仅正样本+困难负样本）
        class_loss = conf_loss_per_anchor[pos_mask].sum() + conf_loss_per_anchor[neg_mask].sum()    
            
        # 总损失计算
        num_pos_total = num_pos.sum().clamp(min=1)
        total_loss = (class_loss + loc_loss) / num_pos_total
        
        return total_loss, class_loss / num_pos_total, loc_loss / num_pos_total


