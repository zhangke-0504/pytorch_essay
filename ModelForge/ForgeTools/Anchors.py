import torch
import math
import numpy as np
from sklearn.cluster import KMeans

# 1. 从Aug_img2object_list提取所有真实框的宽高
def extract_gt_wh(augmented_data):
    """从增强数据中提取所有真实框的宽高"""
    all_wh = []
    for obj in augmented_data:
        for box in obj['Boxes']:
            x1, y1, x2, y2 = box['xyxy']
            w = x2 - x1
            h = y2 - y1
            all_wh.append([w, h])
    return np.array(all_wh)

# 2. 聚类真实框生成锚框尺寸
def cluster_gt_boxes(gt_wh, num_anchors=18, max_iter=300, random_state=42):
    """
    基于真实边界框的K-means聚类生成锚框尺寸
    Args:
        gt_wh: 所有真实框的宽高数组 [N,2]
        num_anchors: 需要生成的锚框总数
    Returns:
        anchor_wh: 聚类得到的锚框宽高 [num_anchors, 2]
    """
    # 归一化处理（消除图像尺寸影响）
    norm_wh = gt_wh / np.sqrt(np.max(gt_wh[:, 0]) * np.max(gt_wh[:, 1]))
    
    # 自定义IoU距离度量
    def iou_distance(boxes, centroids):
        boxes = np.expand_dims(boxes, 1)  # [N, 1, 2]
        centroids = np.expand_dims(centroids, 0)  # [1, K, 2]
        
        min_wh = np.minimum(boxes, centroids)
        inter_area = min_wh[..., 0] * min_wh[..., 1]
        
        box_area = boxes[..., 0] * boxes[..., 1]
        centroid_area = centroids[..., 0] * centroids[..., 1]
        
        union_area = box_area + centroid_area - inter_area
        iou = inter_area / (union_area + 1e-10)
        return 1 - iou  # [N, K]
    
    # K-means++聚类
    kmeans = KMeans(
        n_clusters=num_anchors,
        init='k-means++',
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    
    # 使用自定义距离度量
    kmeans.fit(norm_wh, sample_weight=None)
    
    # 获取聚类中心并反归一化
    centroids = kmeans.cluster_centers_
    centroids = centroids * np.sqrt(np.max(gt_wh[:, 0]) * np.max(gt_wh[:, 1]))
    
    # 按面积排序（小->大）
    centroid_areas = centroids[:, 0] * centroids[:, 1]
    sorted_indices = np.argsort(centroid_areas)
    return centroids[sorted_indices]

# def generate_anchors(img_size, feature_map_sizes, clustered_anchors, device="cuda"):
#     """
#     基于聚类结果生成多尺度锚框
#     Args:
#         img_size: 图像尺寸 (h, w)
#         feature_map_sizes: 各层级特征图尺寸列表 [(h1,w1), (h2,w2), ...]
#         clustered_anchors: 聚类得到的锚框宽高 [K,2]
#         device: 计算设备
#     """
#     h, w = img_size
#     anchors = []
#     total_anchors = len(clustered_anchors)
    
#     # 将聚类锚框分配到不同特征图层级
#     anchors_per_level = total_anchors // len(feature_map_sizes)
#     anchor_groups = [
#         clustered_anchors[i*anchors_per_level : (i+1)*anchors_per_level] 
#         for i in range(len(feature_map_sizes))
#     ]

    
#     # 遍历各特征图层级
#     for i, (f_h, f_w) in enumerate(feature_map_sizes):
#         # 当前层级的锚框尺寸
#         level_anchors = anchor_groups[i]
#         num_anchors = len(level_anchors)
        
#         # 计算特征图步长
#         stride_y = h / f_h
#         stride_x = w / f_w
        
#         # 生成中心点网格
#         shift_y = (torch.arange(f_h, device=device) + 0.5) * stride_y
#         shift_x = (torch.arange(f_w, device=device) + 0.5) * stride_x
#         grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
#         centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        
#         # 生成锚框坐标
#         wh_tensor = torch.tensor(level_anchors, device=device, dtype=torch.float32)
#         cxcy = centers.repeat_interleave(num_anchors, dim=0)
#         wh = wh_tensor.repeat(len(centers), 1)
        
#         # 计算边界框坐标
#         anchors_layer = torch.cat([
#             cxcy - wh/2,  # xmin, ymin
#             cxcy + wh/2   # xmax, ymax
#         ], dim=1)
#         anchors.append(anchors_layer)
    
#     return torch.cat(anchors, dim=0)

def generate_anchors(img_size, feature_map_sizes, clustered_anchors, device="cuda"):
    """
    基于聚类结果生成多尺度锚框（优化版）
    Args:
        img_size: 图像尺寸 (h, w)
        feature_map_sizes: 各层级特征图尺寸列表 [(h1,w1), (h2,w2), ...]
        clustered_anchors: 聚类得到的锚框宽高 [K,2]
        device: 计算设备
    """
    h, w = img_size
    anchors = []
    
    # === 按目标尺寸动态分配锚框到各层级 ===
    # 计算每个锚框的面积
    centroid_areas = clustered_anchors[:, 0] * clustered_anchors[:, 1]
    
    # 按面积分位数划分（大目标分配到低分辨率层）
    sorted_areas = np.sort(centroid_areas)
    quantiles = np.linspace(0, 1, len(feature_map_sizes)+1)
    level_bins = np.quantile(sorted_areas, quantiles)
    
    # 创建层级分组
    anchor_groups = []
    for i in range(len(feature_map_sizes)):
        lower_bound = level_bins[i]
        upper_bound = level_bins[i+1]
        mask = (centroid_areas >= lower_bound) & (centroid_areas <= upper_bound)
        anchor_groups.append(clustered_anchors[mask])
    
    # === 确保每个层级至少有一个锚框 ===
    for i, group in enumerate(anchor_groups):
        if len(group) == 0:
            # 从最近层级复制一个锚框
            nearest_idx = min(i, len(anchor_groups)-1)
            anchor_groups[i] = anchor_groups[nearest_idx][:1].copy()
    
    # 遍历各特征图层级
    for i, (f_h, f_w) in enumerate(feature_map_sizes):
        level_anchors = anchor_groups[i]
        
        # 计算特征图步长
        stride_y = h / f_h
        stride_x = w / f_w
        
        # 生成中心点网格
        shift_y = (torch.arange(f_h, device=device) + 0.5) * stride_y
        shift_x = (torch.arange(f_w, device=device) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
        
        # 生成锚框坐标
        wh_tensor = torch.tensor(level_anchors, device=device, dtype=torch.float32)
        cxcy = centers.repeat_interleave(len(level_anchors), dim=0)
        wh = wh_tensor.repeat(len(centers), 1)
        
        # 计算边界框坐标
        anchors_layer = torch.cat([
            cxcy - wh/2,  # xmin, ymin
            cxcy + wh/2   # xmax, ymax
        ], dim=1)
        anchors.append(anchors_layer)
    
    return torch.cat(anchors, dim=0)



def compute_iou(anchors, gt_boxes):
    """
    计算锚框与真实框的IoU矩阵
    anchors: Tensor[N,4] (xmin, ymin, xmax, ymax)
    gt_boxes: Tensor[M,4]
    Returns: IoU矩阵 Tensor[N,M]
    """
    # 扩展维度用于广播计算
    anchors = anchors.unsqueeze(1)  # [N,1,4]
    gt_boxes = gt_boxes.unsqueeze(0)  # [1,M,4]

    # 计算交集区域坐标
    '''
    多维张量索引 ：
    当需要对多维张量的某些维度进行索引，
    但又不想明确写出所有维度时，
    可以使用 ... 表示省略中间的维度。
    例如，对于一个形状为 [N, M, K] 的张量 tensor，tensor[..., 0] 
    表示取该张量最后一个维度下标为 0 的元素，即形状为 [N, M] 的张量。
    
    广播操作中的维度匹配 ：
    在进行广播操作时，... 可以帮助自动匹配张量的维度。
    比如在 torch.max(anchors[..., 0], gt_boxes[..., 0]) 中，
    anchors 和 gt_boxes 都是多维张量，
    通过 ... 可以让它们在广播时自动匹配前面的维度，
    然后对最后一个维度的对应元素进行比较，求取最大值。
    '''
    inter_xmin = torch.max(anchors[..., 0], gt_boxes[..., 0])
    inter_ymin = torch.max(anchors[..., 1], gt_boxes[..., 1])
    inter_xmax = torch.min(anchors[..., 2], gt_boxes[..., 2])
    inter_ymax = torch.min(anchors[..., 3], gt_boxes[..., 3])

    # 计算交集面积
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h

    # 计算各自面积
    area_anchors = (anchors[..., 2] - anchors[..., 0]) * (anchors[..., 3] - anchors[..., 1])
    area_gt = (gt_boxes[..., 2] - gt_boxes[..., 0]) * (gt_boxes[..., 3] - gt_boxes[..., 1])

    # 计算并集面积
    union_area = area_anchors + area_gt - inter_area

    # 计算IoU
    iou = inter_area / (union_area + 1e-6)
    return iou  # 直接返回二维矩阵

def compute_offsets(anchors, gt_boxes):
    """
    计算锚框到真实框的偏移量
    anchors: Tensor[N,4] (xmin, ymin, xmax, ymax)
    gt_boxes: Tensor[N,4]
    Returns: 偏移量 Tensor[N,4] (dx, dy, dw, dh)
    """
    # 将锚框转换为 (cx, cy, w, h)
    anc_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anc_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anc_w = anchors[:, 2] - anchors[:, 0]
    anc_h = anchors[:, 3] - anchors[:, 1]

    # 真实框转换
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    # 计算偏移量
    eps = 1e-6
    dx = (gt_cx - anc_cx) / (anc_w + eps)
    dy = (gt_cy - anc_cy) / (anc_h + eps)
    dw = torch.log(gt_w / (anc_w + eps) + eps)
    dh = torch.log(gt_h / (anc_h + eps) + eps)

    return torch.stack([dx, dy, dw, dh], dim=1)

def match_anchors(anchors, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    锚框与真实框匹配
    anchors: Tensor[N,4]
    gt_boxes: Tensor[M,4]
    gt_labels: Tensor[M]
    Returns:
        anchor_labels: Tensor[N] 类别标签
        anchor_offsets: Tensor[N,4] 偏移量
        positive_mask: Tensor[N] 正样本掩码
    """
    device = anchors.device
    M = gt_boxes.shape[0]
    N = anchors.shape[0]

    if M == 0:
        # 无真实框情况
        return (torch.zeros(N, dtype=torch.long, device=device),
                torch.zeros((N,4), device=device),
                torch.zeros(N, dtype=torch.bool, device=device))

    # 计算IoU矩阵
    iou_matrix = compute_iou(anchors, gt_boxes)
    
    # 每个锚框的最大IoU和对应索引
    max_iou, matched_gt_idx = torch.max(iou_matrix, dim=1)
    
    # 初步正样本：IoU超过阈值
    positive_mask = max_iou >= iou_threshold
    
    # 确保每个真实框至少匹配一个锚框
    _, best_anchor_idx = torch.max(iou_matrix, dim=0)
    positive_mask[best_anchor_idx] = True
    matched_gt_idx[best_anchor_idx] = torch.arange(M, device=device)
    
    # 获取匹配的真实标签
    matched_gt_labels = gt_labels[matched_gt_idx]
    
    # 生成最终标签（背景类为0）
    anchor_labels = torch.where(positive_mask, 
                               matched_gt_labels, 
                               torch.zeros_like(matched_gt_labels))
    
    # 计算偏移量
    matched_gt_boxes = gt_boxes[matched_gt_idx]
    anchor_offsets = compute_offsets(anchors, matched_gt_boxes)
    
    return anchor_labels, anchor_offsets, positive_mask

def apply_offsets(anchors, offsets):
    """
    将预测的偏移量应用到锚框上，得到预测边界框
    Args:
        anchors: Tensor[N,4] (xmin, ymin, xmax, ymax)
        offsets: Tensor[N,4] (dx, dy, dw, dh)
    Returns:
        pred_boxes: Tensor[N,4] 预测框坐标 (xmin, ymin, xmax, ymax)
    """
    # 将锚框转换为 (cx, cy, w, h)
    anc_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anc_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anc_w = anchors[:, 2] - anchors[:, 0]
    anc_h = anchors[:, 3] - anchors[:, 1]
    
    # 解析偏移量
    dx = offsets[:, 0]
    dy = offsets[:, 1]
    dw = offsets[:, 2]
    dh = offsets[:, 3]
    
    # 应用偏移量公式（反向计算）
    pred_cx = dx * anc_w + anc_cx
    pred_cy = dy * anc_h + anc_cy
    pred_w = torch.exp(dw) * anc_w
    pred_h = torch.exp(dh) * anc_h
    
    # 转换回边界框坐标
    xmin = pred_cx - pred_w / 2
    ymin = pred_cy - pred_h / 2
    xmax = pred_cx + pred_w / 2
    ymax = pred_cy + pred_h / 2
    
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)