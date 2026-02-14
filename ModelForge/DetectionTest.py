from DataProcess import Yelan
from ForgeTools import DataAugmentation, Anchors, VisualizeBoxes
import cv2
import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim
from ModelClasses.SSD import SSD
import os
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob
from PIL import Image, ImageDraw, ImageFont

# 自定义数据集类
class DetectionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 添加image_id返回
        return {
            'image': torch.from_numpy(item['img']).permute(2, 0, 1).float() / 255.0,
            'anchors': item['anchors'],
            'labels': item['anchor_labels'],
            'offsets': item['anchor_offsets'],
            'pos_mask': item['positive_mask'],
            'image_id': item['source_image_id']  # 添加图像ID
        }
    
def calculate_ap(recall, precision):
    """计算PR曲线下面积(AP) - 使用11点插值法[3,7](@ref)"""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算边界框面积
    Args:
        boxes: [N, 4] 格式的边界框坐标 (x1, y1, x2, y2)
    Returns:
        areas: [N] 每个边界框的面积
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框的交并比 (IoU)
    Args:
        boxes1: [N, 4] 格式的边界框坐标 (x1, y1, x2, y2)
        boxes2: [M, 4] 格式的边界框坐标 (x1, y1, x2, y2)
    Returns:
        iou: [N, M] 两两组之间的IoU值
    """
    # 计算所有边界框的面积
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]
    
    # 计算交集区域的左上角和右下角坐标
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    # 计算交集区域的宽高并钳制为非负值
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # 计算并集面积
    union = area1[:, None] + area2 - inter  # [N, M]
    
    # 计算IoU（避免除零错误）
    iou = inter / (union + 1e-7)  # [N, M]
    return iou

def evaluate_map50(model, dataloader, ground_truths, class_info, device, iou_threshold=0.5):
    """计算mAP50指标[1,3](@ref)"""
    model.eval()
    # 存储每个类别的预测结果 {class_id: [confidence, is_tp]}
    class_stats = defaultdict(list)
    # 存储每个类别的真实框数量 {class_id: count}
    gt_counter_per_class = defaultdict(int)
    
    # 统计每个类别的真实框数量[7](@ref)
    for img_id, boxes in ground_truths.items():
        for box in boxes:
            class_id = box['class_id']
            gt_counter_per_class[class_id] += 1

    # 遍历测试集
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="mAP50评估"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            
            # 模型推理
            loc_preds, conf_preds = model(images)
            
            # 处理批次中的每张图像
            for i in range(images.size(0)):
                img_id = image_ids[i]
                anchors = torch.tensor(batch['anchors'][i], device=device)
                
                # 解码预测框[9](@ref)
                anchor_xywh = anchors  # (cx, cy, w, h)
                loc_data = loc_preds[i]
                conf_data = conf_preds[i]
                
                # 转换为xyxy格式
                pred_cx = loc_data[:,0] * anchor_xywh[:,2] + anchor_xywh[:,0]
                pred_cy = loc_data[:,1] * anchor_xywh[:,3] + anchor_xywh[:,1]
                pred_w = anchor_xywh[:,2] * torch.exp(loc_data[:,2])
                pred_h = anchor_xywh[:,3] * torch.exp(loc_data[:,3])
                
                pred_boxes = torch.stack([
                    pred_cx - pred_w/2, 
                    pred_cy - pred_h/2,
                    pred_cx + pred_w/2,
                    pred_cy + pred_h/2
                ], dim=1)
                
                # 处理类别置信度
                conf_probs = torch.softmax(conf_data, dim=1)[:, 1:]  # 跳过背景类
                max_conf, max_idx = torch.max(conf_probs, dim=1)
                class_ids = max_idx + 1  # 类别索引从1开始
                
                # 置信度过滤和NMS
                keep = max_conf > 0.001  # 置信度阈值
                pred_boxes = pred_boxes[keep]
                scores = max_conf[keep]
                class_ids = class_ids[keep]
                
                if len(pred_boxes) > 0:
                    keep_idx = torchvision.ops.nms(pred_boxes.cpu(), scores.cpu(), 0.5)  # NMS阈值0.5
                    # 确保索引在原始设备上
                    keep_idx = keep_idx.to(pred_boxes.device)
                    pred_boxes = pred_boxes[keep_idx]
                    scores = scores[keep_idx]
                    class_ids = class_ids[keep_idx]
                
                # 获取当前图像的真实框
                gt_boxes = ground_truths.get(img_id, [])
                gt_bboxes = [torch.tensor([b['xmin'], b['ymin'], b['xmax'], b['ymax']], 
                                        device=device) for b in gt_boxes]
                gt_classes = [b['class_id'] for b in gt_boxes]
                
                # 记录已匹配的真实框[7](@ref)
                matched_gt_indices = set()
                
                # 处理每个预测框
                for j in range(len(pred_boxes)):
                    pred_box = pred_boxes[j]
                    pred_class = class_ids[j].item()
                    score = scores[j].item()
                    
                    # 初始化为FP
                    is_tp = 0
                    
                    # 只匹配相同类别的真实框
                    for k, gt_box in enumerate(gt_bboxes):
                        if gt_classes[k] != pred_class or k in matched_gt_indices:
                            continue
                            
                        iou = box_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0)).item()
                        if iou >= iou_threshold:
                            is_tp = 1
                            matched_gt_indices.add(k)
                            break
                    
                    # 记录统计信息
                    class_stats[pred_class].append((score, is_tp))
    
    # 计算每个类别的AP
    aps = []
    class_metrics = {}
    
    for class_id, stats in class_stats.items():
        if not stats:
            ap = 0.0
            class_metrics[class_id] = {'ap': ap, 'precision': 0, 'recall': 0}
            continue
            
        # 按置信度降序排序
        stats.sort(key=lambda x: x[0], reverse=True)
        scores_arr = np.array([s[0] for s in stats])
        is_tp_arr = np.array([s[1] for s in stats])
        
        # 计算累积TP和FP
        tp_cumsum = np.cumsum(is_tp_arr)
        fp_cumsum = np.cumsum(1 - is_tp_arr)
        
        # 计算精确率和召回率[3](@ref)
        recall = tp_cumsum / (gt_counter_per_class[class_id] + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # 计算AP
        ap = calculate_ap(recall, precision)
        aps.append(ap)
        
        # 记录类别指标
        class_metrics[class_id] = {
            'ap': ap,
            'precision': precision[-1] if len(precision) > 0 else 0,
            'recall': recall[-1] if len(recall) > 0 else 0,
            'num_pred': len(stats),
            'num_gt': gt_counter_per_class[class_id]
        }
    
    # 计算mAP50
    map50 = np.mean(aps) if aps else 0.0
    
    return map50, class_metrics

class YOLOv5Inference:
    def __init__(self, model_path, class_names, clustered_anchors, feature_map_sizes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.class_names = class_names
        self.id_to_class_name = {i: name for i, name in enumerate(class_names)}
        self.clustered_anchors = clustered_anchors
        self.feature_map_sizes = feature_map_sizes
        self.img_size = (640, 640)  # 与训练保持一致
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        self.model = SSD(
            num_classes=checkpoint['num_classes'],
            num_anchors_per_level=checkpoint['num_anchors_per_level']
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def padded_resize(self, img):
        """
        图像预处理：保持宽高比缩放并填充灰色
        """
        h, w = img.shape[:2]
        t_w, t_h = self.img_size
        
        # 计算缩放比例
        scale_w = w / t_w
        scale_h = h / t_h
        scale = max(scale_w, scale_h)  # 取最长边相对原图的比例
        
        # 计算缩放后尺寸
        new_w = int(w / scale)
        new_h = int(h / scale)
        
        # 缩放图像
        resized = cv2.resize(img, (new_w, new_h))
        
        # 确定填充位置
        if new_h > new_w:
            x_start = (t_w - new_w) // 2
            y_start = 0
        else:
            x_start = 0
            y_start = (t_h - new_h) // 2
        
        # 创建填充图像
        padded_img = np.full((t_h, t_w, 3), 114, dtype=np.uint8)
        padded_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized
        
        # 记录变换信息用于后续坐标映射
        transform_info = {
            'scale': scale,
            'x_start': x_start,
            'y_start': y_start,
            'orig_size': (w, h)
        }
        
        return padded_img, transform_info

    def generate_anchors(self):
        """
        生成锚框（与训练时相同）
        """
        anchors = []
        for i, (fm_w, fm_h) in enumerate(self.feature_map_sizes):
            stride_w = self.img_size[0] / fm_w
            stride_h = self.img_size[1] / fm_h
            
            for y in range(fm_h):
                for x in range(fm_w):
                    cx = (x + 0.5) * stride_w
                    cy = (y + 0.5) * stride_h
                    
                    # 当前特征图位置的锚框
                    for anchor_idx in range(self.clustered_anchors.shape[0]):
                        w, h = self.clustered_anchors[anchor_idx]
                        anchors.append([cx, cy, w, h])
        
        return torch.tensor(anchors, device=self.device)

    def apply_offsets(self, anchors, offsets):
        """
        将预测的偏移量应用到锚框上，得到预测边界框
        """
        # 将锚框转换为 (cx, cy, w, h)
        cx = anchors[:, 0]
        cy = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]
        
        # 解析偏移量
        dx = offsets[:, 0]
        dy = offsets[:, 1]
        dw = offsets[:, 2]
        dh = offsets[:, 3]
        
        # 应用偏移量
        pred_cx = dx * w + cx
        pred_cy = dy * h + cy
        pred_w = w * torch.exp(dw)
        pred_h = h * torch.exp(dh)
        
        # 转换回边界框坐标
        xmin = pred_cx - pred_w / 2
        ymin = pred_cy - pred_h / 2
        xmax = pred_cx + pred_w / 2
        ymax = pred_cy + pred_h / 2
        
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    def decode_predictions(self, loc_preds, conf_preds, anchors, conf_thresh=0.5, iou_thresh=0.5):
        """
        解码模型预测结果
        """
        # 应用softmax获取类别概率
        conf_probs = torch.softmax(conf_preds, dim=2)
        
        # 获取最可能的类别和置信度
        max_conf, max_idx = torch.max(conf_probs[:, :, 1:], dim=2)  # 跳过背景类
        max_idx += 1  # 调整索引以包含背景类
        
        # 解码边界框
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(loc_preds.shape[1]):  # 遍历所有锚框
            # 跳过低置信度预测
            if max_conf[0, i] < conf_thresh:
                continue
                
            # 解码边界框坐标
            cx = anchors[i, 0] + anchors[i, 2] * loc_preds[0, i, 0]
            cy = anchors[i, 1] + anchors[i, 3] * loc_preds[0, i, 1]
            w = anchors[i, 2] * torch.exp(loc_preds[0, i, 2])
            h = anchors[i, 3] * torch.exp(loc_preds[0, i, 3])
            
            # 转换为(xmin, ymin, xmax, ymax)格式
            xmin = cx - w / 2
            ymin = cy - h / 2
            xmax = cx + w / 2
            ymax = cy + h / 2
            
            boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            scores.append(max_conf[0, i].item())
            class_ids.append(max_idx[0, i].item() - 1)  # 转换为0-based索引
        
        # 应用非极大值抑制(NMS)
        if len(boxes) == 0:
            return [], [], []
            
        boxes = torch.tensor(boxes, device=self.device)
        scores = torch.tensor(scores, device=self.device)
        class_ids = torch.tensor(class_ids, device=self.device)
        
        # 将数据移动到CPU执行NMS（避免CUDA问题）
        keep_idx = torchvision.ops.nms(boxes.cpu(), scores.cpu(), iou_thresh)
        keep_idx = keep_idx.to(self.device)
        
        return boxes[keep_idx].cpu().numpy(), scores[keep_idx].cpu().numpy(), class_ids[keep_idx].cpu().numpy()

    def map_to_original(self, boxes, transform_info):
        """
        将边界框映射回原始图像坐标
        """
        scale = transform_info['scale']
        x_start = transform_info['x_start']
        y_start = transform_info['y_start']
        orig_w, orig_h = transform_info['orig_size']
        
        mapped_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            
            # 移除填充并缩放回原始尺寸
            xmin = (xmin - x_start) * scale
            ymin = (ymin - y_start) * scale
            xmax = (xmax - x_start) * scale
            ymax = (ymax - y_start) * scale
            
            # 确保边界框在图像范围内
            xmin = max(0, min(xmin, orig_w))
            ymin = max(0, min(ymin, orig_h))
            xmax = max(0, min(xmax, orig_w))
            ymax = max(0, min(ymax, orig_h))
            
            mapped_boxes.append([xmin, ymin, xmax, ymax])
            
        return np.array(mapped_boxes)

    def visualize_detections(self, orig_img, boxes, scores, class_ids, class_names, save_path=None):
        # 转换BGR为RGB格式
        image_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # 增强字体加载方案
        font_paths = [
            "simhei.ttf",  # 项目目录
            "C:/Windows/Fonts/simhei.ttf",  # Windows系统字体
            "/System/Library/Fonts/PingFang.ttc"  # Mac系统字体
        ]
        font = None
        for path in font_paths:
            try:
                # 动态字体大小：根据图像高度调整
                font_size = max(12, int(orig_img.shape[0] / 40))
                font = ImageFont.truetype(path, font_size, encoding="utf-8")
                break
            except:
                continue
        if font is None:
            print("警告：未找到中文字体，使用默认字体")
            font = ImageFont.load_default()
        
        for i, box in enumerate(boxes):
            class_id = int(class_ids[i])
            class_name = class_names.get(class_id, f"未知{class_id}")
            score = scores[i]
            xmin, ymin, xmax, ymax = box
            
            # 生成带置信度的标签文本
            label = f"{class_name} {score:.2f}"
            
            # 获取文本尺寸
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 边界框绘制 (PIL使用RGB)
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)],
                outline=(255, 0, 0),  # RGB红色
                width=2
            )
            
            # 智能标签位置计算（防止超出图像边界）
            text_x = max(0, min(xmin, orig_img.shape[1] - text_width))
            
            # 垂直位置：优先上方，空间不足则放框内
            if ymin - text_height - 5 > 0:  # 上方有足够空间
                text_y = ymin - text_height - 5
                bg_y1 = text_y
                bg_y2 = ymin
            else:  # 放在框内顶部
                text_y = ymin + 5
                bg_y1 = ymin
                bg_y2 = ymin + text_height + 10
            
            # 文本背景（带透明度效果）
            # bg_alpha = Image.new('RGBA', (text_width, bg_y2 - bg_y1), (255, 0, 0, 200))
            bg_width = int(round(text_width))
            bg_height = int(round(bg_y2 - bg_y1))
            bg_alpha = Image.new('RGBA', (bg_width, bg_height), (255, 0, 0, 200))
            pil_image.paste(bg_alpha, (int(text_x), int(bg_y1)), bg_alpha)
            
            # 绘制文本
            draw.text(
                (text_x, text_y),
                label,
                font=font,
                fill=(255, 255, 255)  # 白色文本
            )
        
        # 转换回OpenCV格式
        final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 保存或显示结果
        if save_path:
            cv2.imwrite(save_path, final_image)
        else:
            cv2.namedWindow('检测结果', cv2.WINDOW_NORMAL)
            cv2.imshow('检测结果', final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return final_image

    def detect(self, img_path, conf_thresh=0.5, iou_thresh=0.5, visualize=True, save_dir=None):
        """
        执行完整检测流程
        """
        # 新增文件检查
        if not os.path.exists(img_path):
            print(f"错误：文件 '{img_path}' 不存在")
            return []  # 返回空列表而非None

        # 1. 读取原始图像
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print(f"错误：无法读取图像 {img_path}")
            return []  # 关键修改：返回空列表
        
        # 2. 图像预处理
        processed_img, transform_info = self.padded_resize(orig_img)
        
        # 3. 转换为模型输入格式
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
        tensor_img = tensor_img.unsqueeze(0).to(self.device)  # 添加batch维度
        
        # 4. 生成锚框
        anchors = self.generate_anchors()
        
        # 5. 模型推理
        with torch.no_grad():
            loc_preds, conf_preds = self.model(tensor_img)
        
        # 6. 解码预测结果
        boxes, scores, class_ids = self.decode_predictions(
            loc_preds, conf_preds, anchors, conf_thresh, iou_thresh
        )
        
        # 7. 映射回原始图像坐标
        if len(boxes) > 0:
            boxes = self.map_to_original(boxes, transform_info)
        
        # 8. 使用新可视化函数
        if visualize:
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                img_name = os.path.basename(img_path)
                save_path = os.path.join(save_dir, f"det_{img_name}")
            
            # 调用新可视化函数
            self.visualize_detections(
                orig_img=orig_img,
                boxes=boxes,
                scores=scores,
                class_ids=class_ids,
                class_names=self.id_to_class_name,  # 确保有此属性
                save_path=save_path
            )
        
        # 9. 返回结构化结果
        results = []
        for i in range(len(boxes)):
            results.append({
                'bbox': boxes[i].tolist(),
                'score': scores[i],
                'class_id': class_ids[i],
                'class_name': self.class_names[class_ids[i]]
            })
        
        return results
    


# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     yelan = Yelan.Yelan()
#     img2object_list = yelan.parse_xml()

#     # 加载保存的数据集划分
#     with open('./Dataset/Yelan/data_splits.pkl', 'rb') as f:
#         data_split_info = pickle.load(f)

#     train_data = data_split_info['train_data']
#     val_data = data_split_info['val_data']
#     test_data = data_split_info['test_data']
#     # test_data = test_data[:1]  # 仅保留前10个测试样本
#     class_info = data_split_info['class_info']
    
#     # 数据加载器（处理锚框数量不均衡问题）
#     def collate_fn(batch):
#         return {
#             'image': torch.stack([x['image'] for x in batch], dim=0),
#             'anchors': [x['anchors'] for x in batch],
#             'labels': [x['labels'] for x in batch],
#             'offsets': [x['offsets'] for x in batch],
#             'pos_mask': [x['pos_mask'] for x in batch],
#             'image_id': [x['image_id'] for x in batch]  # 添加图像ID
#         }

    
#     # ====== 测试集评估 ======
#     print("\n开始测试集评估...")
#     # 准备真实标注数据
#     ground_truths = {}
#     for item in test_data:
#         img_id = item['source_image_id']
#         gt_boxes = []
#         for box in item['Boxes']:
#             gt_boxes.append({
#                 'class_id': yelan.class_name_to_id[box['class_name']],
#                 'xmin': box['xyxy'][0],
#                 'ymin': box['xyxy'][1],
#                 'xmax': box['xyxy'][2],
#                 'ymax': box['xyxy'][3]
#             })
#         ground_truths[img_id] = gt_boxes

#     # 实例化模型并加载权重
#     num_classes = len(yelan.class_name_to_id)
#     model_path = './SaveModels/best_ssd_model.pt'  # 确保路径正确

#     # 安全加载模型
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False) 
#     clustered_anchors = checkpoint['clustered_anchors']
#     num_anchors_per_level = checkpoint['num_anchors_per_level']
#     feature_map_sizes = checkpoint['feature_map_sizes']

#     model = SSD(
#         num_classes=checkpoint['num_classes'],
#         num_anchors_per_level=num_anchors_per_level
#     ).to(device)
    
#     # 创建测试数据加载器
#     test_dataset = DetectionDataset(test_data)
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=32,  # 根据GPU内存调整
#         shuffle=False,
#         collate_fn=collate_fn
#     )

#     print("\n开始测试集评估...")

#     # 计算mAP50
#     map50, class_metrics = evaluate_map50(
#         model, 
#         test_loader, 
#         ground_truths, 
#         class_info, 
#         device,
#         iou_threshold=0.5
#     )

#     # 打印结果
#     print(f"\n测试集 mAP@0.5: {map50:.4f}")
#     print("\n各类别详细指标:")
#     for class_id, metrics in class_metrics.items():
#         class_name = yelan.id_to_class_name[class_id]
#         print(f"  {class_name}: AP={metrics['ap']:.4f}, Precision={metrics['precision']:.4f}, "
#             f"Recall={metrics['recall']:.4f}, 预测数={metrics['num_pred']}, 真实数={metrics['num_gt']}")

#     # 计算整体精确率
#     total_tp = sum(m['ap'] * m['num_gt'] for m in class_metrics.values())
#     total_pred = sum(m['num_pred'] for m in class_metrics.values())
#     overall_precision = total_tp / total_pred if total_pred > 0 else 0

#     print(f"\n整体精确率: {overall_precision:.4f}")


if __name__ == '__main__':
    # 1. 初始化参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './SaveModels/best_ssd_model.pt'  # 模型路径
    img_folder = './Dataset/Yelan/images'  # 图像文件夹
    save_dir = './PredictOutput/Yelan'  # 结果保存目录
    
    # 2. 加载类别信息（从您的Yelan类中获取）
    yelan = Yelan.Yelan()
    class_name_to_id = yelan.class_name_to_id
    id_to_class_name = {v: k for k, v in class_name_to_id.items()}
    class_names = [id_to_class_name[i] for i in range(1, len(class_name_to_id)+1)]
    
    # 3. 加载模型配置（聚类锚框和特征图尺寸）
    checkpoint = torch.load(model_path, map_location=device)
    clustered_anchors = checkpoint['clustered_anchors']
    feature_map_sizes = checkpoint['feature_map_sizes']
    
    # 4. 初始化检测器
    detector = YOLOv5Inference(
        model_path=model_path,
        class_names=class_names,
        clustered_anchors=clustered_anchors,
        feature_map_sizes=feature_map_sizes,
        device=device
    )
    
    # 5. 批量推理
    img_list = glob.glob(os.path.join(img_folder, '*.jpg'))
    print(f'开始进行批量推理，共 {len(img_list)} 张图片')
    
    # 只处理前10张图片（根据需求调整）
    # img_list = img_list[:10]
    
    for img_path in tqdm(img_list, desc="推理进度"):
        # 执行检测
        results = detector.detect(
            img_path, 
            conf_thresh=0.75,  # 置信度阈值
            iou_thresh=0.25,   # NMS的IOU阈值
            visualize=True,   # 可视化结果
            save_dir=save_dir # 结果保存目录
        )
        
        # 打印结果
        img_name = os.path.basename(img_path)
        print(f"\n图像 {img_name} 检测结果:")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['class_name']} (置信度: {result['score']:.2f})")
            print(f"    边界框: {result['bbox']}")
    