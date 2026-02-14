import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_bboxes(image, Boxes):
    """可视化XML文件中的边界框"""
    if image is None:
        print("错误：无法加载图像文件")
        return

    # 转换BGR为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    for box in Boxes:
        class_name = box['class_name']
        xmin, ymin, xmax, ymax = box['xyxy']
                
        # 使用PIL绘制中文标签
        font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")  # 黑体字体
        # 使用textbbox获取尺寸
        left, top, right, bottom = draw.textbbox((0, 0), class_name, font=font)
        text_width = right - left
        text_height = bottom - top
        # 使用PIL绘制边界框
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=(0, 255, 0),  # RGB格式绿色
            width=2
        )
        # 绘制文本
        draw.text(
            (xmin, ymin - text_height - 2),  # 文本起始位置
            class_name,
            font=font,
            fill=(0, 255, 0)  # 黑色字体
        )

    # 转换回OpenCV显示
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    cv2.namedWindow('Annotation Viewer', cv2.WINDOW_NORMAL)
    cv2.imshow('Annotation Viewer', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_anchors(img, anchors):
    """可视化锚框分布"""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 将张量转到CPU并转换为numpy
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    
    for box in anchors:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           linewidth=1, edgecolor='r', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
    
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_matched_anchors(img, anchors, gt_boxes, positive_mask):
    """可视化匹配的正样本锚框"""
    # 转换颜色通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12,8))
    ax = plt.gca()

    # 转换张量到CPU
    if isinstance(anchors, torch.Tensor):
        anchors = anchors.cpu().numpy()
    if isinstance(positive_mask, torch.Tensor):
        positive_mask = positive_mask.cpu().numpy()
    
    # 绘制所有锚框（半透明红色）
    for box in anchors:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           linewidth=0.5, edgecolor='r', facecolor='none', alpha=0.1)
        ax.add_patch(rect)
        
    # 绘制正样本锚框（实心绿色）
    positive_anchors = anchors[positive_mask]
    for box in positive_anchors:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           linewidth=1, edgecolor='g', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
    # 绘制真实框（蓝色）
    for box in gt_boxes:
        xmin, ymin, xmax, ymax = box['xyxy']
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                           linewidth=1.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()