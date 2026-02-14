import cv2
import random
import math
import numpy as np


def padded_resize(img, img2object, target_size=(640, 640)):
    """
    将图像得一边变形到目标尺寸，然后剩下的通过填充达到目标尺寸
    同步修改真实边界框的坐标
    target_size: 目标尺寸（t_w, t_h）
    img: 输入图像
    """ 
    # 计算动态缩放比例
    img_h, img_w = img.shape[:2]
    t_w, t_h = target_size
    scale_w = img_w / t_w
    scale_h = img_h / t_h
    scale = max(scale_w, scale_h)  # 取最长边相对原图的比例
    
    # 按比例缩放裁剪区域
    new_w = int(img_w // scale)
    new_h = int(img_h // scale)
    # print('new_size:', new_w, new_h)
    resized = cv2.resize(img, (new_w, new_h))
    if new_h > new_w:
        x_start = int((t_w - new_w) // 2)
        y_start = 0
    else:
        x_start = 0
        y_start = int((t_h - new_h) // 2)
    
    # 8. 灰度填充至原图尺寸
    padded_img = np.full((t_h, t_w, 3), 114, dtype=np.uint8)
    padded_img[y_start:y_start+new_h, x_start:x_start+new_w] = resized

    Boxes = []
    for obj in img2object:
        class_name = obj["class_name"]
        xmin, ymin, xmax, ymax = obj["xyxy"]
        
        # 根据填充变换边界框坐标
        xmin_padded = int(xmin // scale + x_start)
        ymin_padded = int(ymin // scale + y_start)
        xmax_padded = int(xmax // scale + x_start)
        ymax_padded = int(ymax // scale + y_start)
        Boxes.append({'class_name': class_name,'xyxy':[xmin_padded, ymin_padded, xmax_padded, ymax_padded]})
    
    return {'img': padded_img, 'Boxes': Boxes}


# 翻转
def img_flip(objects, flip_type=0):
    """
    水平翻转（horizontal）是0
    垂直翻转（vertical）是1
    水平和垂直翻转（Both）是-1
    """
    new_objects = {
        "img": objects["img"].copy(),  # 图像深拷贝
        "Boxes": [box.copy() for box in objects["Boxes"]],  # 边界框列表深拷贝
        "source_image_id": objects["source_image_id"]
    }
    img = new_objects['img']
    Boxes = new_objects['Boxes']
    h, w = img.shape[:2]  # 图像高度和宽度
    if flip_type == 0:
        # 水平翻转（左右翻转）
        img = cv2.flip(img, 1)
    if flip_type == 1:
        # 垂直翻转（上下翻转）
        img = cv2.flip(img, 0)
    if flip_type == -1:
        # 水平和垂直翻转
        img = cv2.flip(img, -1)

    # 处理边界框翻转
    new_Boxes = []
    for box in Boxes:
        class_name = box['class_name']
        x1, y1, x2, y2 = box['xyxy']
        assert 0 <= x1 <= 640 and 0 <= x2 <= 640, "无效边界框坐标"
        assert 0 <= y1 <= 640 and 0 <= y2 <= 640, "无效边界框坐标"
        assert x1 < x2 and y1 < y2, "无效边界框坐标"
        if flip_type == 0:  # 水平翻转
            new_x1 = (w - 1) - x2
            new_x2 = (w - 1) - x1
            new_y1, new_y2 = y1, y2
        elif flip_type == 1:  # 垂直翻转
            new_x1, new_x2 = x1, x2
            new_y1 = (h - 1) - y2
            new_y2 = (h - 1) - y1
        elif flip_type == -1:  # 同时翻转
            new_x1 = (w - 1) - x2
            new_x2 = (w - 1) - x1
            new_y1 = (h - 1) - y2
            new_y2 = (h - 1) - y1
        new_Boxes.append({'class_name': class_name, 'xyxy':[int(new_x1), int(new_y1), int(new_x2), int(new_y2)]})

    # 更新并返回结果
    new_objects['img'] = img
    new_objects['Boxes'] = new_Boxes
    return new_objects

# 裁剪
"""
随机裁剪出一块面积为圆面积10%-100%的区域
且该区域的宽和高之比随机取自0.5-2
然后再将该区域较长的一边比上原图的宽高，取最大的比例作为缩放比例
然后用该比例对裁剪后的区域进行缩放
对缩放后的图片用灰度值114进行填充，使得填充后的图片宽高等于原图的宽高
"""
def custom_crop_augmentation(img):
    """
    实现包含随机裁剪、动态缩放和灰度填充的数据增强逻辑
    参数：img为输入BGR格式图像（H,W,C）
    返回：增强后的BGR格式图像
    """
    h, w = img.shape[:2]
    
    # 1. 随机生成裁剪区域面积比例和宽高比
    area_ratio = random.uniform(0.1, 1.0)  # 面积占比10%-100%
    aspect_ratio = random.uniform(0.5, 2.0)  # 宽高比0.5-2
    
    # 2. 计算裁剪区域宽高（基于面积和宽高比）
    crop_area = area_ratio * (w * h)
    crop_h = int(math.sqrt(crop_area / aspect_ratio))
    crop_w = int(crop_h * aspect_ratio)
    
    # 3. 调整裁剪尺寸不超过原图边界
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    
    # 4. 随机选择裁剪起点
    x = random.randint(0, max(0, w - crop_w))
    y = random.randint(0, max(0, h - crop_h))
    
    # 5. 执行裁剪
    cropped = img[y:y+crop_h, x:x+crop_w]
    
    # 6. 计算动态缩放比例
    scale_w = crop_w / w
    scale_h = crop_h / h
    scale = max(scale_w, scale_h)  # 取最长边相对原图的比例
    
    # 7. 按比例缩放裁剪区域
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized = cv2.resize(cropped, (new_w, new_h))
    
    # 8. 灰度填充至原图尺寸
    padded = np.full((h, w, 3), 114, dtype=np.uint8)
    y_start = (h - new_h) // 2
    x_start = (w - new_w) // 2
    padded[y_start:y_start+new_h, x_start:x_start+new_w] = resized
    
    return padded

# 颜色变化
def color_augment(img):
    """
    亮度，对比度， 饱和度， 色相（修正版本）
    """
    img = img.astype(np.float32)
    
    # 亮度调整（±50）
    delta = np.random.uniform(-50, 50)
    img += delta
    img = np.clip(img, 0, 255)
    
    # 转换为HSV空间
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 饱和度调整（±50%）
    saturation_scale = np.random.uniform(0.5, 1.5)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    
    # 对比度调整（±50%）
    contrast_scale = np.random.uniform(0.5, 1.5)
    img = np.clip((img - 127.5) * contrast_scale + 127.5, 0, 255)
    
    # 色相调整（±30度）
    hue_delta = np.random.uniform(-30, 30)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180  # OpenCV的H范围是0-179
    
    # 转换回BGR
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return img


if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('cat.jpg')

    # 查看图像宽高
    h, w = img.shape[:2]
    print('h:', h)
    print('w:', w)
    # 执行水平翻转（上下翻转）并查看图像
    flipped = cv2.flip(img, 0)
    cv2.imshow('horizontal_flipped', flipped)
    # 执行垂直翻转（左右翻转）并查看图像
    flipped = cv2.flip(img, 1)
    cv2.imshow('vertical_flipped', flipped)
    # 执行水平垂直翻转并查看图像
    flipped = cv2.flip(img, -1)
    cv2.imshow('horizontal_vertical_flipped', flipped)
    # 执行2次图像颜色变化
    for i in range(2):
        img = color_augment(img)
        cv2.imshow(f'color_augment_{i}', img)

    # 执行2次随机裁剪并查看图像
    for i in range(2):
        cropped = custom_crop_augmentation(img)
        cv2.imshow(f'cropped_{i}', cropped)
    cv2.waitKey(0)

