import torch
import os
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
from collections import OrderedDict

class Yelan:
    def __init__(self, root='./Dataset/Yelan', batch_size=64, val_ratio=0.2, 
                 transform=None, seed=42):
        self.images_folder = os.path.join(root, "images")
        self.labels_folder = os.path.join(root, "labels")

        # 类名字典初始化
        self.class_names = self._get_all_class_names()
        self.class_name_to_id = self._build_class_mapping()
        # 反向映射：ID → 类别名称
        self.id_to_class_name = self._build_reverse_mapping()

    def _get_all_class_names(self):
        """遍历所有XML文件收集所有类别名称"""
        class_names = set()
        for xml_file in os.listdir(self.labels_folder):
            xml_path = os.path.join(self.labels_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall("object"):
                class_name = obj.find("name").text.strip()
                class_names.add(class_name)
        
        # 按字母顺序排序确保一致性
        return sorted(class_names)

    def _build_class_mapping(self):
        """构建类别名称到ID的映射字典"""
        # 背景类ID固定为0，其他类别从1开始
        return OrderedDict(
            {name: idx + 1 for idx, name in enumerate(self.class_names)}
        )
    
    def _build_reverse_mapping(self):
        """构建ID到类别名称的反向映射"""
        return {v: k for k, v in self.class_name_to_id.items()}

    def parse_xml(self):
        # 读取图像和标签的路径
        img2xml_path_dict = {}
        for xml_file in os.listdir(self.labels_folder):
            img_name = xml_file.replace("xml", "jpg")
            img_path = os.path.join(self.images_folder, img_name)
            xml_path = os.path.join(self.labels_folder, xml_file)
            img2xml_path_dict[img_path] = xml_path

        # 读取标签信息和图像内容，构建object_list
        img2object_list = []
        for img_path, xml_path in img2xml_path_dict.items():
            # 提取标签基本信息
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)
            
            # 校验文件存在性
            if not os.path.exists(img_path):
                print(f"警告：图像文件 {img_path} 不存在，已跳过")
                continue
                
            try:
                # 加载图像并校验结果
                image_data = np.fromfile(img_path, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("图像解码失败")
            except Exception as e:
                print(f"错误：加载 {img_path} 失败，原因：{str(e)}")
                continue

            # 提取所有目标对象
            objects = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # 添加类别ID
                objects.append({
                    "class_name": name,
                    "class_id": self.class_name_to_id[name],
                    "xyxy": [xmin, ymin, xmax, ymax]
                })

            # 添加source_image_id（使用原始文件名作为唯一标识）
            source_id = os.path.splitext(os.path.basename(img_path))[0]  
            img2object_list.append({
                "img": image,
                "size": (width, height),
                "Boxes": objects,
                "source_image_id": source_id
                
            })

        return img2object_list

