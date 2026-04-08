#!/usr/bin/env python3
"""
文本描述到SAM2 Prompt转换系统
专门用于气胸X光图像的自动分割

主要功能：
1. 解析文本描述中的关键信息
2. 生成多种类型的SAM2提示
3. 支持坐标、点、框等多种提示方式
4. 提供形状引导的智能提示生成
"""

import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
from dataclasses import dataclass


@dataclass
class PneumothoraxInfo:
    """气胸信息结构"""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    position: str  # 解剖学位置描述
    shape: str  # 形状描述
    size: str  # 大小描述
    area: int  # 像素面积
    center: Tuple[int, int]  # 中心点
    # 新增字段
    position_code: str = ""  # 位置编码 (A-F)
    lung_side: str = ""  # 肺侧 (左肺/右肺)
    vertical_position: str = ""  # 垂直位置 (上/中/下)
    has_structured_data: bool = False  # 是否来自结构化数据


class TextToSAM2Prompt:
    """文本描述到SAM2提示的转换器"""

    def __init__(self):
        # position_code到相对坐标的映射
        self.position_code_mapping = {
            # A区域：左上肺
            "A": (0.55, 0.1, 0.9, 0.5),
            # B区域：左中肺
            "B": (0.55, 0.3, 0.9, 0.7),
            # C区域：左下肺
            "C": (0.55, 0.5, 0.9, 0.9),
            # D区域：右上肺
            "D": (0.1, 0.1, 0.45, 0.5),
            # E区域：右中肺
            "E": (0.1, 0.3, 0.45, 0.7),
            # F区域：右下肺
            "F": (0.1, 0.5, 0.45, 0.9),
            # 扩展区域
            "LEFT": (0.55, 0.1, 0.9, 0.9),  # 整个左肺
            "RIGHT": (0.1, 0.1, 0.45, 0.9),  # 整个右肺
        }

        # 形状到点分布策略的映射
        self.shape_strategies = {
            "水平条带状": self._generate_horizontal_points,
            "垂直条带状": self._generate_vertical_points,
            "半包裹型条带状": self._generate_curved_points,
            "不规则状": self._generate_scattered_points,
            "团状": self._generate_clustered_points,
            "块状": self._generate_block_points,
        }

    def parse_text_description(self, data: Dict) -> Optional[PneumothoraxInfo]:
        """解析结构化数据，提取关键信息

        Args:
            data: 包含结构化数据的字典

        Returns:
            PneumothoraxInfo对象或None
        """
        try:
            return self._parse_structured_data(data)
        except Exception as e:
            print(f"解析描述时出错: {e}")
            return None

    def _parse_structured_data(self, data: Dict) -> Optional[PneumothoraxInfo]:
        """解析新格式的结构化数据"""
        try:
            # 检查是否存在气胸
            if not data.get("has_pneu", False):
                return None

            pneu_location = data.get("pneu_location", {})
            if not pneu_location:
                return None

            # 提取边界框信息
            bbox_data = pneu_location.get("bounding_box", {})
            if not bbox_data:
                return None

            x1 = bbox_data.get("x", 0)
            y1 = bbox_data.get("y", 0)
            x2 = bbox_data.get("x2", x1 + bbox_data.get("width", 0))
            y2 = bbox_data.get("y2", y1 + bbox_data.get("height", 0))

            bbox = (x1, y1, x2, y2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            area = (x2 - x1) * (y2 - y1)

            # 提取位置信息
            position = pneu_location.get("full_position", "未知位置")
            lung_side = pneu_location.get("lung_side", "")
            vertical_pos = pneu_location.get("vertical_position", "")
            position_code = pneu_location.get("position_code", "")

            # 标准化位置描述
            if lung_side and vertical_pos:
                position = f"{lung_side}{vertical_pos}"

            # 从描述文本中提取形状信息
            caption = data.get("caption", "")
            shape = self._extract_shape_from_text(caption)
            size = self._extract_size_from_text(caption)

            return PneumothoraxInfo(
                bbox=bbox,
                position=position,
                shape=shape,
                size=size,
                area=area,
                center=center,
                position_code=position_code,
                lung_side=lung_side,
                vertical_position=vertical_pos,
                has_structured_data=True,
            )

        except Exception as e:
            print(f"解析结构化数据时出错: {e}")
            return None

    def generate_bbox_prompt(self, pneumo_info: PneumothoraxInfo) -> Dict:
        """生成边界框提示（最直接有效）"""
        return {
            "type": "bbox",
            "coordinates": list(pneumo_info.bbox),
            "label": 1,  # 前景
            "confidence": 0.9,
        }

    def generate_enhanced_bbox_constrained_points(
        self,
        pneumo_info: PneumothoraxInfo,
        image: np.ndarray = None,
        num_points: int = 8,
    ) -> Dict:
        """使用新格式数据生成增强的边界框约束点"""

        # 如果有结构化数据，使用position_code优化点生成
        if pneumo_info.has_structured_data and pneumo_info.position_code:
            positive_points = self._generate_position_code_aware_points(
                pneumo_info, num_points // 2
            )
            negative_points = self._generate_anatomical_aware_negatives(
                pneumo_info, num_points // 2
            )
        else:
            # 回退到原有方法
            positive_points = self._generate_grid_points_in_bbox(pneumo_info, 2, 2)
            negative_points = self._generate_bbox_edge_negatives(pneumo_info)

        # 如果提供了图像，进行内容优化
        if image is not None:
            positive_points = self._optimize_points_by_image_content(
                positive_points, pneumo_info, image
            )

        # 确保点在边界框内
        positive_points = self._ensure_points_in_bbox(positive_points, pneumo_info.bbox)

        # 去重
        positive_points = self._deduplicate_points(positive_points)
        negative_points = self._deduplicate_points(negative_points)

        return {
            "positive_points": positive_points[: num_points // 2],
            "negative_points": negative_points[: num_points // 2],
            "bbox": pneumo_info.bbox,
            "metadata": {
                "position_code": pneumo_info.position_code,
                "lung_side": pneumo_info.lung_side,
                "vertical_position": pneumo_info.vertical_position,
                "has_structured_data": pneumo_info.has_structured_data,
                "method": "enhanced_bbox_constrained_points",
            },
        }

    def _generate_position_code_aware_points(
        self, pneumo_info: PneumothoraxInfo, num_points: int
    ) -> List[Tuple[int, int]]:
        """基于position_code生成位置感知的提示点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        points = [pneumo_info.center]  # 始终包含中心点

        # 根据position_code调整点分布策略
        if pneumo_info.position_code in ["A", "D"]:  # 上肺区域
            # 上肺区域：更多点分布在上半部分
            for i in range(1, num_points):
                offset_x = (i - num_points // 2) * (x2 - x1) // (num_points * 2)
                offset_y = -(i % 2 + 1) * (y2 - y1) // 6  # 向上偏移
                point = (center_x + offset_x, max(y1 + 5, center_y + offset_y))
                points.append(point)

        elif pneumo_info.position_code in ["C", "F"]:  # 下肺区域
            # 下肺区域：更多点分布在下半部分
            for i in range(1, num_points):
                offset_x = (i - num_points // 2) * (x2 - x1) // (num_points * 2)
                offset_y = (i % 2 + 1) * (y2 - y1) // 6  # 向下偏移
                point = (center_x + offset_x, min(y2 - 5, center_y + offset_y))
                points.append(point)

        elif pneumo_info.position_code in ["B", "E"]:  # 中肺区域
            # 中肺区域：水平分布
            for i in range(1, num_points):
                offset_x = (i - num_points // 2) * (x2 - x1) // (num_points + 1)
                offset_y = (i % 3 - 1) * (y2 - y1) // 8  # 轻微垂直变化
                point = (center_x + offset_x, center_y + offset_y)
                points.append(point)

        else:
            # 默认策略：均匀分布
            for i in range(1, num_points):
                angle = 2 * np.pi * i / num_points
                radius_x = (x2 - x1) // 4
                radius_y = (y2 - y1) // 4
                point = (
                    int(center_x + radius_x * np.cos(angle)),
                    int(center_y + radius_y * np.sin(angle)),
                )
                points.append(point)

        return points[:num_points]

    def _generate_anatomical_aware_negatives(
        self, pneumo_info: PneumothoraxInfo, num_points: int
    ) -> List[Tuple[int, int]]:
        """基于解剖学知识生成负样本点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        image_height, image_width = 512, 512  # 假设标准尺寸，实际应从图像获取

        negative_points = []

        # 根据肺侧生成对侧负样本点
        if pneumo_info.lung_side == "左肺":
            # 在右肺区域生成负样本点
            right_lung_region = self.position_code_mapping.get(
                "RIGHT", (0.1, 0.1, 0.45, 0.9)
            )
            neg_x1 = int(right_lung_region[0] * image_width)
            neg_y1 = int(right_lung_region[1] * image_height)
            neg_x2 = int(right_lung_region[2] * image_width)
            neg_y2 = int(right_lung_region[3] * image_height)

            for i in range(num_points // 2):
                neg_x = neg_x1 + (neg_x2 - neg_x1) * (i + 1) // (num_points // 2 + 1)
                neg_y = neg_y1 + (neg_y2 - neg_y1) * (i + 1) // (num_points // 2 + 1)
                negative_points.append((neg_x, neg_y))

        elif pneumo_info.lung_side == "右肺":
            # 在左肺区域生成负样本点
            left_lung_region = self.position_code_mapping.get(
                "LEFT", (0.55, 0.1, 0.9, 0.9)
            )
            neg_x1 = int(left_lung_region[0] * image_width)
            neg_y1 = int(left_lung_region[1] * image_height)
            neg_x2 = int(left_lung_region[2] * image_width)
            neg_y2 = int(left_lung_region[3] * image_height)

            for i in range(num_points // 2):
                neg_x = neg_x1 + (neg_x2 - neg_x1) * (i + 1) // (num_points // 2 + 1)
                neg_y = neg_y1 + (neg_y2 - neg_y1) * (i + 1) // (num_points // 2 + 1)
                negative_points.append((neg_x, neg_y))

        # 在边界框周围生成额外的负样本点
        margin = 20
        bbox_negatives = [
            (x1 - margin, y1 - margin),  # 左上外
            (x2 + margin, y1 - margin),  # 右上外
            (x1 - margin, y2 + margin),  # 左下外
            (x2 + margin, y2 + margin),  # 右下外
        ]

        # 确保点在图像范围内
        bbox_negatives = [
            (max(0, min(image_width - 1, x)), max(0, min(image_height - 1, y)))
            for x, y in bbox_negatives
        ]

        negative_points.extend(bbox_negatives)

        return negative_points[:num_points]

    def generate_bbox_constrained_points(
        self,
        pneumo_info: PneumothoraxInfo,
        image: np.ndarray = None,
        num_points: int = 8,
    ) -> Dict:
        """
        生成边界框约束的正样本点
        将边界框作为正样本点的限制范围，在其内部智能选择点

        Args:
            pneumo_info: 气胸信息
            image: 原始图像（可选，用于基于内容的点选择）
            num_points: 生成的正样本点数量
        """
        x1, y1, x2, y2 = pneumo_info.bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # 基础策略：在边界框内均匀分布点
        points = []

        # 1. 中心点（最重要）
        center_x, center_y = pneumo_info.center
        points.append((center_x, center_y))

        # 2. 基于边界框大小的网格点
        if bbox_width > 50 and bbox_height > 50:
            # 大边界框：使用3x3网格
            grid_points = self._generate_grid_points_in_bbox(pneumo_info, 3, 3)
        elif bbox_width > 30 or bbox_height > 30:
            # 中等边界框：使用2x2网格
            grid_points = self._generate_grid_points_in_bbox(pneumo_info, 2, 2)
        else:
            # 小边界框：使用简单的四点布局
            grid_points = self._generate_simple_bbox_points(pneumo_info)

        points.extend(grid_points)

        # 3. 基于形状的特殊点
        shape_points = self._generate_shape_aware_points(pneumo_info)
        points.extend(shape_points)

        # 4. 如果提供了图像，基于图像内容优化点位置
        if image is not None:
            points = self._optimize_points_by_image_content(points, pneumo_info, image)

        # 5. 确保所有点都在边界框内
        points = self._ensure_points_in_bbox(points, pneumo_info.bbox)

        # 6. 去重并限制数量
        points = self._deduplicate_points(points)
        points = points[:num_points]

        return {
            "type": "bbox_constrained_points",
            "coordinates": points,
            "labels": [1] * len(points),  # 全部为正样本点
            "confidence": 0.9,
            "bbox_constraint": pneumo_info.bbox,
        }

    def _generate_grid_points_in_bbox(
        self, pneumo_info: PneumothoraxInfo, rows: int, cols: int
    ) -> List[Tuple[int, int]]:
        """在边界框内生成网格点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        points = []

        # 留出边距，避免点太靠近边界
        margin_x = max(5, (x2 - x1) * 0.1)
        margin_y = max(5, (y2 - y1) * 0.1)

        effective_x1 = x1 + margin_x
        effective_y1 = y1 + margin_y
        effective_x2 = x2 - margin_x
        effective_y2 = y2 - margin_y

        for i in range(rows):
            for j in range(cols):
                if rows == 1:
                    y = (effective_y1 + effective_y2) / 2
                else:
                    y = effective_y1 + i * (effective_y2 - effective_y1) / (rows - 1)

                if cols == 1:
                    x = (effective_x1 + effective_x2) / 2
                else:
                    x = effective_x1 + j * (effective_x2 - effective_x1) / (cols - 1)

                points.append((int(x), int(y)))

        return points

    def _generate_simple_bbox_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为小边界框生成简单的四点布局"""
        x1, y1, x2, y2 = pneumo_info.bbox

        # 四个象限的点
        quarter_x = (x2 - x1) / 4
        quarter_y = (y2 - y1) / 4

        points = [
            (int(x1 + quarter_x), int(y1 + quarter_y)),  # 左上象限
            (int(x2 - quarter_x), int(y1 + quarter_y)),  # 右上象限
            (int(x1 + quarter_x), int(y2 - quarter_y)),  # 左下象限
            (int(x2 - quarter_x), int(y2 - quarter_y)),  # 右下象限
        ]

        return points

    def _generate_shape_aware_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """基于气胸形状生成特殊点"""
        points = []
        x1, y1, x2, y2 = pneumo_info.bbox

        if "水平" in pneumo_info.shape or "条带" in pneumo_info.shape:
            # 水平条带状：在水平中线上添加点
            mid_y = (y1 + y2) / 2
            for i in range(3):
                x = x1 + (i + 1) * (x2 - x1) / 4
                points.append((int(x), int(mid_y)))

        elif "垂直" in pneumo_info.shape:
            # 垂直形状：在垂直中线上添加点
            mid_x = (x1 + x2) / 2
            for i in range(3):
                y = y1 + (i + 1) * (y2 - y1) / 4
                points.append((int(mid_x), int(y)))

        elif "圆形" in pneumo_info.shape or "椭圆" in pneumo_info.shape:
            # 圆形/椭圆：在边界上添加点
            center_x, center_y = pneumo_info.center
            radius_x = (x2 - x1) / 3
            radius_y = (y2 - y1) / 3

            for angle in [0, 90, 180, 270]:
                rad = np.radians(angle)
                x = center_x + radius_x * np.cos(rad)
                y = center_y + radius_y * np.sin(rad)
                points.append((int(x), int(y)))

        return points

    def _optimize_points_by_image_content(
        self,
        points: List[Tuple[int, int]],
        pneumo_info: PneumothoraxInfo,
        image: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """基于图像内容优化点位置"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            x1, y1, x2, y2 = pneumo_info.bbox

            # 提取边界框区域
            bbox_region = gray[y1:y2, x1:x2]

            # 计算梯度，寻找边缘
            grad_x = cv2.Sobel(bbox_region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(bbox_region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # 对每个点，尝试移动到附近的高梯度区域
            optimized_points = []
            for px, py in points:
                # 转换为边界框内的相对坐标
                rel_x = px - x1
                rel_y = py - y1

                # 确保在边界框内
                if (
                    0 <= rel_x < bbox_region.shape[1]
                    and 0 <= rel_y < bbox_region.shape[0]
                ):
                    # 在3x3邻域内寻找最高梯度点
                    search_radius = 3
                    best_x, best_y = rel_x, rel_y
                    best_gradient = gradient_magnitude[rel_y, rel_x]

                    for dy in range(-search_radius, search_radius + 1):
                        for dx in range(-search_radius, search_radius + 1):
                            new_x = rel_x + dx
                            new_y = rel_y + dy

                            if (
                                0 <= new_x < bbox_region.shape[1]
                                and 0 <= new_y < bbox_region.shape[0]
                            ):

                                grad_val = gradient_magnitude[new_y, new_x]
                                if grad_val > best_gradient:
                                    best_gradient = grad_val
                                    best_x, best_y = new_x, new_y

                    # 转换回全局坐标
                    optimized_points.append((best_x + x1, best_y + y1))
                else:
                    optimized_points.append((px, py))

            return optimized_points

        except Exception as e:
            # 如果优化失败，返回原始点
            print(f"基于图像内容的点优化失败: {e}")
            return points

    def _ensure_points_in_bbox(
        self, points: List[Tuple[int, int]], bbox: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int]]:
        """确保所有点都在边界框内"""
        x1, y1, x2, y2 = bbox
        constrained_points = []

        for px, py in points:
            # 将点约束在边界框内
            constrained_x = max(x1, min(x2 - 1, px))
            constrained_y = max(y1, min(y2 - 1, py))
            constrained_points.append((constrained_x, constrained_y))

        return constrained_points

    def _deduplicate_points(
        self, points: List[Tuple[int, int]], min_distance: int = 5
    ) -> List[Tuple[int, int]]:
        """去除过于接近的重复点"""
        if not points:
            return points

        unique_points = [points[0]]

        for px, py in points[1:]:
            is_duplicate = False
            for ux, uy in unique_points:
                distance = np.sqrt((px - ux) ** 2 + (py - uy) ** 2)
                if distance < min_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append((px, py))

        return unique_points

    def generate_point_prompts(
        self, pneumo_info: PneumothoraxInfo, num_points: int = 5
    ) -> Dict:
        """生成点提示"""
        x1, y1, x2, y2 = pneumo_info.bbox

        # 基础点：中心点 + 四个角点
        points = [
            pneumo_info.center,  # 中心点
            (x1 + 5, y1 + 5),  # 左上角（稍微内移）
            (x2 - 5, y1 + 5),  # 右上角
            (x1 + 5, y2 - 5),  # 左下角
            (x2 - 5, y2 - 5),  # 右下角
        ]

        # 根据形状添加额外的引导点
        if pneumo_info.shape in self.shape_strategies:
            extra_points = self.shape_strategies[pneumo_info.shape](pneumo_info)
            points.extend(extra_points[: num_points - 5])

        return {
            "type": "points",
            "coordinates": points[:num_points],
            "labels": [1] * len(points[:num_points]),  # 全部为前景点
            "confidence": 0.8,
        }

    def generate_negative_prompts(
        self, pneumo_info: PneumothoraxInfo, image_shape: Tuple[int, int]
    ) -> Dict:
        """生成负提示点（排除明显的非气胸区域）"""
        h, w = image_shape
        x1, y1, x2, y2 = pneumo_info.bbox

        # 在气胸区域外选择负提示点
        negative_points = []

        # 心脏区域（通常在中央偏左下）
        heart_region = [(w // 3, h * 2 // 3), (w // 2, h * 3 // 4)]

        # 脊柱区域（中央）
        spine_region = [(w // 2, h // 4), (w // 2, h // 2), (w // 2, h * 3 // 4)]

        # 肋骨区域（避开气胸区域的肋骨）
        rib_points = []
        if x1 > w // 2:  # 气胸在右侧，在左侧选择肋骨点
            rib_points = [(w // 4, h // 3), (w // 4, h // 2)]
        else:  # 气胸在左侧，在右侧选择肋骨点
            rib_points = [(w * 3 // 4, h // 3), (w * 3 // 4, h // 2)]

        negative_points.extend(heart_region)
        negative_points.extend(spine_region)
        negative_points.extend(rib_points)

        return {
            "type": "negative_points",
            "coordinates": negative_points,
            "labels": [0] * len(negative_points),  # 全部为背景点
            "confidence": 0.7,
        }

    def generate_region_based_negative_prompts(
        self,
        pneumo_info: PneumothoraxInfo,
        image_shape: Tuple[int, int],
        num_points_per_region: int = 8,
    ) -> Dict:
        """基于区域分割生成负样本点，更精确地约束SAM2分割范围"""
        h, w = image_shape
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # 将图像分为四个象限
        regions = {
            "top_left": (0, 0, w // 2, h // 2),
            "top_right": (w // 2, 0, w, h // 2),
            "bottom_left": (0, h // 2, w // 2, h),
            "bottom_right": (w // 2, h // 2, w, h),
        }

        # 根据气胸位置确定负样本区域
        pneumo_position = pneumo_info.position.lower()
        negative_regions = []

        if "左" in pneumo_position and "上" in pneumo_position:
            # 气胸在左上，其他三个区域为负样本
            negative_regions = ["top_right", "bottom_left", "bottom_right"]
        elif "左" in pneumo_position and "下" in pneumo_position:
            # 气胸在左下，其他三个区域为负样本
            negative_regions = ["top_left", "top_right", "bottom_right"]
        elif "右" in pneumo_position and "上" in pneumo_position:
            # 气胸在右上，其他三个区域为负样本
            negative_regions = ["top_left", "bottom_left", "bottom_right"]
        elif "右" in pneumo_position and "下" in pneumo_position:
            # 气胸在右下，其他三个区域为负样本
            negative_regions = ["top_left", "top_right", "bottom_left"]
        elif "左" in pneumo_position:
            # 气胸在左侧，右侧为负样本
            negative_regions = ["top_right", "bottom_right"]
        elif "右" in pneumo_position:
            # 气胸在右侧，左侧为负样本
            negative_regions = ["top_left", "bottom_left"]
        else:
            # 默认情况，选择远离气胸中心的区域
            if center_x < w // 2:
                negative_regions = ["top_right", "bottom_right"]
            else:
                negative_regions = ["top_left", "bottom_left"]

        # 在负样本区域生成点
        negative_points = []
        for region_name in negative_regions:
            rx1, ry1, rx2, ry2 = regions[region_name]

            # 在每个负样本区域均匀分布点
            for i in range(num_points_per_region):
                # 随机生成点，但避免太靠近边界
                margin = 20
                x = np.random.randint(rx1 + margin, max(rx1 + margin + 1, rx2 - margin))
                y = np.random.randint(ry1 + margin, max(ry1 + margin + 1, ry2 - margin))

                # 确保点不在气胸边界框内
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    negative_points.append((x, y))

        # 添加边界框外围的负样本点
        bbox_margin = 30
        boundary_points = [
            # 边界框上方
            (center_x, max(0, y1 - bbox_margin)),
            # 边界框下方
            (center_x, min(h - 1, y2 + bbox_margin)),
            # 边界框左侧
            (max(0, x1 - bbox_margin), center_y),
            # 边界框右侧
            (min(w - 1, x2 + bbox_margin), center_y),
        ]

        # 只添加有效的边界点
        for point in boundary_points:
            if 0 <= point[0] < w and 0 <= point[1] < h:
                negative_points.append(point)

        return {
            "type": "region_negative_points",
            "coordinates": negative_points,
            "labels": [0] * len(negative_points),
            "confidence": 0.9,
            "metadata": {
                "negative_regions": negative_regions,
                "total_points": len(negative_points),
            },
        }

    def generate_hybrid_prompt(
        self, pneumo_info: PneumothoraxInfo, image_shape: Tuple[int, int]
    ) -> Dict:
        """生成混合提示（推荐使用）"""
        bbox_prompt = self.generate_bbox_prompt(pneumo_info)
        point_prompt = self.generate_point_prompts(pneumo_info, num_points=3)
        negative_prompt = self.generate_negative_prompts(pneumo_info, image_shape)

        return {
            "type": "hybrid",
            "prompts": {
                "bbox": bbox_prompt["coordinates"],
                "positive_points": point_prompt["coordinates"],
                "negative_points": negative_prompt["coordinates"],
                "point_labels": [1] * len(point_prompt["coordinates"])
                + [0] * len(negative_prompt["coordinates"]),
            },
            "confidence": 0.85,
            "metadata": {
                "position": pneumo_info.position,
                "shape": pneumo_info.shape,
                "size": pneumo_info.size,
                "area": pneumo_info.area,
            },
        }

    def generate_enhanced_hybrid_prompt(
        self, pneumo_info: PneumothoraxInfo, image_shape: Tuple[int, int]
    ) -> Dict:
        """生成增强的混合提示（包含四象限负样本点改进）"""
        h, w = image_shape
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        # 1. 生成正样本点（基于形状的智能分布）
        positive_points = self._generate_shape_based_points(pneumo_info)

        # 2. 生成四象限负样本点（改进版）
        quadrant_negative = self.generate_region_based_negative_prompts(
            pneumo_info, image_shape, num_points_per_region=6
        )

        # 3. 添加边界框内部的精确负样本点
        internal_negative_points = self._generate_internal_negative_points(pneumo_info)

        # 合并所有负样本点
        all_negative_points = (
            quadrant_negative["coordinates"] + internal_negative_points
        )

        return {
            "type": "enhanced_hybrid",
            "prompts": {
                "positive_points": positive_points,
                "negative_points": all_negative_points,
                "bbox": pneumo_info.bbox,
                "point_labels": [1] * len(positive_points)
                + [0] * len(all_negative_points),
            },
            "confidence": 0.95,
            "metadata": {
                "position": pneumo_info.position,
                "shape": pneumo_info.shape,
                "size": pneumo_info.size,
                "area": pneumo_info.area,
                "method": "enhanced_hybrid_with_quadrant_negatives",
            },
        }

    def generate_precision_prompt(
        self, pneumo_info: PneumothoraxInfo, image_shape: Tuple[int, int]
    ) -> Dict:
        """生成精确分割提示，减少过度分割"""
        h, w = image_shape
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        # 1. 生成少量但精确的正样本点
        positive_points = self._generate_precise_positive_points(pneumo_info)

        # 2. 生成大量精确的负样本点
        negative_points = []

        # 2a. 边界框内部的负样本点（避开气胸中心区域）
        internal_negatives = self._generate_bbox_internal_negatives(pneumo_info)
        negative_points.extend(internal_negatives)

        # 2b. 边界框边缘的负样本点
        edge_negatives = self._generate_bbox_edge_negatives(pneumo_info)
        negative_points.extend(edge_negatives)

        # 2c. 胸廓结构相关的负样本点
        anatomical_negatives = self._generate_anatomical_negatives(
            pneumo_info, image_shape
        )
        negative_points.extend(anatomical_negatives)

        # 注意：不使用边界框提示，只使用点提示
        return {
            "type": "precision_point_only",
            "prompts": {
                "positive_points": positive_points,
                "negative_points": negative_points,
                "point_labels": [1] * len(positive_points) + [0] * len(negative_points),
                # 故意不包含bbox，减少过度分割
            },
            "confidence": 0.9,
            "metadata": {
                "position": pneumo_info.position,
                "shape": pneumo_info.shape,
                "size": pneumo_info.size,
                "area": pneumo_info.area,
                "method": "precision_point_only",
            },
        }

    def _generate_shape_based_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """基于形状生成智能分布的正样本点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        # 基础点：中心点
        points = [pneumo_info.center]

        # 根据形状添加特定的引导点
        if pneumo_info.shape in self.shape_strategies:
            extra_points = self.shape_strategies[pneumo_info.shape](pneumo_info)
            points.extend(extra_points[:3])  # 限制点数
        else:
            # 默认添加四个象限的点
            quarter_w, quarter_h = (x2 - x1) // 4, (y2 - y1) // 4
            points.extend(
                [
                    (center_x - quarter_w, center_y - quarter_h),
                    (center_x + quarter_w, center_y - quarter_h),
                    (center_x - quarter_w, center_y + quarter_h),
                ]
            )

        return points[:5]  # 最多5个正样本点

    def _generate_internal_negative_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """在边界框内部生成精确的负样本点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        # 在边界框的四个角落生成负样本点（避开实际气胸区域）
        margin = 10
        corner_points = [
            (x1 + margin, y1 + margin),  # 左上角
            (x2 - margin, y1 + margin),  # 右上角
            (x1 + margin, y2 - margin),  # 左下角
            (x2 - margin, y2 - margin),  # 右下角
        ]

        return corner_points

    def _generate_precise_positive_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """生成少量但精确的正样本点"""
        # 只使用中心点和一个偏移点
        center_x, center_y = pneumo_info.center
        x1, y1, x2, y2 = pneumo_info.bbox

        # 根据形状选择最佳的第二个点
        if "水平" in pneumo_info.shape:
            second_point = (center_x + (x2 - x1) // 4, center_y)
        elif "垂直" in pneumo_info.shape:
            second_point = (center_x, center_y + (y2 - y1) // 4)
        else:
            second_point = (center_x + (x2 - x1) // 6, center_y + (y2 - y1) // 6)

        return [pneumo_info.center, second_point]

    def _generate_bbox_internal_negatives(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """在边界框内部生成负样本点，避开中心区域"""
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        # 定义中心区域（气胸最可能的位置）
        center_margin_x = (x2 - x1) // 3
        center_margin_y = (y2 - y1) // 3

        negative_points = []

        # 在边界框的边缘区域生成负样本点
        edge_margin = 15

        # 上边缘
        for i in range(3):
            x = x1 + (x2 - x1) * (i + 1) // 4
            if abs(x - center_x) > center_margin_x // 2:
                negative_points.append((x, y1 + edge_margin))

        # 下边缘
        for i in range(3):
            x = x1 + (x2 - x1) * (i + 1) // 4
            if abs(x - center_x) > center_margin_x // 2:
                negative_points.append((x, y2 - edge_margin))

        # 左边缘
        for i in range(2):
            y = y1 + (y2 - y1) * (i + 1) // 3
            if abs(y - center_y) > center_margin_y // 2:
                negative_points.append((x1 + edge_margin, y))

        # 右边缘
        for i in range(2):
            y = y1 + (y2 - y1) * (i + 1) // 3
            if abs(y - center_y) > center_margin_y // 2:
                negative_points.append((x2 - edge_margin, y))

        return negative_points

    def _generate_bbox_edge_negatives(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """在边界框边缘外生成负样本点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        center_x, center_y = pneumo_info.center

        edge_distance = 25
        edge_points = []

        # 边界框外围的点
        edge_points.extend(
            [
                (center_x, y1 - edge_distance),  # 上方
                (center_x, y2 + edge_distance),  # 下方
                (x1 - edge_distance, center_y),  # 左侧
                (x2 + edge_distance, center_y),  # 右侧
            ]
        )

        # 过滤掉超出图像边界的点（这里假设图像足够大）
        valid_points = [(x, y) for x, y in edge_points if x > 0 and y > 0]

        return valid_points

    def _generate_anatomical_negatives(
        self, pneumo_info: PneumothoraxInfo, image_shape: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """基于胸部解剖结构生成负样本点"""
        h, w = image_shape
        x1, y1, x2, y2 = pneumo_info.bbox

        anatomical_points = []

        # 心脏区域（左下象限）
        heart_x, heart_y = w // 3, h * 2 // 3
        if not (x1 <= heart_x <= x2 and y1 <= heart_y <= y2):
            anatomical_points.append((heart_x, heart_y))

        # 脊柱区域（中央）
        spine_points = [(w // 2, h // 4), (w // 2, h // 2), (w // 2, h * 3 // 4)]
        for spine_x, spine_y in spine_points:
            if not (x1 <= spine_x <= x2 and y1 <= spine_y <= y2):
                anatomical_points.append((spine_x, spine_y))

        # 对侧肺部（如果气胸在左侧，则在右侧添加负样本点）
        if "左" in pneumo_info.position:
            # 在右侧添加负样本点
            anatomical_points.extend(
                [(w * 3 // 4, h // 3), (w * 3 // 4, h // 2), (w * 3 // 4, h * 2 // 3)]
            )
        elif "右" in pneumo_info.position:
            # 在左侧添加负样本点
            anatomical_points.extend(
                [(w // 4, h // 3), (w // 4, h // 2), (w // 4, h * 2 // 3)]
            )

        return anatomical_points

    def apply_bbox_constraint_to_mask(
        self, mask: np.ndarray, pneumo_info: PneumothoraxInfo
    ) -> np.ndarray:
        """应用边界框约束到分割掩码，确保分割不超出预测区域"""
        x1, y1, x2, y2 = pneumo_info.bbox

        # 创建边界框掩码
        bbox_mask = np.zeros_like(mask, dtype=np.uint8)

        # 扩展边界框以允许一定的容错
        margin = 20
        x1_expanded = max(0, x1 - margin)
        y1_expanded = max(0, y1 - margin)
        x2_expanded = min(mask.shape[1], x2 + margin)
        y2_expanded = min(mask.shape[0], y2 + margin)

        bbox_mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 1

        # 确保mask和bbox_mask都是相同的数据类型
        mask_uint8 = mask.astype(np.uint8)
        bbox_mask_uint8 = bbox_mask.astype(np.uint8)

        # 应用约束：只保留边界框内的分割结果
        constrained_mask = np.logical_and(mask_uint8 > 0, bbox_mask_uint8 > 0)

        # 转换回原始数据类型
        return constrained_mask.astype(mask.dtype)

    def _generate_horizontal_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为水平条带状气胸生成点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        y_center = (y1 + y2) // 2

        points = []
        for i in range(3):
            x = x1 + (x2 - x1) * (i + 1) // 4
            points.append((x, y_center))
        return points

    def _generate_vertical_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为垂直条带状气胸生成点"""
        x1, y1, x2, y2 = pneumo_info.bbox
        x_center = (x1 + x2) // 2

        points = []
        for i in range(3):
            y = y1 + (y2 - y1) * (i + 1) // 4
            points.append((x_center, y))
        return points

    def _generate_curved_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为半包裹型条带状气胸生成点"""
        x1, y1, x2, y2 = pneumo_info.bbox

        points = []
        # 沿着边界生成弧形点
        for i in range(4):
            t = i / 3.0
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * (0.5 + 0.3 * np.sin(t * np.pi)))
            points.append((x, y))
        return points

    def _generate_scattered_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为不规则状气胸生成点"""
        x1, y1, x2, y2 = pneumo_info.bbox

        points = []
        np.random.seed(42)  # 确保可重复性
        for _ in range(4):
            x = np.random.randint(x1 + 5, x2 - 5)
            y = np.random.randint(y1 + 5, y2 - 5)
            points.append((x, y))
        return points

    def _generate_clustered_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为团状气胸生成点"""
        cx, cy = pneumo_info.center

        points = []
        # 在中心周围生成聚集的点
        offsets = [(-10, -10), (10, -10), (-10, 10), (10, 10)]
        for dx, dy in offsets:
            points.append((cx + dx, cy + dy))
        return points

    def _generate_block_points(
        self, pneumo_info: PneumothoraxInfo
    ) -> List[Tuple[int, int]]:
        """为块状气胸生成点"""
        x1, y1, x2, y2 = pneumo_info.bbox

        points = []
        # 在区域内均匀分布点
        for i in range(2):
            for j in range(2):
                x = x1 + (x2 - x1) * (i + 1) // 3
                y = y1 + (y2 - y1) * (j + 1) // 3
                points.append((x, y))
        return points


def load_pneumothorax_data(json_path: str) -> List[Dict]:
    """加载气胸数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def demo_prompt_generation():
    """演示prompt生成过程"""
    # 初始化转换器
    converter = TextToSAM2Prompt()

    # 示例文本描述
    sample_text = """这张胸部X光片显示，肺部存在气胸。气胸的位置位于左肺。
    具体的气胸位置：气胸出现在左肺中部，呈半包裹型条带状，较大面积，
    气胸区域边界框坐标为：左上角(602, 147)，右下角(936, 799)，尺寸334×652像素。"""

    # 解析文本
    pneumo_info = converter.parse_text_description(sample_text)
    if pneumo_info:
        print("=== 解析结果 ===")
        print(f"坐标: {pneumo_info.bbox}")
        print(f"位置: {pneumo_info.position}")
        print(f"形状: {pneumo_info.shape}")
        print(f"大小: {pneumo_info.size}")
        print(f"面积: {pneumo_info.area} 像素")
        print(f"中心: {pneumo_info.center}")

        # 生成不同类型的提示
        image_shape = (1024, 1024)  # 假设图像尺寸

        print("\n=== SAM2 Prompt生成 ===")

        # 1. 边界框提示
        bbox_prompt = converter.generate_bbox_prompt(pneumo_info)
        print(f"边界框提示: {bbox_prompt}")

        # 2. 点提示
        point_prompt = converter.generate_point_prompts(pneumo_info)
        print(f"点提示: {point_prompt}")

        # 3. 混合提示（推荐）
        hybrid_prompt = converter.generate_hybrid_prompt(pneumo_info, image_shape)
        print(f"混合提示: {hybrid_prompt}")


if __name__ == "__main__":
    demo_prompt_generation()
