#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解剖学约束模块
用于对气胸分割结果应用解剖学约束，提高分割的医学合理性
支持利用结构化位置信息进行精确约束
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class AnatomicalRegion:
    """解剖学区域定义"""

    name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: np.ndarray
    confidence: float


class AnatomicalConstraints:
    """解剖学约束处理类"""

    def __init__(self, image_shape: Tuple[int, int]):
        """
        初始化解剖学约束

        Args:
            image_shape: 图像形状 (height, width)
        """
        self.image_shape = image_shape
        self.height, self.width = image_shape

        # 定义肺部区域的position_code映射
        self.position_code_regions = {
            "A": {"name": "左上肺", "x_range": (0.1, 0.45), "y_range": (0.1, 0.45)},
            "B": {"name": "左中肺", "x_range": (0.1, 0.45), "y_range": (0.35, 0.65)},
            "C": {"name": "左下肺", "x_range": (0.1, 0.45), "y_range": (0.55, 0.9)},
            "D": {"name": "右上肺", "x_range": (0.55, 0.9), "y_range": (0.1, 0.45)},
            "E": {"name": "右中肺", "x_range": (0.55, 0.9), "y_range": (0.35, 0.65)},
            "F": {"name": "右下肺", "x_range": (0.55, 0.9), "y_range": (0.55, 0.9)},
            "LEFT": {"name": "左肺", "x_range": (0.1, 0.45), "y_range": (0.1, 0.9)},
            "RIGHT": {"name": "右肺", "x_range": (0.55, 0.9), "y_range": (0.1, 0.9)},
        }

    def detect_lung_boundaries(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        检测肺部边界

        Args:
            image: 输入的X光图像

        Returns:
            包含左肺和右肺掩码的字典
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # 应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 使用Otsu阈值分割
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

            # 寻找轮廓
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 创建肺部掩码（简化版本）
            lung_mask = np.zeros_like(gray, dtype=np.uint8)

            # 选择最大的几个轮廓作为肺部区域
            if contours:
                # 按面积排序
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # 绘制最大的轮廓
                for i, contour in enumerate(contours[:2]):  # 最多取两个最大轮廓
                    if cv2.contourArea(contour) > 1000:  # 过滤小轮廓
                        cv2.fillPoly(lung_mask, [contour], 255)

            # 分离左右肺（简化版本：按中线分割）
            mid_x = self.width // 2

            left_lung = lung_mask.copy()
            left_lung[:, mid_x:] = 0

            right_lung = lung_mask.copy()
            right_lung[:, :mid_x] = 0

            return {
                "left_lung": left_lung,
                "right_lung": right_lung,
                "combined": lung_mask,
            }

        except Exception as e:
            print(f"肺部边界检测失败: {e}")
            # 返回默认的肺部区域（基于经验的近似区域）
            return self._create_default_lung_masks()

    def _create_default_lung_masks(self) -> Dict[str, np.ndarray]:
        """创建默认的肺部掩码（基于经验）"""
        lung_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 定义肺部的大致区域（经验值）
        # 肺部通常占据胸部X光片的中央区域
        margin_x = int(self.width * 0.1)
        margin_y_top = int(self.height * 0.15)
        margin_y_bottom = int(self.height * 0.2)

        # 创建椭圆形的肺部区域
        center_x = self.width // 2
        center_y = self.height // 2

        # 左肺
        left_center = (center_x - self.width // 4, center_y)
        left_axes = (self.width // 4 - margin_x, self.height // 2 - margin_y_top)

        # 右肺
        right_center = (center_x + self.width // 4, center_y)
        right_axes = (self.width // 4 - margin_x, self.height // 2 - margin_y_top)

        left_lung = np.zeros_like(lung_mask)
        right_lung = np.zeros_like(lung_mask)

        cv2.ellipse(left_lung, left_center, left_axes, 0, 0, 360, 255, -1)
        cv2.ellipse(right_lung, right_center, right_axes, 0, 0, 360, 255, -1)

        combined = cv2.bitwise_or(left_lung, right_lung)

        return {"left_lung": left_lung, "right_lung": right_lung, "combined": combined}

    def apply_area_constraint(
        self, mask: np.ndarray, max_area_ratio: float = 0.3
    ) -> np.ndarray:
        """
        应用面积约束

        Args:
            mask: 输入掩码
            max_area_ratio: 最大面积比例（相对于图像总面积）

        Returns:
            约束后的掩码
        """
        total_pixels = self.height * self.width
        max_area = int(total_pixels * max_area_ratio)

        mask_area = np.sum(mask > 0)

        if mask_area > max_area:
            # 如果面积过大，进行形态学腐蚀
            kernel_size = max(3, int(np.sqrt(mask_area / max_area)) * 2 + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def apply_morphological_constraint(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学约束

        Args:
            mask: 输入掩码

        Returns:
            约束后的掩码
        """
        try:
            # 确保mask是uint8类型
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # 去除小的连通区域
            mask = self._remove_small_components(mask, min_area=100)

            # 填充小的空洞
            mask = self._fill_small_holes(mask, max_hole_area=500)

            # 平滑边界
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            return mask

        except Exception as e:
            print(f"形态学约束应用失败: {e}")
            # 如果失败，返回原始mask（确保是uint8类型）
            return mask.astype(np.uint8) if mask.dtype != np.uint8 else mask

    def apply_connectivity_constraint(
        self, mask: np.ndarray, max_components: int = 2
    ) -> np.ndarray:
        """
        应用连通性约束

        Args:
            mask: 输入掩码
            max_components: 最大连通组件数量

        Returns:
            约束后的掩码
        """
        # 寻找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8
        )

        if num_labels <= max_components + 1:  # +1 for background
            return mask

        # 保留最大的几个组件
        component_areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
        largest_indices = np.argsort(component_areas)[-max_components:]

        # 创建新掩码
        new_mask = np.zeros_like(mask)
        for idx in largest_indices:
            component_label = idx + 1  # +1 because we excluded background
            new_mask[labels == component_label] = mask[labels == component_label]

        return new_mask

    def apply_lung_boundary_constraint(
        self, mask: np.ndarray, image: np.ndarray
    ) -> np.ndarray:
        """
        应用肺部边界约束

        Args:
            mask: 输入掩码
            image: 原始图像

        Returns:
            约束后的掩码
        """
        # 检测肺部边界
        lung_masks = self.detect_lung_boundaries(image)
        lung_boundary = lung_masks["combined"]

        # 确保两个掩码都是uint8类型
        mask_uint8 = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
        lung_boundary_uint8 = (
            lung_boundary.astype(np.uint8)
            if lung_boundary.dtype != np.uint8
            else lung_boundary
        )

        # 将分割结果限制在肺部区域内
        constrained_mask = cv2.bitwise_and(mask_uint8, lung_boundary_uint8)

        return constrained_mask

    def apply_position_code_constraint(
        self,
        mask: np.ndarray,
        position_code: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> np.ndarray:
        """
        基于position_code应用位置约束

        Args:
            mask: 输入掩码
            position_code: 位置编码 (A-F, LEFT, RIGHT)
            bbox: 可选的边界框约束

        Returns:
            约束后的掩码
        """
        if position_code not in self.position_code_regions:
            print(f"未知的position_code: {position_code}")
            return mask

        # 获取对应区域的范围
        region_info = self.position_code_regions[position_code]
        x_range = region_info["x_range"]
        y_range = region_info["y_range"]

        # 转换为像素坐标
        x1 = int(x_range[0] * self.width)
        x2 = int(x_range[1] * self.width)
        y1 = int(y_range[0] * self.height)
        y2 = int(y_range[1] * self.height)

        # 如果提供了边界框，进一步约束区域
        if bbox is not None:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            # 取交集
            x1 = max(x1, bbox_x1 - 20)  # 允许一定的容差
            y1 = max(y1, bbox_y1 - 20)
            x2 = min(x2, bbox_x2 + 20)
            y2 = min(y2, bbox_y2 + 20)

        # 创建区域掩码
        region_mask = np.zeros_like(mask, dtype=np.uint8)
        region_mask[y1:y2, x1:x2] = 255

        # 确保掩码是正确的数据类型
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if region_mask.dtype == bool:
            region_mask = region_mask.astype(np.uint8) * 255
        elif region_mask.dtype != np.uint8:
            region_mask = region_mask.astype(np.uint8)

        # 应用位置约束
        constrained_mask = cv2.bitwise_and(mask, region_mask)

        print(f"应用{region_info['name']}位置约束: ({x1},{y1}) -> ({x2},{y2})")
        return constrained_mask

    def apply_bbox_constraint(
        self,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        expansion_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        应用边界框约束

        Args:
            mask: 输入掩码
            bbox: 边界框 (x1, y1, x2, y2)
            expansion_ratio: 边界框扩展比例

        Returns:
            约束后的掩码
        """
        x1, y1, x2, y2 = bbox

        # 计算扩展量
        width = x2 - x1
        height = y2 - y1
        expand_x = int(width * expansion_ratio)
        expand_y = int(height * expansion_ratio)

        # 扩展边界框
        expanded_x1 = max(0, x1 - expand_x)
        expanded_y1 = max(0, y1 - expand_y)
        expanded_x2 = min(self.width, x2 + expand_x)
        expanded_y2 = min(self.height, y2 + expand_y)

        # 创建边界框掩码
        bbox_mask = np.zeros_like(mask, dtype=np.uint8)
        bbox_mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = 255

        # 确保掩码是正确的数据类型
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        if bbox_mask.dtype == bool:
            bbox_mask = bbox_mask.astype(np.uint8) * 255
        elif bbox_mask.dtype != np.uint8:
            bbox_mask = bbox_mask.astype(np.uint8)

        # 应用边界框约束
        constrained_mask = cv2.bitwise_and(mask, bbox_mask)

        print(
            f"应用边界框约束: ({expanded_x1},{expanded_y1}) -> ({expanded_x2},{expanded_y2})"
        )
        return constrained_mask

    def apply_structured_constraints(
        self, mask: np.ndarray, pneumo_info, image: np.ndarray = None
    ) -> np.ndarray:
        """
        应用基于结构化数据的约束

        Args:
            mask: 输入掩码
            pneumo_info: 气胸信息对象（包含结构化数据）
            image: 原始图像（可选）

        Returns:
            约束后的掩码
        """
        constrained_mask = mask.copy()

        # 检查是否有结构化数据
        if (
            hasattr(pneumo_info, "has_structured_data")
            and pneumo_info.has_structured_data
        ):
            print("使用结构化数据进行约束...")

            # 1. 应用position_code约束
            if hasattr(pneumo_info, "position_code") and pneumo_info.position_code:
                constrained_mask = self.apply_position_code_constraint(
                    constrained_mask, pneumo_info.position_code, pneumo_info.bbox
                )

            # 2. 应用精确的边界框约束
            if pneumo_info.bbox:
                constrained_mask = self.apply_bbox_constraint(
                    constrained_mask, pneumo_info.bbox, expansion_ratio=0.15
                )
        else:
            print("使用传统约束方法...")
            # 使用传统的约束方法
            constrained_mask = self.apply_all_constraints(constrained_mask, image)
            return constrained_mask

        # 3. 应用基础的形态学和连通性约束
        constrained_mask = self.apply_morphological_constraint(constrained_mask)
        constrained_mask = self.apply_connectivity_constraint(constrained_mask)

        # 4. 应用面积约束（对结构化数据使用更宽松的限制）
        constrained_mask = self.apply_area_constraint(
            constrained_mask, max_area_ratio=0.4
        )

        # 5. 如果提供了图像，应用肺部边界约束
        if image is not None:
            try:
                constrained_mask = self.apply_lung_boundary_constraint(
                    constrained_mask, image
                )
            except Exception as e:
                print(f"肺部边界约束失败，跳过: {e}")

        return constrained_mask

    def _remove_small_components(
        self, mask: np.ndarray, min_area: int = 100
    ) -> np.ndarray:
        """移除小的连通组件"""
        # 确保mask是uint8类型
        binary_mask = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        new_mask = np.zeros_like(mask, dtype=np.uint8)

        for i in range(1, num_labels):  # 跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255  # 使用255作为前景值

        return new_mask

    def _fill_small_holes(
        self, mask: np.ndarray, max_hole_area: int = 500
    ) -> np.ndarray:
        """填充小的空洞"""
        # 确保mask是uint8类型
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # 反转掩码以找到空洞
        inverted = cv2.bitwise_not(binary_mask)

        # 寻找空洞的连通组件
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )

        filled_mask = binary_mask.copy()

        for i in range(1, num_labels):  # 跳过背景
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
                # 填充小空洞
                hole_region = labels == i
                filled_mask[hole_region] = 255  # 使用255填充

        return filled_mask

    def apply_all_constraints(
        self, mask: np.ndarray, image: np.ndarray = None
    ) -> np.ndarray:
        """
        应用所有解剖学约束

        Args:
            mask: 输入掩码
            image: 原始图像（可选）

        Returns:
            约束后的掩码
        """
        constrained_mask = mask.copy()

        # 1. 应用形态学约束
        constrained_mask = self.apply_morphological_constraint(constrained_mask)

        # 2. 应用连通性约束
        constrained_mask = self.apply_connectivity_constraint(constrained_mask)

        # 3. 应用面积约束
        constrained_mask = self.apply_area_constraint(constrained_mask)

        # 4. 如果提供了图像，应用肺部边界约束
        if image is not None:
            constrained_mask = self.apply_lung_boundary_constraint(
                constrained_mask, image
            )

        return constrained_mask
