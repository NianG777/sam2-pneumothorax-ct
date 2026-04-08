import cv2
import numpy as np
from pycocotools import mask as coco_mask

class DisplayUtils:
    def __init__(self):
        self.transparency = 0.3
        self.box_width = 2
        self.text_size = 1.5  # 默认文字大小

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)
    
    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def increase_text_size(self):
        self.text_size = min(5.0, self.text_size + 0.1)  # 最大文字大小为 5.0

    def decrease_text_size(self):
        self.text_size = max(0.5, self.text_size - 0.1)  # 最小文字大小为 0.5

    def overlay_mask_on_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color=(0, 0, 255),
        alpha: float = None,
        outline: bool = True,
        outline_color=(255, 255, 0),
        outline_thickness: int = 1,
    ) -> np.ndarray:

        
        if alpha is None:
            alpha = float(self.transparency)

        # 准备二值掩码（0/1）
        mask_bin = (mask > 0).astype(np.uint8)
        if image.ndim != 3:
            raise ValueError("overlay_mask_on_image 期望三通道图像")

        h, w = mask_bin.shape
        # 构造彩色层，仅在掩码区域着色
        colored = np.zeros_like(image, dtype=np.uint8)
        colored[mask_bin.astype(bool)] = np.array(color, dtype=np.uint8)

        # 混合：仅在掩码像素处进行加权
        blended_full = cv2.addWeighted(image, 1.0, colored, alpha, 0)
        result = image.copy()
        result[mask_bin.astype(bool)] = blended_full[mask_bin.astype(bool)]

        # 可选绘制边界
        if outline:
            # 寻找轮廓并描边
            mask_u8 = (mask_bin * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cv2.drawContours(result, contours, -1, outline_color, outline_thickness)

        return result

    def refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 3,
        remove_small: int = 0,
        smooth_outline: bool = False,
    ) -> np.ndarray:

        if mask.ndim != 2:
            # 压缩到 2D
            mask = np.squeeze(mask)
        mask_bin = (mask > 0).astype(np.uint8)

        # 形态学开闭运算，去毛刺与填小孔
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        # 移除小连通域
        if remove_small and remove_small > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
            refined = np.zeros_like(mask_bin)
            for i in range(1, num_labels):  # 跳过背景 0
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= remove_small:
                    refined[labels == i] = 1
            mask_bin = refined

        # 轮廓平滑（可选）
        if smooth_outline:
            blurred = cv2.GaussianBlur((mask_bin * 255).astype(np.uint8), (3, 3), 0)
            _, mask_bin = cv2.threshold(blurred, 127, 1, cv2.THRESH_BINARY)

        return mask_bin.astype(np.uint8)

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = ann["segmentation"]
        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, self.text_size, 1)[0]
        thickness = max(1, int(self.text_size * 3.33))

        cv2.rectangle(image, (x, y + 1), (x + txt_size[0] + 1, y + int(self.text_size*txt_size[1])), color, -1)
        cv2.putText(image, text, (x, y + txt_size[1]), font, self.text_size, txt_color, thickness=thickness)
        return image

    def draw_annotations(self, image, categories, annotations, colors):
        for ann, color in zip(annotations, colors):
            image = self.draw_box_on_image(image, categories, ann, color)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
