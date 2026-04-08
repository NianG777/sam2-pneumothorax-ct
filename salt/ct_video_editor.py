"""
CT序列视频编辑器
基于SAM2的视频分割功能，处理CT切片序列
"""

import os
import atexit
import json
import numpy as np
import cv2
import torch
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# SAM2相关导入
from sam2.build_sam import build_sam2_video_predictor
from .mask_debug_utils import save_mask_with_validation, analyze_mask_data
from sam2.utils.misc import load_video_frames

from salt.display_utils import DisplayUtils


# 兼容 Windows 中文路径的图像读写工具
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """使用 np.fromfile + cv2.imdecode 读取图像，兼容中文/非ASCII路径"""
    try:
        # Path/str 都可，统一转为字符串
        p = str(path)
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def natural_sort_key(path):
    """
    自然排序键函数，用于正确排序包含数字的文件名
    例如：IM0, IM1, IM2, ..., IM10, IM11 而不是 IM0, IM1, IM10, IM11, IM2
    """
    filename = path.name
    # 将文件名中的数字部分转换为整数进行排序
    parts = re.split(r'(\d+)', filename)
    for i in range(len(parts)):
        if parts[i].isdigit():
            parts[i] = int(parts[i])
    return parts

def imwrite_unicode(path, image) -> bool:
    """使用 cv2.imencode + np.tofile 写图像，兼容中文/非ASCII路径"""
    try:
        ext = Path(path).suffix
        if ext == "":
            # 默认保存为PNG
            ext = ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False

def to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """将任意类型的掩码转换为 0/255 的 uint8 单通道图像。
    - 支持 bool、float、int 类型
    - 自动处理 (H,W,1) 或 (1,H,W) 取第一通道
    """
    if mask is None:
        return None
    m = np.array(mask)
    # 压缩到 2D
    if m.ndim == 3:
        m = m.squeeze()
        if m.ndim != 2:
            # 如仍非2D，尝试取第一个通道
            m = m[..., 0]
    # 转换类型
    if m.dtype == np.bool_ or m.dtype == bool:
        m = (m.astype(np.uint8)) * 255
    elif np.issubdtype(m.dtype, np.floating):
        # 归一化到 [0,1] 再放大到 [0,255]
        m = np.clip(m, 0.0, 1.0)
        m = (m * 255.0).astype(np.uint8)
    elif np.issubdtype(m.dtype, np.integer):
        # 若为 {0,1}，放大；否则裁剪到 [0,255]
        unique_vals = None
        try:
            unique_vals = np.unique(m)
        except Exception:
            pass
        if unique_vals is not None and unique_vals.size <= 3 and set(unique_vals.tolist()).issubset({0, 1, 255}):
            # 常见二值情况
            if 255 in unique_vals:
                m = np.clip(m, 0, 255).astype(np.uint8)
            else:
                m = (m.astype(np.uint8)) * 255
        else:
            m = np.clip(m, 0, 255).astype(np.uint8)
    else:
        # 其它类型，尝试强制转为 uint8
        try:
            m = m.astype(np.uint8)
        except Exception:
            m = np.zeros_like(m, dtype=np.uint8)
    return m


class CTSequenceLoader:
    """CT序列加载器"""
    
    def __init__(self, dataset_root: str, sequence_name: Optional[str] = None, images_path: Optional[str] = None):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.sequence_name = sequence_name
        
        # 支持直接传入图像目录，或使用传统 dataset_root/images/sequence_name 结构
        if images_path:
            self.images_path = Path(images_path)
        else:
            if self.dataset_root is None or self.sequence_name is None:
                raise ValueError("CTSequenceLoader 需要提供 images_path 或 dataset_root+sequence_name")
            self.images_path = self.dataset_root / "images" / sequence_name
        
        # 尝试定位嵌入目录（可选）
        if self.dataset_root and self.sequence_name:
            maybe_embeddings = self.dataset_root / "embeddings" / self.sequence_name
            self.embeddings_path = maybe_embeddings if maybe_embeddings.exists() else None
        else:
            self.embeddings_path = None
        
        # 检查路径是否存在
        if not self.images_path.exists():
            raise FileNotFoundError(f"图像路径不存在: {self.images_path}")
        # 嵌入路径可以不存在，SAM2会在运行时计算所需特征
        # 如果存在则用于兼容旧流程
        
        # 获取所有图像文件（支持png/jpg/jpeg，大小写不敏感）
        allowed_exts = {'.png', '.jpg', '.jpeg'}
        image_list = []
        for f in self.images_path.iterdir():
            if f.is_file() and f.suffix.lower() in allowed_exts:
                image_list.append(f)
        # 使用自然排序，正确处理IM0, IM1, IM10等文件名
        self.image_files = sorted(image_list, key=natural_sort_key)

        if len(self.image_files) == 0:
            raise ValueError(f"在序列目录中未找到PNG/JPG图像文件: {self.images_path}")

        # 获取embedding文件（如存在）
        if self.embeddings_path and self.embeddings_path.exists():
            embedding_list = [f for f in self.embeddings_path.glob("*.npy")]
            self.embedding_files = sorted(embedding_list, key=natural_sort_key)
        else:
            self.embedding_files = []
        
        self.num_frames = len(self.image_files)
        print(f"加载CT序列 {sequence_name}: {self.num_frames} 帧")
    
    def load_frame(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """加载指定帧的图像和嵌入"""
        if frame_idx >= self.num_frames:
            raise IndexError(f"帧索引超出范围: {frame_idx} >= {self.num_frames}")
        
        # 加载图像（强制以3通道读取，忽略PNG透明通道）
        image_bgr = imread_unicode(self.image_files[frame_idx], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"读取图像失败或格式不支持: {self.image_files[frame_idx]}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 加载嵌入（如果存在），否则返回None以兼容SAM2在线计算
        embedding = None
        if self.embedding_files and frame_idx < len(self.embedding_files):
            try:
                embedding = np.load(str(self.embedding_files[frame_idx]))
            except Exception as e:
                print(f"Warning: 加载嵌入失败: {self.embedding_files[frame_idx]} -> {e}")
                embedding = None
        
        return image, embedding
    
    def load_all_frames(self) -> List[np.ndarray]:
        """加载所有帧的图像"""
        frames = []
        for i in range(self.num_frames):
            image, _ = self.load_frame(i)
            frames.append(image)
        return frames


class CTVideoEditor:
    """CT序列视频编辑器主类"""
    
    def __init__(self, 
                 sam2_checkpoint: str,
                 sam2_config: str,
                 dataset_root: Optional[str],
                 sequence_name: Optional[str],
                 output_root: str,
                 categories: List[str],
                 device: str = "cuda",
                 images_path: Optional[str] = None,
                 base_data_root: Optional[str] = None):
        
        self.device = device
        self.categories = categories
        # 若未显式提供sequence_name，则从图像目录名推断
        self.sequence_name = sequence_name or (Path(images_path).name if images_path else "sequence")
        # 基础数据根路径（用于在输出中保留相对目录结构，例如包含“人名”的上级目录）
        self.base_data_root = Path(base_data_root) if base_data_root else None
        
        # 设置输出路径
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化CT序列加载器（支持直接传入图像目录）
        self.ct_loader = CTSequenceLoader(dataset_root, sequence_name, images_path=images_path)

        # 记录所选图像目录
        self.images_path = self.ct_loader.images_path
        # 计算在输出中的相对目录结构：
        # 如果提供了 base_data_root 且 images_path 位于其之下，则使用相对路径；否则退化为 sequence_name
        try:
            if self.base_data_root is not None:
                rel = self.images_path.relative_to(self.base_data_root)
                self.sequence_rel_path = rel
            else:
                self.sequence_rel_path = Path(self.sequence_name)
        except Exception:
            # 不在 base_data_root 下或计算失败时，退化为单层目录
            self.sequence_rel_path = Path(self.sequence_name)
        
        # 初始化SAM2视频预测器
        print("初始化SAM2视频预测器...")
        self.predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)
        # 记录SAM2配置以便界面切换序列时复用
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # 初始化显示工具
        self.display_utils = DisplayUtils()
        
        # 当前状态
        self.current_frame_idx = 0
        self.current_category_id = 0
        self.inference_state = None
        self.current_object_id = 1
        
        # 标注数据存储
        self.annotations = {}  # frame_idx -> list of annotations
        self.keyframes = set()  # 关键帧集合
        
        # 加载CT序列并初始化视频状态
        self._initialize_video_state()
        
        # 当前显示的图像
        self.current_image = None
        self.display_image = None
        self._load_current_frame()

        # 注册程序退出清理：删除临时缓存目录
        atexit.register(self.cleanup_temp_cache)
    def reload_sequence(self, images_path: str, base_data_root: Optional[str] = None, sequence_name: Optional[str] = None) -> bool:
        try:
            # 更新基础数据根路径（用于输出相对层级）
            if base_data_root is not None:
                self.base_data_root = Path(base_data_root)
            
            # 更新序列名（默认用目录名）
            self.sequence_name = sequence_name or Path(images_path).name
            
            # 重新构建加载器
            self.ct_loader = CTSequenceLoader(
                dataset_root=None,
                sequence_name=self.sequence_name,
                images_path=images_path
            )
            self.images_path = self.ct_loader.images_path
            
            # 重新计算在输出中的相对路径（包含人名等上级目录）
            try:
                if self.base_data_root is not None:
                    rel = self.images_path.relative_to(self.base_data_root)
                    self.sequence_rel_path = rel
                else:
                    self.sequence_rel_path = Path(self.sequence_name)
            except Exception:
                self.sequence_rel_path = Path(self.sequence_name)
            
            # 重置状态（不重建 predictor）
            self.current_frame_idx = 0
            self.current_object_id = 1
            self.inference_state = None
            self.annotations.clear()
            self.keyframes.clear()
            self.prompts_cache.clear()
            self.click_history = []
            
            # 初始化视频状态（会调用 predictor.init_state，但不重建模型）
            self._initialize_video_state()
            
            # 刷新当前帧显示
            self._load_current_frame()
            return True
        except Exception as e:
            print(f"重新加载序列失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _initialize_video_state(self):
        """初始化视频状态"""
        print("初始化视频状态...")
        
        # 直接将源图像序列编码为临时MP4视频，避免生成多余的临时帧文件
        temp_video_file_dir = self.output_root / "temp_video"
        temp_video_file_dir.mkdir(parents=True, exist_ok=True)
        temp_video_file = temp_video_file_dir / f"{self.sequence_name}.mp4"

        # 读取第一张源图像，确定视频尺寸
        if len(self.ct_loader.image_files) == 0:
            raise ValueError(f"no images found in {self.ct_loader.images_path}")
        first_src_path = self.ct_loader.image_files[0]
        first_img = imread_unicode(first_src_path, cv2.IMREAD_COLOR)
        if first_img is None:
            raise ValueError(f"无法读取第一张图像: {first_src_path}")
        height, width = first_img.shape[:2]
        fps = 25  # 默认帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(temp_video_file), fourcc, fps, (width, height))

        written_count = 0
        first_few = []
        for idx, img_path in enumerate(self.ct_loader.image_files):
            img = imread_unicode(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: 写视频时读取源图像失败，跳过: {img_path}")
                continue
            if img.shape[0] != height or img.shape[1] != width:
                # 尺寸不一致时进行缩放以保证视频编码成功
                img = cv2.resize(img, (width, height))
            writer.write(img)
            written_count += 1
            if len(first_few) < 3:
                first_few.append(Path(img_path).name)
        writer.release()

        if not temp_video_file.exists() or temp_video_file.stat().st_size == 0:
            raise ValueError(f"临时视频文件生成失败: {temp_video_file}")

        print(f"临时视频生成完成: {temp_video_file.resolve()} (帧数: {written_count}, 尺寸: {width}x{height}, FPS: {fps})")
        # if first_few:
        #     print(f"首3个源帧示例: {first_few}")

        video_path = str(temp_video_file.resolve())
        
        # 添加点击历史记录
        self.click_history = []  # 存储所有点击的历史记录
        
        # key: (frame_idx, obj_id) -> {'points': List[List[int]], 'labels': List[int]}
        self.prompts_cache: Dict[Tuple[int, int], Dict[str, List]] = {}
        
        # 初始化推理状态
        with torch.inference_mode():
            self.inference_state = self.predictor.init_state(
                video_path=video_path,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False
            )
    
    def _load_current_frame(self):
        """加载当前帧"""
        self.current_image, _ = self.ct_loader.load_frame(self.current_frame_idx)
        self.display_image = self.current_image.copy()
        self._update_display()
    
    def _update_display(self):
        """更新显示图像"""
        self.display_image = self.current_image.copy()
        
        # 绘制已有标注
        if self.current_frame_idx in self.annotations:
            for ann in self.annotations[self.current_frame_idx]:
                if 'mask' in ann:
                    self.display_image = self.display_utils.overlay_mask_on_image(
                        self.display_image, ann['mask']
                    )
    
    def add_point(self, point: List[int], label: int) -> bool:
        """添加点击标注"""
        try:
            with torch.inference_mode():
                # 记录点击历史
                click_info = {
                    'frame_idx': self.current_frame_idx,
                    'point': point,
                    'label': label,
                    'type': '正样本' if label == 1 else '负样本'
                }
                self.click_history.append(click_info)
                
                # 调试输出
                print(f"添加点击: 坐标={point}, 标签={label} ({'正样本' if label == 1 else '负样本'})")
                print(f"点击历史总数: {len(self.click_history)}")
                
                # 累积当前帧/对象的全部提示点（包括之前的正负样本）
                cache_key = (self.current_frame_idx, self.current_object_id)
                if cache_key not in self.prompts_cache:
                    self.prompts_cache[cache_key] = {"points": [], "labels": []}
                self.prompts_cache[cache_key]["points"].append(point)
                self.prompts_cache[cache_key]["labels"].append(label)

                all_points = np.array(self.prompts_cache[cache_key]["points"], dtype=np.int32)
                all_labels = np.array(self.prompts_cache[cache_key]["labels"], dtype=np.int32)

                # 清理该帧在 SAM2 状态中的旧提示，再一次性提交所有提示点，避免右键清空正样本的现象
                self.predictor.clear_all_prompts_in_frame(
                    self.inference_state,
                    self.current_frame_idx,
                    self.current_object_id,
                )

                frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=self.current_object_id,
                    points=all_points,
                    labels=all_labels,
                )
                
                # 调试输出
                print(f"SAM2返回: frame_idx={frame_idx}, object_ids={object_ids}, masks数量={len(masks)}")
                
                # 更新显示
                if len(masks) > 0:
                    mask = masks[0].cpu().numpy()
                    
                    # 调试输出mask信息
                    print(f"原始mask形状: {mask.shape}, 数据类型: {mask.dtype}")
                    print(f"Mask值范围: min={mask.min():.3f}, max={mask.max():.3f}, 非零像素数: {np.count_nonzero(mask)}")
                    
                    # 修复mask维度问题：确保mask是2D数组
                    if mask.ndim == 3:
                        # 如果是3D mask (1, H, W)，取第一个通道
                        mask = mask[0]
                    elif mask.ndim > 3:
                        # 如果维度更高，压缩到2D
                        mask = mask.squeeze()
                    
                    # 确保mask是2D的
                    if mask.ndim != 2:
                        print(f"Warning: 无法将mask转换为2D，当前维度: {mask.shape}")
                        return False
                    
                    print(f"处理后mask形状: {mask.shape}, 非零像素数: {np.count_nonzero(mask)}")
                    
                    # 重新加载原始图像，避免mask累积 - 这是关键修复！
                    # SAM2返回的mask已经是考虑了所有正负样本点后的最终结果
                    # 我们需要从原始图像开始重新叠加，而不是在已有显示上累积
                    original_image = self.current_image.copy()
                    
                    
                    mask = self.display_utils.refine_mask(mask, kernel_size=3, remove_small=50, smooth_outline=False)

                    # 叠加最终mask到原始图像上
                    self.display_image = self.display_utils.overlay_mask_on_image(
                        original_image, mask, outline=True
                    )
                    
                    # 在显示图像上绘制点击历史
                    self._draw_click_history()
                    
                    # 标记为关键帧
                    self.keyframes.add(self.current_frame_idx)
                    
                    print(f"点击处理完成，标记为关键帧: {self.current_frame_idx}")
                    
                return True
                
        except Exception as e:
            print(f"添加点标注失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def propagate_in_video(self) -> bool:
        """在视频中传播标注"""
        try:
            print("开始时序传播...")
            
            with torch.inference_mode():
                # 执行视频传播
                for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                    self.inference_state
                ):
                    # 保存传播结果
                    if len(masks) > 0:
                        mask = masks[0].cpu().numpy()
                        
                        # 修复mask维度问题：确保mask是2D数组
                        if mask.ndim == 3:
                            # 如果是3D mask (1, H, W)，取第一个通道
                            mask = mask[0]
                        elif mask.ndim > 3:
                            # 如果维度更高，压缩到2D
                            mask = mask.squeeze()
                        
                        # 确保mask是2D的
                        if mask.ndim != 2:
                            print(f"Warning: 无法将mask转换为2D，当前维度: {mask.shape}")
                            continue
                        
                        # 掩码细化
                        mask = self.display_utils.refine_mask(mask, kernel_size=3, remove_small=50, smooth_outline=False)

                        # 保存标注
                        if frame_idx not in self.annotations:
                            self.annotations[frame_idx] = []
                        
                        annotation = {
                            'category_id': self.current_category_id,
                            'mask': mask,
                            'object_id': self.current_object_id,
                            'is_keyframe': frame_idx in self.keyframes
                        }
                        
                        self.annotations[frame_idx].append(annotation)
                
                print(f"传播完成，共处理 {len(self.annotations)} 帧")
                self._update_display()
                return True
                
        except Exception as e:
            print(f"视频传播失败: {e}")
            return False
    
    def next_frame(self):
        """下一帧"""
        if self.current_frame_idx < self.ct_loader.num_frames - 1:
            self.current_frame_idx += 1
            self._load_current_frame()
    
    def prev_frame(self):
        """上一帧"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self._load_current_frame()
    
    def set_category(self, category_id: int):
        """设置当前标注类别"""
        if 0 <= category_id < len(self.categories):
            self.current_category_id = category_id
    
    def reset_current_frame(self):
        """重置当前帧的标注"""
        if self.current_frame_idx in self.annotations:
            del self.annotations[self.current_frame_idx]
        
        # 从关键帧集合中移除
        self.keyframes.discard(self.current_frame_idx)
        
        # 清除SAM2状态中的当前帧标注
        try:
            with torch.inference_mode():
                self.predictor.clear_all_prompts_in_frame(
                    self.inference_state, 
                    self.current_frame_idx, 
                    self.current_object_id
                )
        except Exception as e:
            print(f"清除帧标注失败: {e}")
       
        cache_key = (self.current_frame_idx, self.current_object_id)
        if cache_key in self.prompts_cache:
            del self.prompts_cache[cache_key]
        self.click_history = [c for c in self.click_history if c['frame_idx'] != self.current_frame_idx]
        
        self._load_current_frame()
    
    def save_annotations(self) -> bool:
        """保存标注结果"""
        try:
            # 创建序列特定的输出目录（带相对层级，以包含人名等上级目录）
            sequence_output_dir = self.output_root / self.sequence_rel_path
            sequence_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存为JSON格式
            json_path = self.output_root / "annotations.json"
            
            # 读取现有的annotations.json（如果存在）
            existing_data = {}
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = {}
            
            # 转换标注数据为可序列化格式，并保证从第一帧开始输出掩码文件
            sequence_annotations = {}
            for frame_idx in range(self.ct_loader.num_frames):
                sequence_annotations[str(frame_idx)] = []
                anns = self.annotations.get(frame_idx, [])

                # 使用原图扩展名保持与原图一致的掩码文件格式
                src_path = self.ct_loader.image_files[frame_idx]
                src_ext = src_path.suffix  # 包含点，如 .png 或 .jpg
                frame_filename = src_path.stem + src_ext
                mask_path = sequence_output_dir / frame_filename

                if len(anns) == 0:
                    # 无任何掩码信息，保存与原图尺寸一致的纯黑掩码
                    src_img = imread_unicode(src_path, cv2.IMREAD_COLOR)
                    if src_img is None:
                        print(f"Warning: 无法读取源图像以生成纯黑掩码: {src_path}")
                        black = np.zeros((256, 256), dtype=np.uint8)
                    else:
                        h, w = src_img.shape[:2]
                        black = np.zeros((h, w), dtype=np.uint8)
                    ok = imwrite_unicode(mask_path, black)
                    if ok:
                        print(f"✓ 帧 {frame_idx} 无掩码，已保存纯黑图: {mask_path}")
                    else:
                        print(f"✗ 保存纯黑掩码失败: {mask_path}")
                else:
                    # 有掩码时按原流程保存（注意：多个对象会多次覆盖同一路径，保持现有逻辑）
                    for ann in anns:
                        mask_data = ann.get('mask')
                        mask_is_empty = True
                        if mask_data is not None:
                            try:
                                mask_np = np.array(mask_data)
                                mask_is_empty = (mask_np.ndim < 1) or (np.count_nonzero(mask_np) == 0)
                            except Exception:
                                mask_is_empty = True

                        if mask_is_empty:
                            # 掩码为空时仍输出纯黑掩码
                            src_img = imread_unicode(src_path, cv2.IMREAD_COLOR)
                            if src_img is None:
                                print(f"Warning: 无法读取源图像以生成纯黑掩码: {src_path}")
                                black = np.zeros((256, 256), dtype=np.uint8)
                            else:
                                h, w = src_img.shape[:2]
                                black = np.zeros((h, w), dtype=np.uint8)
                            ok = imwrite_unicode(mask_path, black)
                            if ok:
                                print(f"✓ 帧 {frame_idx} 掩码为空，已保存纯黑图: {mask_path}")
                            else:
                                print(f"✗ 保存纯黑掩码失败: {mask_path}")
                        else:
                            refined_mask = self.display_utils.refine_mask(mask_data, kernel_size=3, remove_small=50, smooth_outline=False)
                            # 转换到 0/255 的 uint8 单通道后再保存，避免保存成近黑的 0/1 浮点或布尔图
                            mask_u8 = to_uint8_mask(refined_mask)
                            # 使用 Unicode 兼容写入
                            ok = imwrite_unicode(mask_path, mask_u8)
                            if ok:
                                print(f"✓ 成功保存帧 {frame_idx} 的mask到 {mask_path}")
                            else:
                                print(f"✗ 保存帧 {frame_idx} 的mask失败: {mask_path}")

                        serializable_ann = {
                            'category_id': ann['category_id'],
                            'category_name': self.categories[ann['category_id']],
                            'object_id': ann['object_id'],
                            'is_keyframe': ann['is_keyframe'],
                            # 在JSON中也记录带相对层级的路径，便于区分不同人的序列
                            'mask_path': f"{self.sequence_rel_path.as_posix()}/{frame_filename}"
                        }
                        sequence_annotations[str(frame_idx)].append(serializable_ann)
            
            # 更新现有数据
            if 'sequences' not in existing_data:
                existing_data['sequences'] = {}
            
            # 以相对路径字符串作为键，确保不同人员/序列能正确区分
            existing_data['sequences'][self.sequence_rel_path.as_posix()] = {
                'num_frames': self.ct_loader.num_frames,
                'categories': self.categories,
                'keyframes': list(self.keyframes),
                'annotations': sequence_annotations
            }
            
            # 保存更新后的数据
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"标注结果已保存到: {json_path}")
            print(f"Mask文件保存到: {sequence_output_dir}")
            return True
            
        except Exception as e:
            print(f"保存标注失败: {e}")
            return False

    def cleanup_temp_cache(self):
        """清理临时缓存目录（如临时视频目录）"""
        try:
            tmp_dir = Path(self.output_root) / "temp_video"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
                print(f"已清理临时缓存目录: {tmp_dir}")
        except Exception as e:
            print(f"清理临时缓存目录失败: {e}")
    
    def add_current_object(self) -> bool:
        """添加当前对象到标注（类似于普通标注工具的添加对象功能）"""
        try:
            # 检查是否有当前的推理状态
            if not hasattr(self, 'inference_state') or self.inference_state is None:
                print("没有可用的推理状态")
                return False
            
            # 检查当前帧是否有临时的mask预测结果
            # 由于SAM2的架构，我们需要通过传播来获取当前帧的mask
            with torch.inference_mode():
                # 尝试获取当前帧的预测结果
                # 这里我们使用一个简化的方法：重新执行一次传播来获取当前帧的mask
                current_masks = []
                
                # 执行单帧传播来获取当前帧的mask
                for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                    self.inference_state, start_frame_idx=self.current_frame_idx, max_frame_num_to_track=1
                ):
                    if frame_idx == self.current_frame_idx and len(masks) > 0:
                        mask = masks[0].cpu().numpy()
                        
                        # 修复mask维度问题
                        if mask.ndim == 3:
                            mask = mask[0]
                        elif mask.ndim > 3:
                            mask = mask.squeeze()
                        
                        if mask.ndim != 2:
                            print(f"Warning: 无法将mask转换为2D，当前维度: {mask.shape}")
                            return False
                        
                        current_masks.append(mask)
                        break
                
                if len(current_masks) > 0:
                    mask = current_masks[0]
                    mask = self.display_utils.refine_mask(mask, kernel_size=3, remove_small=50, smooth_outline=False)
                    
                    # 添加到标注
                    if self.current_frame_idx not in self.annotations:
                        self.annotations[self.current_frame_idx] = []
                    
                    annotation = {
                        'category_id': self.current_category_id,
                        'mask': mask,
                        'object_id': self.current_object_id,
                        'is_keyframe': self.current_frame_idx in self.keyframes
                    }
                    
                    self.annotations[self.current_frame_idx].append(annotation)
                    print(f"成功添加对象到帧 {self.current_frame_idx}")
                    return True
                else:
                    print("当前帧没有可用的mask，请先添加点击标注")
                    return False
                    
        except Exception as e:
            print(f"添加当前对象失败: {e}")
            return False
    
    def get_frame_info(self) -> Dict:
        """获取当前帧信息"""
        return {
            'current_frame': self.current_frame_idx,
            'total_frames': self.ct_loader.num_frames,
            'current_category': self.categories[self.current_category_id],
            'is_keyframe': self.current_frame_idx in self.keyframes,
            'has_annotations': self.current_frame_idx in self.annotations
        }

    def _draw_click_history(self):
        """在显示图像上绘制点击历史"""
        # 只绘制当前帧的点击历史
        current_frame_clicks = [click for click in self.click_history 
                               if click['frame_idx'] == self.current_frame_idx]

        for click in current_frame_clicks:
            point = click['point']
            label = click['label']
            
            # 根据标签选择颜色和标记
            if label == 1:  # 正样本
                color = (0, 255, 0)  # 绿色
                marker = '+'
            else:  # 负样本
                color = (255, 0, 0)  # 红色
                marker = '-'
            
            # 绘制圆点
            cv2.circle(self.display_image, tuple(point), 1, color, -1)
            
            # 绘制标记符号
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.display_image, marker, 
                       (point[0] - 5, point[1] + 15), 
                       font, 0.3, color, 2)