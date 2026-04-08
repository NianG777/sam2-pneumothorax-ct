"""
Mask调试和验证工具
用于诊断和修复CT序列分割中的mask导出问题
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_mask_data(mask, frame_idx=None, object_id=None):
    """
    分析mask数据的详细信息
    
    Args:
        mask: numpy数组，mask数据
        frame_idx: 帧索引（可选）
        object_id: 对象ID（可选）
    
    Returns:
        dict: 包含mask分析结果的字典
    """
    info = {
        'shape': mask.shape,
        'dtype': str(mask.dtype),
        'ndim': mask.ndim,
        'min_value': float(mask.min()),
        'max_value': float(mask.max()),
        'mean_value': float(mask.mean()),
        'unique_values': len(np.unique(mask)),
        'non_zero_count': int(np.count_nonzero(mask)),
        'total_pixels': int(mask.size),
        'frame_idx': frame_idx,
        'object_id': object_id
    }
    
    # 计算非零像素比例
    info['non_zero_ratio'] = info['non_zero_count'] / info['total_pixels']
    
    # 检查是否为二值mask
    unique_vals = np.unique(mask)
    info['is_binary'] = len(unique_vals) <= 2
    info['unique_values_list'] = unique_vals.tolist()
    
    return info


def validate_mask_conversion(original_mask, converted_mask):
    """
    验证mask转换是否正确
    
    Args:
        original_mask: 原始mask数据
        converted_mask: 转换后的mask数据
    
    Returns:
        dict: 验证结果
    """
    result = {
        'conversion_valid': True,
        'issues': []
    }
    
    # 检查形状是否一致
    if original_mask.shape != converted_mask.shape:
        result['conversion_valid'] = False
        result['issues'].append(f"形状不匹配: {original_mask.shape} -> {converted_mask.shape}")
    
    # 检查转换后是否全为0
    if converted_mask.max() == 0:
        result['conversion_valid'] = False
        result['issues'].append("转换后mask全为0")
    
    # 检查转换后是否全为同一值
    if converted_mask.min() == converted_mask.max() and converted_mask.max() > 0:
        result['conversion_valid'] = False
        result['issues'].append(f"转换后mask全为同一值: {converted_mask.min()}")
    
    # 检查数据类型
    if converted_mask.dtype != np.uint8:
        result['conversion_valid'] = False
        result['issues'].append(f"转换后数据类型不正确: {converted_mask.dtype}")
    
    # 检查值范围
    if converted_mask.max() > 255 or converted_mask.min() < 0:
        result['conversion_valid'] = False
        result['issues'].append(f"转换后值范围超出0-255: [{converted_mask.min()}, {converted_mask.max()}]")
    
    return result


def fix_mask_format(mask):
    """
    修复mask格式，确保正确转换为可保存的图像格式
    
    Args:
        mask: 输入的mask数据
    
    Returns:
        numpy.ndarray: 修复后的uint8格式mask
    """
    # 确保是numpy数组
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    # 确保是2D数组
    if mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            # 如果还是多维，取第一个通道
            mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
    
    # 根据数据类型进行转换
    if mask.dtype == bool:
        # 布尔类型：True->255, False->0
        mask_uint8 = mask.astype(np.uint8) * 255
    elif mask.dtype in [np.float32, np.float64]:
        # 浮点类型：确保在0-1范围内，然后转换为0-255
        mask_normalized = np.clip(mask, 0, 1)
        mask_uint8 = (mask_normalized * 255).astype(np.uint8)
    elif mask.dtype == np.uint8:
        # 已经是uint8类型，检查值范围
        if mask.max() <= 1:
            # 如果值在0-1范围内，需要放大到0-255
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            # 已经在0-255范围内
            mask_uint8 = mask
    else:
        # 其他数据类型，先转换为float再处理
        mask_float = mask.astype(np.float32)
        mask_normalized = np.clip(mask_float, 0, 1)
        mask_uint8 = (mask_normalized * 255).astype(np.uint8)
    
    return mask_uint8


def save_mask_with_validation(mask, save_path, frame_idx=None, object_id=None):
    """
    保存mask并进行验证
    
    Args:
        mask: mask数据
        save_path: 保存路径
        frame_idx: 帧索引（可选）
        object_id: 对象ID（可选）
    
    Returns:
        dict: 保存结果和验证信息
    """
    result = {
        'success': False,
        'original_info': None,
        'converted_info': None,
        'validation': None,
        'save_path': str(save_path)
    }
    
    try:
        # 分析原始mask
        result['original_info'] = analyze_mask_data(mask, frame_idx, object_id)
        
        # 修复mask格式
        mask_fixed = fix_mask_format(mask)
        
        # 分析转换后的mask
        result['converted_info'] = analyze_mask_data(mask_fixed, frame_idx, object_id)
        
        # 验证转换
        result['validation'] = validate_mask_conversion(mask, mask_fixed)
        
        # 保存mask
        success = cv2.imwrite(str(save_path), mask_fixed)
        result['success'] = success
        
        if not success:
            result['error'] = f"无法保存mask到 {save_path}"
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_mask_visualization(mask, title="Mask Visualization", save_path=None):
    """
    创建mask的可视化图像
    
    Args:
        mask: mask数据
        title: 图像标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    
    # 原始mask
    plt.subplot(2, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title(f'Original Mask\nShape: {mask.shape}, Type: {mask.dtype}')
    plt.colorbar()
    
    # 修复后的mask
    mask_fixed = fix_mask_format(mask)
    plt.subplot(2, 2, 2)
    plt.imshow(mask_fixed, cmap='gray')
    plt.title(f'Fixed Mask\nShape: {mask_fixed.shape}, Type: {mask_fixed.dtype}')
    plt.colorbar()
    
    # 直方图
    plt.subplot(2, 2, 3)
    plt.hist(mask.flatten(), bins=50, alpha=0.7, label='Original')
    plt.hist(mask_fixed.flatten(), bins=50, alpha=0.7, label='Fixed')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Pixel Value Distribution')
    plt.legend()
    
    # 统计信息
    plt.subplot(2, 2, 4)
    info_text = f"""
    Original:
    Min: {mask.min():.3f}
    Max: {mask.max():.3f}
    Mean: {mask.mean():.3f}
    Non-zero: {np.count_nonzero(mask)}
    
    Fixed:
    Min: {mask_fixed.min()}
    Max: {mask_fixed.max()}
    Mean: {mask_fixed.mean():.1f}
    Non-zero: {np.count_nonzero(mask_fixed)}
    """
    plt.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    plt.title('Statistics')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化图像已保存到: {save_path}")
    
    plt.show()


def batch_validate_masks(mask_dir):
    """
    批量验证mask目录中的所有图像
    
    Args:
        mask_dir: mask图像目录路径
    
    Returns:
        dict: 验证结果统计
    """
    mask_dir = Path(mask_dir)
    results = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'issues': []
    }
    
    for mask_file in mask_dir.glob('*.jpg'):
        results['total_files'] += 1
        
        try:
            # 读取mask图像
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                results['invalid_files'] += 1
                results['issues'].append(f"{mask_file.name}: 无法读取图像")
                continue
            
            # 检查图像内容
            if mask_img.max() == 0:
                results['invalid_files'] += 1
                results['issues'].append(f"{mask_file.name}: 图像全为黑色")
            elif mask_img.min() == mask_img.max():
                results['invalid_files'] += 1
                results['issues'].append(f"{mask_file.name}: 图像为纯色 (值: {mask_img.min()})")
            else:
                results['valid_files'] += 1
                
        except Exception as e:
            results['invalid_files'] += 1
            results['issues'].append(f"{mask_file.name}: 处理错误 - {str(e)}")
    
    return results