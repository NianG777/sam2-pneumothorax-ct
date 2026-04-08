"""CT序列视频分割标注工具
基于SAM2的视频分割功能，将CT切片序列作为视频帧进行处理
支持关键帧标注和时序传播，减少逐帧操作工作量

使用方法:
python ct_sequence_annotator.py --dataset_path /path/to/ct/images --categories 气胸 肺部 心脏

操作说明:
1. 左键点击添加正样本点，右键点击添加负样本点
2. 在关键帧上添加标注后，点击"传播标注"进行时序传播
3. 使用A/D键或按钮导航帧，R键重置当前帧
4. Ctrl+S保存标注结果
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
# import os
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.fonts=false'   #HYDRA_FULL_ERROR=1
os.environ['HYDRA_FULL_ERROR'] = '1'   #HYDRA_FULL_ERROR=1
# 添加SAM2路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sam2_path = os.path.join(current_dir, '..', 'sam2-main')
sys.path.append(sam2_path)

from salt.ct_video_editor import CTVideoEditor
from salt.ct_video_interface import CTVideoInterface


def check_dependencies():
    """检查依赖项"""
    try:
        import torch
        import cv2
        import numpy as np
        from PyQt5 import QtWidgets
        print("✓ 基础依赖检查通过")
        return True
    except ImportError as e:
        print(f"✗ 依赖项缺失: {e}")
        return False


def check_sam2_installation():
    """检查SAM2安装"""
    try:
        from sam2.build_sam import build_sam2_video_predictor
        print("✓ SAM2安装检查通过")
        return True
    except ImportError as e:
        print(f"✗ SAM2未正确安装: {e}")
        print("请确保SAM2已正确安装并且路径正确")
        return False


def validate_selected_dir(images_dir, sam2_checkpoint):
    """验证用户选择的图像目录与SAM2检查点"""
    if not images_dir or not os.path.exists(images_dir):
        print(f"✗ 选择的图像目录不存在: {images_dir}")
        return False

    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    if len(image_files) == 0:
        print(f"✗ 在所选目录中未找到PNG/JPG图像文件: {images_dir}")
        return False

    print(f"✓ 找到图像序列: {len(image_files)} 个文件，目录: {images_dir}")

    if not os.path.exists(sam2_checkpoint):
        print(f"✗ SAM2检查点文件不存在: {sam2_checkpoint}")
        print("请下载SAM2模型检查点文件")
        return False

    print("✓ 路径验证通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CT序列视频分割标注工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python ct_sequence_annotator.py --dataset_root /path/to/dataset --sequence_name 20230309江能斌-CT
  python ct_sequence_annotator.py --dataset_root /path/to/dataset --sequence_name 20230309江能斌-CT --categories 气胸 肺部 心脏
  python ct_sequence_annotator.py --dataset_root /path/to/dataset --sequence_name 20230309江能斌-CT --sam2_checkpoint /path/to/sam2_model.pt

数据集结构:
  dataset/
  ├── images/
  │   ├── 20230309江能斌-CT/
  │   │   ├── 00000001.png
  │   │   └── ...
  │   └── ...
  ├── embeddings/
  │   ├── 20230309江能斌-CT/
  │   │   ├── 00000001.npy
  │   │   └── ...
  │   └── ...

输出结构:
  ct_annotations/
  ├── annotations.json
  ├── 20230309江能斌-CT/
  │   ├── 00000001.png
  │   └── ...
  └── ...

操作说明:
  • 左键点击：添加正样本点
  • 右键点击：添加负样本点  
  • A/D键：上一帧/下一帧
  • R键：重置当前帧
  • Ctrl+S：保存标注
  • ESC：退出程序
        """
    )
    
    # 移除 dataset_root 与 sequence_name 参数，启动时由用户选择图像目录
    parser.add_argument('--sam2_checkpoint', type=str, 
                       default=r'D:\develops\VScode\code\qxfg\qx_tools\sam2.1_hiera_large.pt')
                       
    parser.add_argument('--sam2_config', type=str,
                       default=r'D:\develops\VScode\code\qxfg\qx_tools\sam2-main\sam2\configs\sam2.1\sam2.1_hiera_l.yaml',
                       help='SAM2配置文件名')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['气胸', '肺部', '心脏'],
                       help='标注类别列表')
    parser.add_argument('--output_dir', type=str,
                       default=r'G:\CT\qxfg\SAM-Tool-main\ct_annotations',
                       help='标注结果输出目录')
    parser.add_argument('--device', type=str,
                       default='cuda',
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("CT序列视频分割标注工具启动中...")
    print("=" * 50)
    
    # 检查依赖项
    if not check_dependencies():
        print("请安装必要的依赖项后重试")
        sys.exit(1)
    
    if not check_sam2_installation():
        print("请正确安装SAM2后重试")
        sys.exit(1)
    
    # 启动后让用户选择图像序列目录
    app = QApplication(sys.argv)
    app.setApplicationName("CT序列视频分割标注工具")

    # 先选择数据总路径（用于在保存时保留相对目录结构，例如包含"人名"的上级目录）
    base_root_dir = QFileDialog.getExistingDirectory(None, "请选择数据总路径（例如 F:\\qxfg\\export-image）")
    if not base_root_dir:
        print("未选择数据总路径，将仅使用序列目录名称进行保存分类。")

    images_dir = QFileDialog.getExistingDirectory(None, "请选择包含PNG/JPG序列的目录")
    if not images_dir:
        print("已取消选择目录，程序退出")
        sys.exit(0)

    # 验证选择的目录与SAM2检查点
    if not validate_selected_dir(images_dir, args.sam2_checkpoint):
        print("路径验证失败，请重新选择目录或检查SAM2文件")
        sys.exit(1)
    
    # 从选择的目录推断 sequence_name（用于输出命名）
    sequence_name = os.path.basename(images_dir.rstrip("/\\"))
    
    try:
        print("正在初始化编辑器...")
        
        # 初始化编辑器
        editor = CTVideoEditor(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            dataset_root=None,
            sequence_name=sequence_name,
            output_root=args.output_dir,
            categories=args.categories,
            device=args.device,
            images_path=images_dir,
            base_data_root=base_root_dir if base_root_dir else None
        )
        
        print("正在创建用户界面...")
        
        # 创建界面
        interface = CTVideoInterface(app, editor)
        # 将基础数据根路径传递给界面，便于后续切换序列时沿用
        try:
            interface.base_data_root = base_root_dir if base_root_dir else None
        except Exception:
            pass
        interface.show()
        
        print("✓ 启动成功！")
        print("=" * 50)
        print("使用说明:")
        print("• 左键点击添加正样本点，右键点击添加负样本点")
        print("• 在关键帧上标注后，点击'传播标注'进行时序传播")
        print("• 使用A/D键导航帧，R键重置，Ctrl+S保存")
        print("• ESC键退出程序")
        print("=" * 50)
        
        # 退出时清理临时缓存目录
        try:
            app.aboutToQuit.connect(editor.cleanup_temp_cache)
        except Exception:
            pass

        # 运行应用
        sys.exit(app.exec_()) 
        
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        
        # 显示错误对话框
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("启动错误")
        msg.setText(f"程序启动失败:\n{str(e)}")
        msg.setDetailedText(f"错误详情:\n{str(e)}")
        msg.exec_()
        
        sys.exit(1)

if __name__ == "__main__":
    main()