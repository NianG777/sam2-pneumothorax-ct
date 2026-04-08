"""
CT图像文件名转换脚本
将IM0、IM1格式的文件重命名为00000001.png、00000002.png格式
并移动到第一级目录下，去掉中间的子目录结构
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def natural_sort_key(filename: str) -> List:
    """
    自然排序键函数，用于正确排序包含数字的文件名
    例如：IM0, IM1, IM2, ..., IM10, IM11 而不是 IM0, IM1, IM10, IM11, IM2
    """
    parts = re.split(r'(\d+)', filename)
    for i in range(len(parts)):
        if parts[i].isdigit():
            parts[i] = int(parts[i])
    return parts


def read_file_paths(file_path: str) -> List[str]:
    """读取文件路径列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths
    except Exception as e:
        print(f"读取文件路径失败: {e}")
        return []


def get_target_directory(relative_path: str) -> str:
    """
    获取目标目录（第一级目录）
    例如：20230714钱益龙-CT\ST0\SE1 -> 20230714钱益龙-CT
    """
    parts = relative_path.split('\\')
    if len(parts) > 0:
        return parts[0]
    return relative_path


def find_png_files(directory: Path) -> List[Path]:
    """查找目录下的PNG文件并按自然顺序排序"""
    if not directory.exists():
        return []
    
    png_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() == '.png':
            png_files.append(file)
    
    # 使用自然排序
    png_files.sort(key=lambda x: natural_sort_key(x.name))
    return png_files


def convert_and_move_files(base_dir: str, relative_paths: List[str], dry_run: bool = True) -> None:
    """
    转换并移动文件
    
    Args:
        base_dir: 基础目录路径 (G:\CT\qxfg\export-image)
        relative_paths: 相对路径列表
        dry_run: 是否为试运行模式（不实际执行操作）
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"基础目录不存在: {base_dir}")
        return
    
    total_processed = 0
    total_files = 0
    
    for relative_path in relative_paths:
        print(f"\n处理路径: {relative_path}")
        
        # 构建源目录路径
        source_dir = base_path / relative_path
        if not source_dir.exists():
            print(f"  ❌ 源目录不存在: {source_dir}")
            continue
        
        # 获取目标目录
        target_dir_name = get_target_directory(relative_path)
        target_dir = base_path / target_dir_name
        
        # 查找PNG文件
        png_files = find_png_files(source_dir)
        if not png_files:
            print(f"  ⚠️  未找到PNG文件")
            continue
        
        print(f"  📁 找到 {len(png_files)} 个PNG文件")
        print(f"  🎯 目标目录: {target_dir}")
        
        # 创建目标目录
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个文件
        for i, png_file in enumerate(png_files, 1):
            # 生成新文件名
            new_filename = f"{i:08d}.png"  # 8位数字，前面补0
            target_file = target_dir / new_filename
            
            print(f"    {png_file.name} -> {new_filename}")
            
            if not dry_run:
                try:
                    # 如果目标文件已存在，先删除
                    if target_file.exists():
                        target_file.unlink()
                    
                    # 移动文件到目标位置（而不是复制）
                    shutil.move(str(png_file), str(target_file))
                    total_processed += 1
                    
                except Exception as e:
                    print(f"    ❌ 处理失败: {e}")
            else:
                total_processed += 1
        
        total_files += len(png_files)
    
    print(f"\n{'=' * 50}")
    if dry_run:
        print(f"🔍 试运行完成:")
        print(f"  - 将处理 {total_files} 个文件")
        print(f"  - 涉及 {len(relative_paths)} 个目录")
        print(f"\n如果确认无误，请设置 dry_run=False 执行实际操作")
    else:
        print(f"✅ 转换完成:")
        print(f"  - 成功处理 {total_processed}/{total_files} 个文件")
        print(f"  - 涉及 {len(relative_paths)} 个目录")


def main():
    """主函数"""
    # 配置参数
    BASE_DIR = r"G:\CT\qxfg\export-image"
    FILE_PATH_TXT = r"d:\develops\VScode\code\qxfg\qx_tools\sam-pneumothorax-segmentation\file_path.txt"
    
    print("CT图像文件名转换脚本")
    print("=" * 50)
    print(f"基础目录: {BASE_DIR}")
    print(f"路径文件: {FILE_PATH_TXT}")
    
    # 读取文件路径
    relative_paths = read_file_paths(FILE_PATH_TXT)
    if not relative_paths:
        print("未找到有效的文件路径")
        return
    
    print(f"找到 {len(relative_paths)} 个路径需要处理")
    
    # 显示前几个路径作为示例
    print("\n路径示例:")
    for i, path in enumerate(relative_paths[:5]):
        target_dir = get_target_directory(path)
        print(f"  {i+1}. {path} -> {target_dir}")
    if len(relative_paths) > 5:
        print(f"  ... 还有 {len(relative_paths) - 5} 个路径")
    
    # 询问用户是否继续
    print(f"\n⚠️  注意事项:")
    print(f"  - 文件将从子目录移动到第一级目录")
    print(f"  - IM格式文件名将转换为8位数字格式")
    print(f"  - 如果目标文件已存在将被覆盖")
    
    while True:
        choice = input(f"\n请选择操作:\n  1. 试运行（查看将要执行的操作）\n  2. 正式执行\n  3. 退出\n请输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            print(f"\n🔍 开始试运行...")
            convert_and_move_files(BASE_DIR, relative_paths, dry_run=True)
            break
        elif choice == '2':
            confirm = input(f"\n⚠️  确认要执行文件转换操作吗？(输入 'yes' 确认): ").strip().lower()
            if confirm == 'yes':
                print(f"\n🚀 开始正式执行...")
                convert_and_move_files(BASE_DIR, relative_paths, dry_run=False)
            else:
                print("操作已取消")
            break
        elif choice == '3':
            print("退出程序")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()