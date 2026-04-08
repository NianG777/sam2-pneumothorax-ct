#!/usr/bin/env python3
"""
分析CT序列标注文件的内容和统计信息
"""
import json
import os

def analyze_annotations(json_path):
    """分析标注文件"""
    print(f"正在分析标注文件: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"错误: 文件不存在 {json_path}")
        return
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取JSON文件 - {e}")
        return
    
    # 基本信息
    print("\n=== 基本信息 ===")
    sequences = data.get('sequences', {})
    print(f"序列数量: {len(sequences)}")
    
    for seq_name, seq_data in sequences.items():
        print(f"\n序列名称: {seq_name}")
        print(f"总帧数: {seq_data.get('num_frames', 0)}")
        print(f"类别: {seq_data.get('categories', [])}")
        
        keyframes = seq_data.get('keyframes', [])
        print(f"关键帧列表: {sorted(keyframes)}")
        print(f"关键帧数量: {len(keyframes)}")
        
        # 分析标注数据
        annotations = seq_data.get('annotations', {})
        print(f"\n=== 标注统计 ===")
        print(f"标注帧数量: {len(annotations)}")
        
        # 统计关键帧和非关键帧
        keyframe_count = 0
        non_keyframe_count = 0
        total_annotations = 0
        frame_annotation_counts = {}
        
        for frame_id, frame_annotations in annotations.items():
            frame_count = len(frame_annotations)
            total_annotations += frame_count
            frame_annotation_counts[frame_id] = frame_count
            
            keyframes_in_frame = 0
            for ann in frame_annotations:
                if ann.get('is_keyframe', False):
                    keyframe_count += 1
                    keyframes_in_frame += 1
                else:
                    non_keyframe_count += 1
        
        print(f"总标注数量: {total_annotations}")
        print(f"关键帧标注数量: {keyframe_count}")
        print(f"非关键帧标注数量: {non_keyframe_count}")
        
        # 检查一些具体帧的标注
        print(f"\n=== 部分帧的标注详情 ===")
        sample_frames = ['29', '37', '72', '100', '105', '106', '107', '108']
        for frame_id in sample_frames:
            if frame_id in annotations:
                frame_anns = annotations[frame_id]
                keyframes_in_frame = sum(1 for ann in frame_anns if ann.get('is_keyframe', False))
                print(f"  帧 {frame_id}: {len(frame_anns)} 个标注, {keyframes_in_frame} 个关键帧标注")
        
        # 检查标注数据的一致性
        print(f"\n=== 数据一致性检查 ===")
        
        # 检查每帧的标注数量分布
        annotation_counts = list(frame_annotation_counts.values())
        if annotation_counts:
            min_count = min(annotation_counts)
            max_count = max(annotation_counts)
            avg_count = sum(annotation_counts) / len(annotation_counts)
            print(f"每帧标注数量 - 最小: {min_count}, 最大: {max_count}, 平均: {avg_count:.2f}")
        
        # 检查mask路径格式
        print(f"\n=== Mask路径检查 ===")
        mask_paths = set()
        for frame_id, frame_annotations in annotations.items():
            for ann in frame_annotations:
                mask_path = ann.get('mask_path', '')
                mask_paths.add(mask_path)
        
        print(f"唯一mask路径数量: {len(mask_paths)}")
        
        # 显示前几个mask路径作为示例
        sample_paths = sorted(list(mask_paths))[:10]
        print("示例mask路径:")
        for path in sample_paths:
            print(f"  {path}")
        
        # 检查关键帧标记的一致性
        print(f"\n=== 关键帧一致性检查 ===")
        keyframe_list = set(keyframes)
        marked_keyframes = set()
        
        for frame_id, frame_annotations in annotations.items():
            for ann in frame_annotations:
                if ann.get('is_keyframe', False):
                    marked_keyframes.add(int(frame_id))
        
        print(f"keyframes列表中的关键帧: {sorted(keyframe_list)}")
        print(f"标注中标记的关键帧: {sorted(marked_keyframes)}")
        
        # 检查是否一致
        if keyframe_list == marked_keyframes:
            print("✓ 关键帧标记一致")
        else:
            missing_in_annotations = keyframe_list - marked_keyframes
            extra_in_annotations = marked_keyframes - keyframe_list
            if missing_in_annotations:
                print(f"⚠ keyframes列表中有但标注中缺失的关键帧: {sorted(missing_in_annotations)}")
            if extra_in_annotations:
                print(f"⚠ 标注中有但keyframes列表中缺失的关键帧: {sorted(extra_in_annotations)}")

if __name__ == "__main__":
    json_path = "d:/develops/VScode/code/qxfg/qx_tools/dataset/test/annotations.json"
    analyze_annotations(json_path)