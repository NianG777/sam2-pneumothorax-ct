#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气胸分割系统综合测试脚本
测试完整的分割流程：数据加载 → 提示生成 → SAM2分割 → 解剖学约束 → 可视化

主要测试功能：
1. 数据集加载
2. 多种提示生成策略
3. SAM2分割性能
4. 解剖学约束效果
5. 可视化结果生成
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

CURRENT_FILE = Path(__file__).resolve()
if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))

try:
    from font_config import setup_chinese_font

    setup_chinese_font(suppress_warnings=True)
except Exception:
    pass

from text_to_sam2_prompt import TextToSAM2Prompt, PneumothoraxInfo
from sam2_pneumothorax_segmenter import SAM2PneumothoraxSegmenter
from anatomical_constraints import AnatomicalConstraints


class PneumothoraxSegmentationTester:
    """气胸分割系统测试器"""

    def __init__(self, dataset_path: str, image_dir: str = "", output_dir: str = ""):
        """
        初始化测试器

        Args:
            dataset_path: 数据集JSON文件路径
            image_dir: 图像目录路径
            output_dir: 测试结果输出目录
        """
        self.dataset_path = dataset_path
        self.image_dir = image_dir
        self.output_dir = output_dir or str(CURRENT_FILE.parents[1] / "test_results")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化组件
        self.prompt_converter = TextToSAM2Prompt()
        self.segmenter = None  # 延迟初始化，避免GPU内存问题

        # 加载数据集
        self.dataset = self.load_dataset()
        print(f"数据集加载完成，共 {len(self.dataset)} 个样本")

    def load_dataset(self) -> List[Dict]:
        """加载数据集"""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            # 过滤有气胸的样本
            pneumo_samples = [
                sample for sample in dataset if sample.get("has_pneu", False)
            ]
            print(f"找到 {len(pneumo_samples)} 个气胸样本")

            return pneumo_samples

        except Exception as e:
            print(f"数据集加载失败: {e}")
            return []

    def test_prompt_generation(self, num_samples: int = 3) -> Dict:
        """测试提示生成功能"""
        print("\n" + "=" * 50)
        print("测试1: 提示生成功能")
        print("=" * 50)

        results = {
            "total_samples": num_samples,
            "prompt_strategies": [
                "bbox",
                "enhanced_bbox_constrained",
                "hybrid",
                "precision",
            ],
            "strategy_results": {},
        }

        # 随机选择样本
        test_samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))

        for strategy in results["prompt_strategies"]:
            results["strategy_results"][strategy] = {
                "successful": 0,
                "failed": 0,
                "details": [],
            }

        for i, sample in enumerate(test_samples):
            print(f"\n--- 样本 {i+1} ---")

            # 解析气胸信息
            pneumo_info = self.prompt_converter.parse_text_description(sample)
            if not pneumo_info:
                print(f"跳过样本 {i+1}：解析失败")
                continue

            image_shape = (512, 512)  # 假设标准尺寸

            # 测试不同的提示生成策略
            for strategy in results["prompt_strategies"]:
                try:
                    if strategy == "bbox":
                        prompts = self.prompt_converter.generate_bbox_prompt(
                            pneumo_info
                        )
                    elif strategy == "enhanced_bbox_constrained":
                        prompts = self.prompt_converter.generate_enhanced_bbox_constrained_points(
                            pneumo_info, num_points=8
                        )
                    elif strategy == "hybrid":
                        prompts = self.prompt_converter.generate_hybrid_prompt(
                            pneumo_info, image_shape
                        )
                    elif strategy == "precision":
                        prompts = self.prompt_converter.generate_precision_prompt(
                            pneumo_info, image_shape
                        )

                    print(f"  ✓ {strategy} 策略成功")
                    if isinstance(prompts, dict):
                        if "positive_points" in prompts:
                            print(f"    正样本点数: {len(prompts['positive_points'])}")
                        if "negative_points" in prompts:
                            print(f"    负样本点数: {len(prompts['negative_points'])}")
                        if "bbox" in prompts:
                            print(f"    边界框: {prompts['bbox']}")

                    results["strategy_results"][strategy]["successful"] += 1
                    results["strategy_results"][strategy]["details"].append(
                        {"sample_id": i + 1, "success": True, "prompts": prompts}
                    )

                except Exception as e:
                    print(f"  ✗ {strategy} 策略失败: {e}")
                    results["strategy_results"][strategy]["failed"] += 1
                    results["strategy_results"][strategy]["details"].append(
                        {"sample_id": i + 1, "success": False, "error": str(e)}
                    )

        # 打印策略统计
        print(f"\n提示生成测试完成")
        for strategy in results["prompt_strategies"]:
            strategy_result = results["strategy_results"][strategy]
            total = strategy_result["successful"] + strategy_result["failed"]
            if total > 0:
                success_rate = strategy_result["successful"] / total * 100
                print(
                    f"{strategy} 策略成功率: {success_rate:.1f}% ({strategy_result['successful']}/{total})"
                )

        return results

    def test_anatomical_constraints(self, num_samples: int = 2) -> Dict:
        """测试解剖学约束功能"""
        print("\n" + "=" * 50)
        print("测试2: 解剖学约束功能")
        print("=" * 50)

        results = {
            "total_samples": num_samples,
            "constraint_tests": [],
            "successful_constraints": 0,
            "failed_constraints": 0,
        }

        # 随机选择样本
        test_samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))

        for i, sample in enumerate(test_samples):
            print(f"\n--- 样本 {i+1} ---")

            try:
                # 解析气胸信息
                pneumo_info = self.prompt_converter.parse_text_description(sample)
                if not pneumo_info:
                    print(f"跳过样本 {i+1}：解析失败")
                    continue

                # 创建模拟掩码（实际应用中来自SAM2分割结果）
                image_shape = (512, 512)
                constraints = AnatomicalConstraints(image_shape)

                # 创建测试掩码
                test_mask = np.zeros(image_shape, dtype=np.uint8)
                x1, y1, x2, y2 = pneumo_info.bbox
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
                test_mask[y1:y2, x1:x2] = 255

                print(f"  原始掩码面积: {np.sum(test_mask > 0)} 像素")

                # 测试不同约束
                constraint_results = {}

                # 1. 面积约束
                area_constrained = constraints.apply_area_constraint(test_mask.copy())
                constraint_results["area"] = np.sum(area_constrained > 0)
                print(f"  面积约束后: {constraint_results['area']} 像素")

                # 2. 形态学约束
                morph_constrained = constraints.apply_morphological_constraint(
                    test_mask.copy()
                )
                constraint_results["morphological"] = np.sum(morph_constrained > 0)
                print(f"  形态学约束后: {constraint_results['morphological']} 像素")

                # 3. 连通性约束
                conn_constrained = constraints.apply_connectivity_constraint(
                    test_mask.copy()
                )
                constraint_results["connectivity"] = np.sum(conn_constrained > 0)
                print(f"  连通性约束后: {constraint_results['connectivity']} 像素")

                # 4. 位置编码约束（如果有）
                if pneumo_info.position_code:
                    pos_constrained = constraints.apply_position_code_constraint(
                        test_mask.copy(), pneumo_info.position_code, pneumo_info.bbox
                    )
                    constraint_results["position_code"] = np.sum(pos_constrained > 0)
                    print(
                        f"  位置编码约束后: {constraint_results['position_code']} 像素"
                    )

                # 5. 边界框约束
                bbox_constrained = constraints.apply_bbox_constraint(
                    test_mask.copy(), pneumo_info.bbox
                )
                constraint_results["bbox"] = np.sum(bbox_constrained > 0)
                print(f"  边界框约束后: {constraint_results['bbox']} 像素")

                print(f"  ✓ 解剖学约束测试成功")
                results["successful_constraints"] += 1
                results["constraint_tests"].append(
                    {
                        "sample_id": i + 1,
                        "success": True,
                        "original_area": np.sum(test_mask > 0),
                        "constraint_results": constraint_results,
                        "pneumo_info": pneumo_info,
                    }
                )

            except Exception as e:
                print(f"  ✗ 解剖学约束测试失败: {e}")
                results["failed_constraints"] += 1
                results["constraint_tests"].append(
                    {"sample_id": i + 1, "success": False, "error": str(e)}
                )

        success_rate = (
            results["successful_constraints"] / results["total_samples"] * 100
        )
        print(f"\n解剖学约束测试完成")
        print(
            f"成功率: {success_rate:.1f}% ({results['successful_constraints']}/{results['total_samples']})"
        )

        return results

    def test_end_to_end_segmentation(self, num_samples: int = 2) -> Dict:
        """测试端到端分割功能"""
        print("\n" + "=" * 50)
        print("测试3: 端到端分割功能")
        print("=" * 50)

        results = {
            "total_samples": num_samples,
            "successful_segmentations": 0,
            "failed_segmentations": 0,
            "segmentation_details": [],
        }

        # 初始化分割器（如果还没有初始化）
        if self.segmenter is None:
            try:
                print("初始化SAM2分割器...")
                self.segmenter = SAM2PneumothoraxSegmenter(
                    use_anatomical_constraints=True
                )
                print("SAM2分割器初始化成功")
            except Exception as e:
                print(f"SAM2分割器初始化失败: {e}")
                print("跳过端到端测试")
                return results

        # 随机选择样本
        test_samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))

        for i, sample in enumerate(test_samples):
            print(f"\n--- 样本 {i+1} ---")

            try:
                # 获取图像路径
                image_path = sample.get("image", "")
                if not os.path.exists(image_path):
                    print(f"图像文件不存在: {image_path}")
                    continue

                print(f"处理图像: {os.path.basename(image_path)}")

                # 执行端到端分割
                start_time = time.time()

                result = self.segmenter.process_single_case(
                    image_path=image_path,
                    text_or_data=sample,
                    output_dir=os.path.join(self.output_dir, f"sample_{i+1}"),
                    use_enhanced=True,
                    use_precision=False,
                )

                end_time = time.time()
                processing_time = end_time - start_time

                if result.get("success", False):
                    print(f"  ✓ 分割成功")
                    print(f"  处理时间: {processing_time:.2f} 秒")
                    print(f"  分割得分: {result.get('score', 0):.3f}")
                    print(
                        f"  掩码面积: {np.sum(result.get('mask', np.array([])) > 0)} 像素"
                    )

                    if "visualization_path" in result:
                        print(f"  可视化结果: {result['visualization_path']}")

                    results["successful_segmentations"] += 1
                    results["segmentation_details"].append(
                        {
                            "sample_id": i + 1,
                            "success": True,
                            "processing_time": processing_time,
                            "score": result.get("score", 0),
                            "mask_area": np.sum(result.get("mask", np.array([])) > 0),
                            "result": result,
                        }
                    )
                else:
                    error_msg = result.get("error", "未知错误")
                    print(f"  ✗ 分割失败: {error_msg}")
                    results["failed_segmentations"] += 1
                    results["segmentation_details"].append(
                        {"sample_id": i + 1, "success": False, "error": error_msg}
                    )

            except Exception as e:
                print(f"  ✗ 分割过程异常: {e}")
                results["failed_segmentations"] += 1
                results["segmentation_details"].append(
                    {"sample_id": i + 1, "success": False, "error": str(e)}
                )

        success_rate = (
            results["successful_segmentations"] / results["total_samples"] * 100
        )
        print(f"\n端到端分割测试完成")
        print(
            f"成功率: {success_rate:.1f}% ({results['successful_segmentations']}/{results['total_samples']})"
        )

        return results

    def test_compatibility(self, num_samples: int = 3) -> Dict:
        """测试兼容性功能"""
        print("\n" + "=" * 50)
        print("测试4: 兼容性功能")
        print("=" * 50)

        results = {
            "total_samples": num_samples,
            "structured_format_success": 0,
            "text_format_success": 0,
            "compatibility_details": [],
        }

        # 随机选择样本
        test_samples = random.sample(self.dataset, min(num_samples, len(self.dataset)))

        for i, sample in enumerate(test_samples):
            print(f"\n--- 样本 {i+1} ---")

            try:
                # 测试结构化格式解析
                structured_info = self.prompt_converter.parse_text_description(sample)
                structured_success = structured_info is not None

                # 测试文本格式解析（从caption提取）
                caption = sample.get("caption", "")
                text_info = self.prompt_converter.parse_text_description(caption)
                text_success = text_info is not None

                print(f"  结构化格式解析: {'✓' if structured_success else '✗'}")
                print(f"  文本格式解析: {'✓' if text_success else '✗'}")

                if structured_success:
                    results["structured_format_success"] += 1
                    print(f"    结构化 - 位置编码: {structured_info.position_code}")
                    print(f"    结构化 - 边界框: {structured_info.bbox}")

                if text_success:
                    results["text_format_success"] += 1
                    print(f"    文本 - 位置: {text_info.position}")
                    print(f"    文本 - 边界框: {text_info.bbox}")

                # 比较两种格式的结果
                if structured_success and text_success:
                    bbox_match = structured_info.bbox == text_info.bbox
                    print(f"    边界框匹配: {'✓' if bbox_match else '✗'}")

                results["compatibility_details"].append(
                    {
                        "sample_id": i + 1,
                        "structured_success": structured_success,
                        "text_success": text_success,
                        "structured_info": structured_info,
                        "text_info": text_info,
                    }
                )

            except Exception as e:
                print(f"  ✗ 兼容性测试异常: {e}")
                results["compatibility_details"].append(
                    {"sample_id": i + 1, "error": str(e)}
                )

        structured_rate = (
            results["structured_format_success"] / results["total_samples"] * 100
        )
        text_rate = results["text_format_success"] / results["total_samples"] * 100

        print(f"\n兼容性测试完成")
        print(
            f"结构化格式成功率: {structured_rate:.1f}% ({results['structured_format_success']}/{results['total_samples']})"
        )
        print(
            f"文本格式成功率: {text_rate:.1f}% ({results['text_format_success']}/{results['total_samples']})"
        )

        return results

    def generate_test_report(self, all_results: Dict) -> str:
        """生成测试报告"""
        report_path = os.path.join(self.output_dir, "test_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("气胸分割系统测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集路径: {self.dataset_path}\n")
            f.write(f"图像目录: {self.image_dir}\n")
            f.write(f"总样本数: {len(self.dataset)}\n\n")

            # 各项测试结果
            for test_name, result in all_results.items():
                f.write(f"{test_name}\n")
                f.write("-" * 30 + "\n")

                if "successful_parses" in result:
                    success_rate = (
                        result["successful_parses"] / result["total_samples"] * 100
                    )
                    f.write(f"成功率: {success_rate:.1f}%\n")
                    f.write(
                        f"成功: {result['successful_parses']}, 失败: {result['failed_parses']}\n"
                    )

                elif "successful_segmentations" in result:
                    success_rate = (
                        result["successful_segmentations"]
                        / result["total_samples"]
                        * 100
                    )
                    f.write(f"成功率: {success_rate:.1f}%\n")
                    f.write(
                        f"成功: {result['successful_segmentations']}, 失败: {result['failed_segmentations']}\n"
                    )

                elif "successful_constraints" in result:
                    success_rate = (
                        result["successful_constraints"] / result["total_samples"] * 100
                    )
                    f.write(f"成功率: {success_rate:.1f}%\n")
                    f.write(
                        f"成功: {result['successful_constraints']}, 失败: {result['failed_constraints']}\n"
                    )

                f.write("\n")

            f.write("测试完成\n")

        print(f"\n测试报告已保存到: {report_path}")
        return report_path

    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print("开始气胸分割系统综合测试")
        print("=" * 60)

        all_results = {}

        try:
            # 设置随机种子以确保可重现性
            random.seed(42)
            np.random.seed(42)

            # 1. 提示生成测试
            all_results["提示生成测试"] = self.test_prompt_generation(num_samples=3)

            # 2. 解剖学约束测试
            all_results["解剖学约束测试"] = self.test_anatomical_constraints(
                num_samples=2
            )

            # 3. 兼容性测试
            all_results["兼容性测试"] = self.test_compatibility(num_samples=3)

            # 4. 端到端分割测试（可选，需要GPU）
            try:
                all_results["端到端分割测试"] = self.test_end_to_end_segmentation(
                    num_samples=1
                )
            except Exception as e:
                print(f"端到端测试跳过: {e}")
                all_results["端到端分割测试"] = {"error": str(e)}

            # 生成测试报告
            self.generate_test_report(all_results)

            print("\n" + "=" * 60)
            print("所有测试完成！")
            print("=" * 60)

            return all_results

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    print("气胸分割系统综合测试")
    print("=" * 50)

    tester = PneumothoraxSegmentationTester(
        dataset_path=args.dataset_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
    )

    results = tester.run_all_tests()

    print("\n测试结果概览:")
    for test_name, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            print(f"- {test_name}: 完成")
        else:
            print(f"- {test_name}: 跳过或失败")


if __name__ == "__main__":
    main()
