#!/usr/bin/env python3
"""
SAM2气胸分割器
整合文本描述解析、prompt生成和SAM2分割的完整流程

主要功能：
1. 加载SAM2模型
2. 解析文本描述生成提示
3. 执行图像分割
4. 生成可视化结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
DEFAULT_SAM2_DIR = PROJECT_ROOT.parent / "sam2-main"

if str(CURRENT_FILE.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_FILE.parent))
if DEFAULT_SAM2_DIR.exists() and str(DEFAULT_SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SAM2_DIR))

try:
    from font_config import setup_chinese_font

    setup_chinese_font(suppress_warnings=True)
except Exception:
    pass

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from text_to_sam2_prompt import TextToSAM2Prompt, PneumothoraxInfo
from anatomical_constraints import AnatomicalConstraints


class SAM2PneumothoraxSegmenter:
    """SAM2气胸分割器"""

    def __init__(
        self,
        model_cfg: Optional[str] = None,
        model_ckpt: Optional[str] = None,
        sam2_repo_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_anatomical_constraints: bool = True,
    ):
        """
        初始化SAM2气胸分割器

        Args:
            model_cfg: SAM2模型配置文件路径
            model_ckpt: SAM2模型权重文件路径
            device: 计算设备
            use_anatomical_constraints: 是否使用解剖学约束
        """
        self.device = device
        env_sam2_dir = os.getenv("SAM2_REPO_DIR")
        self.sam2_dir = str(
            Path(sam2_repo_dir or env_sam2_dir or DEFAULT_SAM2_DIR).resolve()
        )
        self.use_anatomical_constraints = use_anatomical_constraints

        default_cfg = (
            Path(self.sam2_dir) / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
        )
        default_ckpt = Path(os.getenv("SAM2_CHECKPOINT", ""))
        if not default_ckpt.exists():
            default_ckpt = Path(self.sam2_dir) / "checkpoints" / "sam2.1_hiera_large.pt"

        self.model_cfg = (
            str(Path(model_cfg).resolve()) if model_cfg else str(default_cfg)
        )
        self.model_ckpt = (
            str(Path(model_ckpt).resolve()) if model_ckpt else str(default_ckpt)
        )

        # 初始化文本到提示转换器
        self.prompt_converter = TextToSAM2Prompt()

        # 初始化解剖学约束处理器（延迟初始化）
        self.anatomical_constraints = None

        # 初始化SAM2模型
        self.predictor = None
        self._load_model()

        print(f"SAM2气胸分割器初始化完成")
        print(f"设备: {self.device}")
        print(f"解剖学约束: {'启用' if self.use_anatomical_constraints else '禁用'}")

    def _load_model(self):
        """加载SAM2模型"""
        try:
            print(f"正在加载SAM2模型...")
            print(f"配置文件: {self.model_cfg}")
            print(f"权重文件: {self.model_ckpt}")

            if Path(self.sam2_dir).exists() and self.sam2_dir not in sys.path:
                sys.path.insert(0, self.sam2_dir)

            if not os.path.exists(self.model_ckpt):
                print("本地权重文件不存在，尝试使用Hugging Face预训练模型...")
                self.predictor = SAM2ImagePredictor.from_pretrained(
                    "facebook/sam2.1-hiera-large", device=self.device
                )
            else:
                current_dir = os.getcwd()
                sam2_dir = self.sam2_dir

                try:
                    os.chdir(sam2_dir)
                    print(f"切换到SAM2目录: {sam2_dir}")

                    config_name = self.model_cfg
                    config_path = Path(config_name)
                    if config_path.exists():
                        config_name = str(config_path.resolve())
                    else:
                        candidate_relative = Path("sam2") / config_name
                        if candidate_relative.exists():
                            config_name = config_name
                        else:
                            config_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    print(f"使用相对配置路径: {config_name}")

                    sam2_model = build_sam2(
                        config_name, self.model_ckpt, device=self.device
                    )
                    self.predictor = SAM2ImagePredictor(sam2_model)

                finally:
                    # 恢复原始工作目录
                    os.chdir(current_dir)

            print("SAM2模型加载成功！")

        except Exception as e:
            print(f"SAM2模型加载失败: {e}")
            print("尝试使用CPU版本...")
            try:
                self.device = "cpu"
                self.predictor = SAM2ImagePredictor.from_pretrained(
                    "facebook/sam2.1-hiera-large", device=self.device
                )
                print("CPU版本SAM2模型加载成功！")
            except Exception as e2:
                raise RuntimeError(f"无法加载SAM2模型: {e2}")

    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # 使用OpenCV加载图像（BGR格式）
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def parse_text_and_generate_prompts(
        self,
        text_or_data: Union[str, Dict],
        image_shape: Tuple[int, int],
        use_enhanced: bool = True,
        use_precision: bool = False,
        use_bbox_constrained_points: bool = False,
        image: np.ndarray = None,
    ) -> Dict:
        """解析文本描述并生成SAM2提示"""
        # 解析文本描述
        pneumo_info = self.prompt_converter.parse_text_description(text_or_data)
        if pneumo_info is None:
            raise ValueError("无法从文本描述中提取气胸信息")

        # 根据数据类型选择提示生成策略
        if (
            hasattr(pneumo_info, "has_structured_data")
            and pneumo_info.has_structured_data
            and use_enhanced
        ):
            # 使用新的增强提示生成方法
            prompts = self.prompt_converter.generate_enhanced_bbox_constrained_points(
                pneumo_info, image, num_points=8 if use_precision else 6
            )
        elif use_bbox_constrained_points:
            # 使用新的边界框约束点策略
            prompts = self.prompt_converter.generate_bbox_constrained_points(
                pneumo_info, image=image, num_points=8
            )
        elif use_precision:
            # 使用精确分割方法（减少过度分割）
            prompts = self.prompt_converter.generate_precision_prompt(
                pneumo_info, image_shape
            )
        elif use_enhanced:
            # 使用增强的混合提示（包含区域负样本约束）
            prompts = self.prompt_converter.generate_enhanced_hybrid_prompt(
                pneumo_info, image_shape
            )
        else:
            # 使用标准混合提示
            prompts = self.prompt_converter.generate_hybrid_prompt(
                pneumo_info, image_shape
            )

        return {"pneumo_info": pneumo_info, "prompts": prompts}

    def segment_pneumothorax(
        self,
        image: np.ndarray,
        prompts: Dict,
        apply_bbox_constraint: bool = True,
        apply_anatomical_constraint: bool = None,
    ) -> Dict:
        """
        使用SAM2进行气胸分割

        Args:
            image: 输入图像
            prompts: 提示信息
            apply_bbox_constraint: 是否应用边界框约束
            apply_anatomical_constraint: 是否应用解剖学约束（None时使用初始化设置）
        """
        # 设置图像
        self.predictor.set_image(image)

        # 确定是否应用解剖学约束
        if apply_anatomical_constraint is None:
            apply_anatomical_constraint = self.use_anatomical_constraints

        # 延迟初始化解剖学约束处理器
        if (
            apply_anatomical_constraint
            and self.anatomical_constraints is None
            and self.use_anatomical_constraints
        ):
            image_shape = (image.shape[0], image.shape[1])  # (height, width)
            self.anatomical_constraints = AnatomicalConstraints(image_shape)

        # 准备提示参数
        prompt_data = prompts["prompts"]

        # 提取提示信息
        point_coords = None
        point_labels = None
        box = None

        if "positive_points" in prompt_data and len(prompt_data["positive_points"]) > 0:
            positive_points = np.array(prompt_data["positive_points"])
            positive_labels = np.ones(len(positive_points), dtype=int)

            if (
                "negative_points" in prompt_data
                and len(prompt_data["negative_points"]) > 0
            ):
                negative_points = np.array(prompt_data["negative_points"])
                negative_labels = np.zeros(len(negative_points), dtype=int)

                point_coords = np.vstack([positive_points, negative_points])
                point_labels = np.concatenate([positive_labels, negative_labels])
            else:
                point_coords = positive_points
                point_labels = positive_labels

        if "bbox" in prompt_data:
            box = np.array(prompt_data["bbox"])

        # 执行分割
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

            # 选择最佳掩码（得分最高的）
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]

            # 应用边界框约束（如果启用且有气胸信息）
            if apply_bbox_constraint and "pneumo_info" in prompts:
                pneumo_info = prompts["pneumo_info"]
                best_mask = self.prompt_converter.apply_bbox_constraint_to_mask(
                    best_mask, pneumo_info
                )

                # 对所有掩码应用约束
                constrained_masks = []
                for mask in masks:
                    constrained_mask = (
                        self.prompt_converter.apply_bbox_constraint_to_mask(
                            mask, pneumo_info
                        )
                    )
                    constrained_masks.append(constrained_mask)
                masks = constrained_masks  # 保持为列表，避免类型转换问题

            # 应用解剖学约束（如果启用）
            if apply_anatomical_constraint and self.anatomical_constraints is not None:
                # 检查是否有结构化数据，选择相应的约束方法
                if "pneumo_info" in prompts:
                    pneumo_info = prompts["pneumo_info"]
                    if (
                        hasattr(pneumo_info, "has_structured_data")
                        and pneumo_info.has_structured_data
                    ):
                        print("使用结构化约束...")
                        # 对最佳掩码应用结构化约束
                        best_mask = (
                            self.anatomical_constraints.apply_structured_constraints(
                                best_mask, pneumo_info, image
                            )
                        )

                        # 对所有掩码应用结构化约束
                        anatomically_constrained_masks = []
                        for mask in masks:
                            constrained_mask = self.anatomical_constraints.apply_structured_constraints(
                                mask, pneumo_info, image
                            )
                            anatomically_constrained_masks.append(constrained_mask)
                        masks = anatomically_constrained_masks
                    else:
                        print("使用传统约束...")
                        # 对最佳掩码应用传统约束
                        best_mask = self.anatomical_constraints.apply_all_constraints(
                            best_mask, image
                        )

                        # 对所有掩码应用传统约束
                        anatomically_constrained_masks = []
                        for mask in masks:
                            constrained_mask = (
                                self.anatomical_constraints.apply_all_constraints(
                                    mask, image
                                )
                            )
                            anatomically_constrained_masks.append(constrained_mask)
                        masks = anatomically_constrained_masks
                else:
                    # 传统约束方法
                    best_mask = self.anatomical_constraints.apply_all_constraints(
                        best_mask, image
                    )

                    anatomically_constrained_masks = []
                    for mask in masks:
                        constrained_mask = (
                            self.anatomical_constraints.apply_all_constraints(
                                mask, image
                            )
                        )
                        anatomically_constrained_masks.append(constrained_mask)
                    masks = anatomically_constrained_masks

            return {
                "mask": best_mask,
                "score": best_score,
                "all_masks": masks,
                "all_scores": scores,
                "logits": logits,
                "bbox_constrained": apply_bbox_constraint,
                "anatomically_constrained": apply_anatomical_constraint,
            }

        except Exception as e:
            raise RuntimeError(f"SAM2分割失败: {e}")

    def visualize_results(
        self,
        image: np.ndarray,
        pneumo_info: PneumothoraxInfo,
        mask: np.ndarray,
        score: float,
        prompts: Dict,
        save_path: str = None,
    ) -> np.ndarray:
        """可视化分割结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 原始图像 + 边界框 + 提示点
        axes[0].imshow(image)
        axes[0].set_title("原始图像 + 提示信息", fontsize=14)

        # 绘制边界框
        x1, y1, x2, y2 = pneumo_info.bbox
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[0].add_patch(rect)

        # 绘制提示点
        prompt_data = prompts["prompts"]
        if "positive_points" in prompt_data:
            pos_points = np.array(prompt_data["positive_points"])
            axes[0].scatter(
                pos_points[:, 0],
                pos_points[:, 1],
                c="green",
                s=50,
                marker="o",
                label="正提示点",
            )

        if "negative_points" in prompt_data:
            neg_points = np.array(prompt_data["negative_points"])
            axes[0].scatter(
                neg_points[:, 0],
                neg_points[:, 1],
                c="red",
                s=50,
                marker="x",
                label="负提示点",
            )

        axes[0].legend()
        axes[0].axis("off")

        # 2. 分割掩码
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"分割掩码 (得分: {score:.3f})", fontsize=14)
        axes[1].axis("off")

        # 3. 叠加结果
        overlay = image.copy()
        mask_colored = np.zeros_like(image)

        # 确保mask是布尔类型
        mask_bool = mask.astype(bool)
        mask_colored[mask_bool] = [255, 0, 0]  # 红色掩码

        # 半透明叠加
        alpha = 0.4
        overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)

        axes[2].imshow(overlay)
        axes[2].set_title("叠加结果", fontsize=14)
        axes[2].axis("off")

        # 添加文本信息
        info_text = f"""气胸信息:
位置: {pneumo_info.position}
形状: {pneumo_info.shape}  
大小: {pneumo_info.size}
面积: {pneumo_info.area} 像素
中心: {pneumo_info.center}
分割得分: {score:.3f}"""

        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"可视化结果已保存到: {save_path}")

        # 转换为numpy数组返回
        fig.canvas.draw()
        result_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result_image = result_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return result_image

    def create_overlay_image(
        self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """创建简单的叠加图像"""
        overlay = image.copy()
        mask_colored = np.zeros_like(image)

        # 确保mask是布尔类型
        mask_bool = mask.astype(bool)
        mask_colored[mask_bool] = [255, 0, 0]  # 红色掩码

        # 半透明叠加
        result = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
        return result

    def process_single_case(
        self,
        image_path: str,
        text_or_data: Union[str, Dict],
        output_dir: str = None,
        use_precision: bool = False,
        use_enhanced: bool = True,
    ) -> Dict:
        """处理单个气胸案例"""
        try:
            # 1. 加载图像
            print(f"正在处理: {image_path}")
            image = self.load_image(image_path)
            image_shape = image.shape[:2]

            # 2. 解析文本并生成提示
            print("解析文本描述...")
            prompt_result = self.parse_text_and_generate_prompts(
                text_or_data,
                image_shape,
                use_enhanced=use_enhanced,
                use_precision=use_precision,
                use_bbox_constrained_points=True,
                image=image,
            )
            pneumo_info = prompt_result["pneumo_info"]
            prompts = prompt_result["prompts"]

            print(
                f"检测到气胸: {pneumo_info.position}, {pneumo_info.shape}, {pneumo_info.size}"
            )

            # 3. 执行分割
            print("执行SAM2分割...")
            seg_result = self.segment_pneumothorax(image, prompt_result)
            mask = seg_result["mask"]
            score = seg_result["score"]

            print(f"分割完成，得分: {score:.3f}")

            # 4. 生成输出
            result = {
                "image_path": image_path,
                "pneumo_info": pneumo_info,
                "mask": mask,
                "score": score,
                "success": True,
            }

            # 5. 保存结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

                # 保存可视化结果
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
                self.visualize_results(
                    image, pneumo_info, mask, score, prompt_result, vis_path
                )

                # 保存叠加图像
                overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                overlay_image = self.create_overlay_image(image, mask)
                cv2.imwrite(
                    overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
                )

                # 保存掩码
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

                result.update(
                    {
                        "visualization_path": vis_path,
                        "overlay_path": overlay_path,
                        "mask_path": mask_path,
                    }
                )

            return result

        except Exception as e:
            print(f"处理失败: {e}")
            return {"image_path": image_path, "error": str(e), "success": False}


def load_pneumothorax_dataset(json_path: str) -> List[Dict]:
    """加载气胸数据集"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--sam2-repo-dir", default=str(DEFAULT_SAM2_DIR))
    parser.add_argument("--model-cfg", default="")
    parser.add_argument("--model-ckpt", default="")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--disable-anatomical-constraints", action="store_true")
    parser.add_argument("--use-precision", action="store_true")
    parser.add_argument("--disable-enhanced", action="store_true")
    args = parser.parse_args()

    print("初始化SAM2气胸分割器...")
    segmenter = SAM2PneumothoraxSegmenter(
        model_cfg=args.model_cfg or None,
        model_ckpt=args.model_ckpt or None,
        sam2_repo_dir=args.sam2_repo_dir or None,
        device=args.device,
        use_anatomical_constraints=not args.disable_anatomical_constraints,
    )

    data_json = args.dataset_json
    output_dir = args.output_dir
    images_dir = args.images_dir

    print(f"加载数据集: {data_json}")
    dataset = load_pneumothorax_dataset(data_json)

    test_cases = dataset[: max(1, args.max_cases)]

    results = []
    for i, case in enumerate(test_cases):
        print(f"\n=== 处理案例 {i+1}/{len(test_cases)} ===")

        use_new_format = "pneu_location" in case and "has_pneu" in case
        if use_new_format:
            # 新格式：直接使用结构化数据
            image_path = case["image"]
            data_input = case  # 传入完整的结构化数据
            print(f"新格式数据 - 图像: {image_path}")
            print(
                f"气胸位置: {case.get('pneu_location', {}).get('full_position', 'N/A')}"
            )
        else:
            # 旧格式：提取文本描述
            image_name = os.path.basename(case["image"]).replace(".png", ".jpg")
            image_path = os.path.join(images_dir, image_name)
            data_input = case["conversations"][1]["content"]  # 文本描述
            print(f"旧格式数据 - 图像: {image_path}")

        # 处理案例 (使用增强分割模式)
        result = segmenter.process_single_case(
            image_path,
            data_input,
            output_dir,
            use_precision=args.use_precision,
            use_enhanced=not args.disable_enhanced,
        )
        results.append(result)

        if result["success"]:
            print(f"✓ 成功处理: {os.path.basename(image_path)}")
            print(f"  分割得分: {result['score']:.3f}")
            if hasattr(result["pneumo_info"], "has_structured_data"):
                print(
                    f"  数据类型: {'结构化' if result['pneumo_info'].has_structured_data else '文本'}"
                )
        else:
            print(f"✗ 处理失败: {result['error']}")

    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    print(f"\n=== 处理完成 ===")
    print(f"成功: {success_count}/{len(results)}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
