# SAM2 Pneumothorax Segmentation

基于 SAM2 的气胸分割项目，支持胸片（X-ray）结构化标注数据驱动的自动分割与结果导出。

## 1. 环境准备

在项目根目录执行：

```bash
pip install -r requirements.txt
pip install -e .
```

如果你使用本地 `sam2-main` 仓库，建议设置：

```bash
set SAM2_REPO_DIR=..\sam2-main
```

Linux/macOS:

```bash
export SAM2_REPO_DIR=../sam2-main
```

如果你有本地权重，也可以设置：

```bash
set SAM2_CHECKPOINT=D:\path\to\sam2.1_hiera_large.pt
```

Linux/macOS:

```bash
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
```

## 2. 快速测试命令

### 2.1 代码可运行性检查（不依赖数据集）

```bash
python -m py_compile script\sam2_pneumothorax_segmenter.py script\test_pneumothorax_segmentation.py script\text_to_sam2_prompt.py script\anatomical_constraints.py script\__init__.py
```

### 2.2 查看主脚本参数

```bash
python script\sam2_pneumothorax_segmenter.py --help
```

或安装后命令：

```bash
pneumo-seg --help
```

### 2.3 运行分割（主流程）

```bash
pneumo-seg ^
  --dataset-json D:\data\all_data_result.json ^
  --output-dir .\outputs ^
  --sam2-repo-dir ..\sam2-main ^
  --model-ckpt D:\models\sam2.1_hiera_large.pt ^
  --max-cases 3 ^
  --use-precision
```

Linux/macOS:

```bash
pneumo-seg \
  --dataset-json /data/all_data_result.json \
  --output-dir ./outputs \
  --sam2-repo-dir ../sam2-main \
  --model-ckpt /models/sam2.1_hiera_large.pt \
  --max-cases 3 \
  --use-precision
```

### 2.4 运行综合测试脚本

```bash
pneumo-test ^
  --dataset-path D:\data\all_data_result.json ^
  --output-dir .\test_results
```

Linux/macOS:

```bash
pneumo-test \
  --dataset-path /data/all_data_result.json \
  --output-dir ./test_results
```

## 3. 输入数据要求

- `--dataset-json` / `--dataset-path` 指向 JSON 列表文件。
- 新格式样本建议包含：
  - `has_pneu`
  - `image`
  - `pneu_location.bounding_box`
- 输出包含：
  - 可视化图：`*_visualization.png`
  - 叠加图：`*_overlay.png`
  - 掩码图：`*_mask.png`

## 4. 常见报错排查

- `ModuleNotFoundError: sam2...`
  - 检查 `sam2-main` 路径是否正确，并通过 `--sam2-repo-dir` 或 `SAM2_REPO_DIR` 指定。
- `无法加载SAM2模型`
  - 检查 `--model-ckpt` 文件是否存在；若不存在会尝试 Hugging Face 预训练模型。
- `图像文件不存在`
  - 检查 JSON 中 `image` 路径是否为可访问绝对路径，或是否与当前机器路径一致。

