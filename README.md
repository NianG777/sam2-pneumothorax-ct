# SAM2 Pneumothorax CT Segmentation | 气胸CT智能分割系统

> **医疗AI测试实战项目** | 复旦大学附属肿瘤医院合作 | 获计算机软件著作权

基于 SAM2 大模型的气胸CT影像自动分割系统，解决人工分割效率低（30分钟→60秒）、误差大的临床痛点，支持结构化标注数据驱动的全流程质量保障。

---

## 🏥 项目背景与业务价值

| 指标 | 传统人工 | 本系统 | 提升 |
|:---|:---|:---|:---|
| 单例分割时间 | 30分钟 | 60秒 | **30倍** |
| 分割一致性 | 医师间差异大 | Dice 0.91 | 标准化 |
| 临床落地 | 无法批量处理 | 已交付复旦肿瘤医院试用 | 可用 |

**合作单位**：复旦大学附属肿瘤医院  
**知识产权**：计算机软件著作权（登记号：2026SR0505326）

---

## 🧪 测试体系架构（核心亮点）

独立从0搭建自动化测试框架，覆盖功能/接口/性能/模型精度四维质量保障。

## 🚀 快速开始
### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/NianG777/sam2-pneumothorax-ct.git
cd sam2-pneumothorax-ct

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 可选：设置本地SAM2仓库路径（如有）
export SAM2_REPO_DIR=../sam2-main        # Linux/macOS
set SAM2_REPO_DIR=..\sam2-main            # Windows

# 可选：设置本地模型权重路径
export SAM2_CHECKPOINT=/path/to/sam2.1_hiera_large.pt
```

### 2. 快速命令

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

---

## 📁 项目结构

```plain
sam2-pneumothorax-ct/
│
├── 📂 script/                  # 核心源码
│   ├── sam2_pneumothorax_segmenter.py    # 主分割引擎
│   ├── test_pneumothorax_segmentation.py # 功能测试脚本
│   ├── text_to_sam2_prompt.py            # 文本提示生成
│   └── anatomical_constraints.py         # 解剖约束模块
│
├── README.md                   # 本文件
├── requirements.txt            # 依赖清单
```