# MolTrainer 用户手册

**版本：** 0.1.0  
**作者：** MolTrainer 开发团队  
**最后更新：** 2025年10月

## 目录

1. [简介](#简介)
2. [安装](#安装)
3. [快速开始](#快速开始)
4. [数据格式](#数据格式)
5. [核心模块](#核心模块)
   - [描述性统计](#描述性统计)
   - [数据清洗](#数据清洗)
   - [数据可视化](#数据可视化)
   - [数据分割](#数据分割)
   - [模型训练](#模型训练)
   - [模型预测](#模型预测)
6. [高级功能](#高级功能)
7. [配置文件](#配置文件)
8. [输出文件](#输出文件)
9. [故障排除](#故障排除)
10. [常见问题](#常见问题)

---

## 简介

MolTrainer 是一个全面的命令行工具，专为分子数据的机器学习而设计。它提供从数据清洗到模型训练和预测的完整工作流，特别支持基于SMILES的分子描述符。

### 主要特性

- **解耦的模块化架构**，易于扩展
- **SMILES特征化**，使用RDKit描述符
- **多种机器学习算法**：随机森林、SVM、XGBoost、LightGBM、逻辑/线性回归
- **自动超参数搜索**，支持网格搜索和随机搜索
- **学术级可视化**，遵循Nature/Science标准
- **全面的报告系统**，自动时间戳和存储
- **模型元数据管理**，实现可复现的预测

---

## 安装

### 前置要求

- Python 3.8 或更高版本
- conda（推荐用于安装RDKit）

### 步骤1：安装RDKit

RDKit是SMILES处理所必需的：

```bash
conda install -c conda-forge rdkit
```

或使用pip：

```bash
pip install rdkit-pypi
```

### 步骤2：安装MolTrainer

**可编辑安装（推荐用于开发）：**

```bash
git clone https://github.com/yourusername/moltrainer.git
cd moltrainer
pip install -e .
```

**标准安装：**

```bash
pip install .
```

### 步骤3：安装可选依赖

用于XGBoost和LightGBM支持：

```bash
pip install xgboost lightgbm
```

### 步骤4：验证安装

```bash
moltrainer -h
```

---

## 快速开始

```bash
# 1. 探索数据
moltrainer -i data.csv -desc_stats

# 2. 清洗数据
moltrainer -i data.csv -clean -validate_smiles -remove_duplicates -o clean.csv

# 3. 分割为训练/验证/测试集
moltrainer -i clean.csv -split -stratify activity

# 4. 训练模型
moltrainer -i clean_train.csv -train -target activity -smiles smiles -o results/

# 5. 进行预测
moltrainer -predict -load_model results/clean_train_model.pkl -i new_data.csv -o predictions.csv
```

---

## 数据格式

### CSV要求

- **文件格式**：CSV（逗号分隔值）
- **编码**：推荐UTF-8
- **表头**：第一行必须包含列名
- **无需索引列**（如果存在会被忽略）

### 支持的数据类型

#### 1. 基于SMILES的分类任务

```csv
compound_id,smiles,activity
COMP001,CCO,active
COMP002,CC(C)O,active
COMP003,c1ccccc1,inactive
```

#### 2. 基于SMILES的回归任务

```csv
compound_id,smiles,ic50
COMP001,CCO,10.5
COMP002,CC(C)O,15.2
COMP003,c1ccccc1,45.8
```

#### 3. 数值特征

```csv
compound_id,logp,mw,tpsa,activity
COMP001,0.23,46.07,20.23,active
COMP002,0.65,60.10,20.23,active
```

### 数据质量要求

- **SMILES**：应该有效且标准化
- **缺失值**：明确标记（空、NaN或NA）
- **目标列**：
  - 分类：分类标签（例如"active"、"inactive"）
  - 回归：数值
- **特征列**：仅数值

---

## 核心模块

### 描述性统计

生成数据集的全面统计摘要。

**使用方法：**

```bash
moltrainer -i data.csv -desc_stats
```

**输出：**

- 数据集形状和内存使用情况
- 每列的数据类型
- 缺失值分析
- 数值特征：均值、标准差、最小值、最大值、四分位数、偏度、峰度
- 分类特征：唯一值、高频值、频率
- 自动保存报告到 `reports/YYYYMMDD_HHMMSS_descriptive_stats.txt`

**示例：**

```bash
moltrainer -i compounds.csv -desc_stats -v
```

---

### 数据清洗

交互式或批量数据清洗，支持多种操作。

#### 交互模式

```bash
moltrainer -i data.csv -clean -o cleaned.csv
```

用户将被提示选择清洗操作。

#### 批处理模式

直接指定清洗操作：

```bash
moltrainer -i data.csv -clean \
  -remove_duplicates \
  -handle_missing \
  -missing_method drop \
  -validate_smiles \
  -smiles_column smiles \
  -filter_value "ic50 < 100" \
  -remove_outliers \
  -outlier_method iqr \
  -outlier_columns ic50 \
  -o cleaned.csv
```

#### 清洗操作

**1. 删除重复行**

```bash
-remove_duplicates                     # 删除重复行
-duplicate_subset "col1,col2"          # 仅考虑特定列
```

**2. 处理缺失值**

```bash
-handle_missing                        # 启用缺失值处理
-missing_method drop                   # drop（删除）, fill（填充）
-fill_method mean                      # mean（均值）, median（中位数）, mode（众数）
-fill_value 0                          # 或指定一个值
```

**3. 删除异常值**

```bash
-remove_outliers                       # 启用异常值删除
-outlier_method iqr                    # iqr 或 zscore
-outlier_threshold 1.5                 # 阈值（IQR为1.5，z-score为3）
-outlier_columns "col1,col2"           # 要检查的列
```

**4. 验证SMILES**

```bash
-validate_smiles                       # 删除无效SMILES
-smiles_column smiles                  # SMILES列名
```

**5. 按值过滤**

```bash
-filter_value "column > value"         # 比较运算符：>, <, >=, <=, ==, !=
```

示例：

```bash
-filter_value "ic50 < 100"
-filter_value "activity == active"
-filter_value "logp >= 0"
```

**6. 删除列**

```bash
-drop_columns                          # 启用列删除
-columns_to_drop "col1,col2"           # 要删除的列
```

#### 输出

- 清洗后的CSV文件
- 详细的清洗报告（控制台 + `reports/` 目录）
- 显示每步删除的行数

---

### 数据可视化

生成符合学术标准的出版级图表。

**使用方法：**

```bash
moltrainer -i data.csv -visualize -plot_type TYPE -o output.png
```

#### 图表类型

**1. 分布图**

```bash
moltrainer -i data.csv -visualize \
  -plot_type distribution \
  -columns "logp,mw,tpsa" \
  -o distribution.svg
```

**2. 相关性热图**

```bash
moltrainer -i data.csv -visualize \
  -plot_type correlation \
  -columns "logp,mw,tpsa" \
  -o correlation.png
```

**3. 箱线图**

```bash
moltrainer -i data.csv -visualize \
  -plot_type boxplot \
  -columns "logp,mw" \
  -title "LogP and MW Distribution" \
  -o boxplot.jpg
```

#### 选项

```bash
-plot_type TYPE          # distribution, correlation, boxplot, all
-columns "col1,col2"     # 要绘制的列（可选，默认为所有数值列）
-sample_size N           # 样本大小：绝对数量或百分比（例如"50%"）
-title "我的标题"        # 自定义图表标题
-xlabel "X轴标签"        # 自定义x轴标签
-ylabel "Y轴标签"        # 自定义y轴标签
```

#### 输出格式

- `.svg` - 矢量格式（推荐用于出版）
- `.png` - 光栅格式（DPI=600，高质量）
- `.jpg` / `.jpeg` - 压缩格式

#### 可视化标准

- 字体：Arial，8-10pt
- 风格：Nature/Science出版标准
- 调色板：色盲友好
- DPI：光栅格式为600
- 无顶部/右侧边框
- 清晰的图例和标签

---

### 数据分割

将数据集分割为训练集、验证集和测试集。

**基本用法：**

```bash
moltrainer -i data.csv -split
```

这将创建三个文件：
- `data_train.csv` (70%)
- `data_val.csv` (15%)
- `data_test.csv` (15%)

#### 自定义比例

```bash
moltrainer -i data.csv -split \
  -train_ratio 0.8 \
  -val_ratio 0.1 \
  -test_ratio 0.1
```

#### 双向分割（仅训练/测试）

```bash
moltrainer -i data.csv -split \
  -train_ratio 0.8 \
  -val_ratio 0.0 \
  -test_ratio 0.2
```

#### 分层分割

用于分类任务：

```bash
moltrainer -i data.csv -split \
  -stratify activity
```

这确保类别分布在各分割中保持平衡。

#### 选项

```bash
-train_ratio 0.7         # 训练集比例（默认：0.7）
-val_ratio 0.15          # 验证集比例（默认：0.15）
-test_ratio 0.15         # 测试集比例（默认：0.15）
-stratify COLUMN         # 分层抽样的列
-shuffle                 # 分割前打乱（默认：True）
```

**注意：** 比例之和必须为1.0

#### 输出

- 三个自动命名的CSV文件
- 分割报告，包含类别分布（如果分层）
- 报告保存到 `reports/` 目录

---

### 模型训练

使用自动化工作流训练机器学习模型。

#### 使用SMILES的基本训练

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -o results/
```

#### 使用数值特征的训练

```bash
moltrainer -i train.csv -train \
  -target ic50 \
  -features "logp,mw,tpsa" \
  -task regression \
  -o results/
```

#### 模型类型

```bash
-model rf                # 随机森林（默认）
-model svm               # 支持向量机
-model xgb               # XGBoost
-model lgb               # LightGBM
-model lr                # 逻辑/线性回归
```

#### 任务类型

```bash
-task auto               # 自动检测（默认）
-task classification     # 分类
-task regression         # 回归
```

#### 特征工程选项 ⭐ 新功能

MolTrainer支持从SMILES生成全面的分子特征，包括**200+种理化描述符**和**5种分子指纹**。

**可用特征类型：**

1. **仅描述符**（默认）
2. **仅指纹**
3. **组合**（描述符 + 指纹）

##### 1. 理化描述符

提供三种描述符集：

```bash
-feat_type descriptors -desc_set basic      # 10个基础描述符（快速）
-feat_type descriptors -desc_set extended   # ~30个扩展描述符（中等）
-feat_type descriptors -desc_set all        # 200+个全面描述符（全面）
```

**基础描述符（10个）：**
- 分子量、LogP（亲脂性）
- 氢键供体/受体数
- TPSA（拓扑极性表面积）
- 可旋转键数
- 环计数（芳香环、饱和环、脂肪环）

**扩展描述符（~30个）：**
- 基础描述符 + 分子折射率
- 拓扑指数（BertzCT, Chi, Kappa）
- 电子性质
- 结构特征（CSP3分数、杂原子数）
- 药效团特征（LabuteASA, PEOE_VSA, SMR_VSA）

**全部描述符（200+）：**
- 完整的RDKit 2D描述符集
- 全面的分子表征

**示例：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type descriptors \
  -desc_set extended \
  -o results/
```

##### 2. 分子指纹

支持五种指纹类型：

```bash
-feat_type fingerprints -fp_type morgan       # Morgan（圆形）指纹
-feat_type fingerprints -fp_type maccs        # MACCS键（167位，固定）
-feat_type fingerprints -fp_type rdk          # RDKit指纹
-feat_type fingerprints -fp_type atompair     # 原子对指纹
-feat_type fingerprints -fp_type topological  # 拓扑扭转指纹
```

**指纹选项：**

```bash
-fp_bits 2048           # 指纹位数（默认：2048）
-fp_radius 2            # Morgan指纹半径（默认：2）
```

**常用位数：**
- 小型：256-512位（更快，内存占用少）
- 中型：1024位（平衡）
- 大型：2048-4096位（信息更多）

**示例 - Morgan指纹：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -fp_bits 1024 \
  -fp_radius 3 \
  -o results/
```

**示例 - MACCS键：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type maccs \
  -o results/
```

##### 3. 组合特征

组合描述符和指纹以获取最大信息量：

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type combined \
  -desc_set basic \
  -fp_type morgan \
  -fp_bits 512 \
  -o results/
```

这将生成：**10个描述符 + 512个指纹位 = 522个特征**

**常用组合：**

```bash
# 基础描述符 + MACCS（快速，可解释）
-feat_type combined -desc_set basic -fp_type maccs

# 扩展描述符 + Morgan 1024（平衡）
-feat_type combined -desc_set extended -fp_type morgan -fp_bits 1024

# 全部描述符 + Morgan 2048（全面）
-feat_type combined -desc_set all -fp_type morgan -fp_bits 2048
```

##### 4. 自动指纹长度优化 ⭐

通过在不同位数下训练模型，自动找到最优指纹长度：

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -optimize_fp \
  -fp_start 16 \
  -fp_step 16 \
  -fp_max 2048 \
  -o results/
```

**优化参数：**

```bash
-optimize_fp            # 启用指纹长度优化
-fp_start 16            # 起始位数（默认：16）
-fp_step 16             # 步进大小（默认：16）
-fp_max 2048            # 最大位数（默认：2048）
```

**工作原理：**
1. 测试从16到2048位的指纹长度（步进=16）
2. 在每个位数下使用交叉验证训练模型
3. 报告性能最佳的最优长度
4. 使用最优指纹长度训练最终模型

**示例输出：**

```
Testing 16 bits... Score: 0.75 (+/- 0.05)
Testing 32 bits... Score: 0.82 (+/- 0.04)
Testing 48 bits... Score: 0.85 (+/- 0.03)
...
Testing 512 bits... Score: 0.91 (+/- 0.02)  ← 最佳
Testing 1024 bits... Score: 0.90 (+/- 0.03)
...

最优指纹长度：512位
最佳分数：0.9100
```

**优化技巧：**

- **快速搜索**：`-fp_start 64 -fp_step 64 -fp_max 1024`
- **精细搜索**：`-fp_start 256 -fp_step 32 -fp_max 768`
- **全面搜索**：`-fp_start 16 -fp_step 16 -fp_max 2048`（默认，较慢）

##### 特征工程最佳实践

**快速实验：**
```bash
-feat_type descriptors -desc_set basic      # 最快
```

**发表级模型：**
```bash
-feat_type combined -desc_set extended -fp_type morgan -fp_bits 1024
```

**追求最高性能：**
```bash
-feat_type combined -desc_set all -fp_type morgan -optimize_fp
```

**内存考虑：**
- 大型指纹（2048+位）在大数据集上可能占用大量内存
- 如果内存有限，考虑先对数据采样
- 对大数据集使用 `-desc_set basic` 而不是 `all`

##### 特征类型选择指南

| 任务 | 推荐特征 | 理由 |
|------|---------|------|
| ADMET预测 | 组合（扩展 + Morgan） | 需要结构和性质信息 |
| 活性分类 | 指纹（Morgan/MACCS） | 结构-活性关系 |
| 性质回归 | 描述符（扩展/全部） | 直接性质计算 |
| 相似性搜索 | MACCS或Morgan | 快速比较 |
| 可解释模型 | 仅描述符 | 有名称，可解释 |

#### 超参数搜索

**随机搜索（推荐）：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -search random \
  -search_depth deep \
  -search_iter 20 \
  -search_cv 5 \
  -o results/
```

**网格搜索：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -search grid \
  -search_depth shallow \
  -search_cv 3 \
  -o results/
```

**搜索选项：**

```bash
-search METHOD           # none, grid, random（默认：none）
-search_depth LEVEL      # shallow（浅层）, deep（深层）（默认：shallow）
-search_iter N           # 随机搜索的迭代次数（默认：10）
-search_cv N             # 搜索的交叉验证折数（默认：3）
-search_timeout SECONDS  # 最大搜索时间（秒）
```

**搜索深度：**

- `shallow`（浅层）：更快，较少参数（适合快速实验）
- `deep`（深层）：彻底，更多参数（更好性能，更长时间）

#### 自动数据分割

如果未提供验证/测试集：

```bash
# 三向分割（train/val/test）
-auto_split 3way         # 默认

# 二向分割（train/test）
-auto_split 2way

# 不自动分割（必须提供 -val 和 -test）
-auto_split none
```

自定义分割比例：

```bash
-train_split 0.6         # 使用60%用于训练（三向：60/20/20，二向：60/40）
```

#### 交叉验证

```bash
-cv 5                    # 5折交叉验证（默认）
-cv 10                   # 10折交叉验证
-no_cv                   # 禁用交叉验证
```

#### 使用单独的验证/测试集

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -val validation.csv \
  -test test.csv \
  -auto_split none \
  -o results/
```

#### 训练输出

所有文件都保存到指定的输出文件夹：

1. **模型文件**：`{basename}_model.pkl`
   - 包含模型、元数据和标签编码器
   - 自包含，可用于预测

2. **训练日志**：`{basename}_training_log.txt`
   - 详细的训练信息
   - 超参数、时间、结果

3. **图表**（SVG + PNG）：
   - 特征重要性（基于树的模型）
   - 混淆矩阵（分类）
   - 预测散点图（回归）

4. **图表数据**（CSV）：
   - 生成每个图表所用的数据
   - 用于自定义可视化

5. **训练报告**：保存到 `reports/` 目录

#### 无效SMILES处理

无效的SMILES会自动跳过并给出警告：

```
Warning: 发现并跳过了5个无效SMILES
建议：在训练前使用 -clean -validate_smiles
```

---

### 模型预测

使用训练好的模型进行预测。

#### 查看模型信息

```bash
moltrainer -model_info results/model.pkl
```

**输出：**
- 模型类型和任务
- 训练日期
- 特征信息（SMILES或数值列）
- 类别标签（分类）
- 超参数
- 交叉验证分数
- 使用说明

#### 进行预测

```bash
moltrainer -predict \
  -load_model results/model.pkl \
  -i new_data.csv \
  -o predictions.csv
```

#### 要求

输入CSV必须包含：
- **对于基于SMILES的模型**：相同的SMILES列
- **对于数值特征模型**：相同的特征列

#### 预测输出

**分类：**
- `predicted_{target}`：预测的类别标签
- `probability_{class1}`, `probability_{class2}`, ...：类别概率

**回归：**
- `predicted_{target}`：预测的数值

**无效数据：**
- 具有无效SMILES或缺失特征的行在预测列中标记为NaN
- 原始数据得以保留

#### 示例输出

```csv
compound_id,smiles,activity,predicted_activity,probability_active,probability_inactive
C001,CCO,active,active,0.94,0.06
C002,INVALID,active,NaN,NaN,NaN
C003,c1ccccc1,inactive,inactive,0.01,0.99
```

#### 预测报告

- 控制台输出和保存的报告
- 成功预测的数量
- 无效数据的警告
- 模型信息摘要

---

## 高级功能

### 配置文件

对于复杂的训练设置，使用YAML或JSON配置文件。

#### 创建示例配置

```bash
moltrainer -create_config my_config.yaml
```

#### 配置文件结构

```yaml
# 输入/输出
input_file: data/train.csv
output_folder: results/experiment_001
target_column: activity

# 特征规范
smiles_column: smiles
# 或
# feature_columns:
#   - logp
#   - molecular_weight

# 可选：验证/测试数据
validation_file: data/val.csv
test_file: data/test.csv

# 模型设置
model_type: rf
task: auto

# 超参数
n_estimators: 100
max_depth: null
random_state: 42

# 交叉验证
cv_folds: 5
no_cv: false

# 自动数据分割
auto_split_mode: 3way
train_split_ratio: null

# 超参数搜索
search_method: random
search_depth: deep
search_iterations: 20
search_cv_folds: 5

# 输出
verbose: true
```

#### 使用配置运行训练

```bash
moltrainer -config my_config.yaml
```

#### 用CLI参数覆盖配置

CLI参数优先级更高：

```bash
moltrainer -config my_config.yaml -n_estimators 200 -search_iter 30
```

---

## 输出文件

### 自动报告目录

所有报告都保存到 `reports/`，带时间戳：

```
reports/
├── 20251024_143000_descriptive_stats.txt
├── 20251024_143100_data_cleaning.txt
├── 20251024_143200_visualization.txt
├── 20251024_143300_data_split.txt
├── 20251024_143400_training.txt
└── 20251024_143500_prediction.txt
```

### 训练输出结构

```
results/
├── model_name_model.pkl                      # 模型 + 元数据
├── model_name_training_log.txt               # 详细日志
├── model_name_feature_importance.png         # 图表（PNG）
├── model_name_feature_importance.svg         # 图表（SVG）
├── model_name_feature_importance_data.csv    # 图表数据
├── model_name_confusion_matrix.png           # 分类
├── model_name_confusion_matrix.svg
├── model_name_confusion_matrix_data.csv
├── model_name_predictions.png                # 回归
├── model_name_predictions.svg
└── model_name_predictions_data.csv
```

---

## 故障排除

### 常见问题

#### 1. RDKit导入错误

```
ImportError: SMILES特征化需要RDKit
```

**解决方案：**

```bash
conda install -c conda-forge rdkit
```

#### 2. 无效SMILES

```
Warning: 发现10个无效SMILES
```

**解决方案：** 先清洗数据：

```bash
moltrainer -i data.csv -clean -validate_smiles -smiles_column smiles -o clean.csv
```

#### 3. 模型文件不兼容

```
Error: 未找到模型元数据。此模型可能来自旧版本。
```

**解决方案：** 使用当前版本重新训练模型。

#### 4. 特征列不匹配

```
ValueError: 在输入数据中未找到特征列：logp, mw
```

**解决方案：** 确保预测数据具有训练时使用的相同列。检查模型信息：

```bash
moltrainer -model_info model.pkl
```

#### 5. 大数据集内存错误

**解决方案：**
- 对可视化使用采样：`-sample_size 1000` 或 `-sample_size 10%`
- 减少搜索迭代：`-search_iter 5`
- 使用浅层搜索：`-search_depth shallow`

#### 6. 未找到XGBoost/LightGBM

```
ImportError: 未安装XGBoost
```

**解决方案：**

```bash
pip install xgboost lightgbm
```

---

## 常见问题

### 一般问题

**问：支持哪些文件格式？**  
答：目前仅支持CSV格式。确保使用UTF-8编码。

**问：我可以使用自己的描述符而不是SMILES吗？**  
答：可以，使用 `-features "col1,col2,col3"` 指定数值特征列。

**问：如何更新MolTrainer？**  
答：对于可编辑安装：在MolTrainer目录中执行 `git pull`。

### 数据问题

**问：训练需要多少样本？**  
答：最少50-100个样本，但建议500+以获得稳健模型。对于小数据集使用交叉验证。

**问：我应该标准化特征吗？**  
答：对于SVM和逻辑/线性回归，应该（自动）。对于基于树的模型（RF、XGBoost、LightGBM），不必要。

**问：我可以在不平衡数据上训练吗？**  
答：可以，使用分层分割（`-stratify`）并考虑调整类别权重（未来功能）。

### 训练问题

**问：我应该使用哪个模型？**  
答：
- 从随机森林（`rf`）开始 - 良好的默认选择
- 尝试XGBoost（`xgb`）或LightGBM（`lgb`）以获得更好性能
- 对小数据集使用SVM（`svm`）
- 使用逻辑/线性回归（`lr`）作为基线

**问：我应该使用网格搜索还是随机搜索？**  
答：大多数情况下推荐随机搜索（更快，效果好）。对窄参数范围进行最终微调时使用网格搜索。

**问：超参数搜索需要多长时间？**  
答：取决于：
- 搜索方法：随机 < 网格
- 搜索深度：浅层 < 深层
- 数据集大小
- 模型类型

使用 `-search_timeout SECONDS` 限制搜索时间。

**问：浅层搜索和深层搜索有什么区别？**  
答：
- **浅层**：3-5个参数，每个4-6个值，更快
- **深层**：6-10个参数，每个5-12个值，彻底

### 预测问题

**问：我可以在来自不同来源的数据上使用模型吗？**  
答：可以，只要：
1. 存在相同的特征/SMILES列
2. 特征分布相似（否则模型可能表现不佳）

**问：如何解释预测概率？**  
答：对于分类，概率表示模型置信度：
- >0.8：高置信度
- 0.5-0.8：中等置信度
- <0.5：低置信度

**问：如果我得到NaN预测怎么办？**  
答：NaN表示无效的输入数据（无效SMILES或缺失特征）。检查输入数据质量。

### 输出问题

**问：报告保存在哪里？**  
答：所有报告自动保存到带时间戳的 `reports/` 目录。

**问：我可以更改输出目录吗？**  
答：对于训练：使用 `-o <folder>`。对于报告：目前固定为 `reports/`（自定义功能即将推出）。

**问：我应该使用什么格式的图表？**  
答：
- 出版物：SVG（矢量，可缩放）
- 演示文稿：PNG（高DPI，广泛兼容）
- 网页：JPG（文件更小）

---

## 引用

如果您在研究中使用MolTrainer，请引用：

```bibtex
@software{moltrainer2025,
  title = {MolTrainer: 分子数据的机器学习工具},
  author = {MolTrainer开发团队},
  year = {2025},
  url = {https://github.com/yourusername/moltrainer}
}
```

---

## 支持

- **文档**：本文件和 `help.md`
- **问题反馈**：https://github.com/yourusername/moltrainer/issues
- **讨论**：https://github.com/yourusername/moltrainer/discussions

---

## 许可证

MIT许可证 - 详见LICENSE文件

---

**最后更新：** 2025年10月  
**版本：** 0.1.0

