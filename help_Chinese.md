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
git clone https://github.com/Marissapy/MolTrainer.git
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

## 参数指南与常见场景

### 理解参数

#### 必需与可选参数

**始终必需：**
- `-i <input.csv>` - 所有操作的输入文件

**操作特定必需：**
- **训练**: `-target <列名>` + (`-smiles <列名>` 或 `-features <列名>`)
- **清洗**: `-o <output.csv>` (输出文件)
- **可视化**: `-o <plot.png>` (输出文件)
- **预测**: `-load_model <model.pkl>` + `-o <output.csv>`

#### 参数规则

**规则1：SMILES vs Features（互斥）**

训练时必须选择其中一个：

```bash
# 选项A：使用SMILES进行特征化
-smiles <列名>

# 选项B：使用现有的数值特征
-features "<列1>,<列2>,<列3>"

# ❌ 错误：不能同时使用
-smiles smiles -features "mw,logp"  # 错误！
```

**规则2：任务类型检测**

```bash
# 自动检测（默认）- 推荐
-task auto

# 如果自动检测失败，显式指定
-task classification  # 分类任务
-task regression      # 回归任务
```

**规则3：输出要求**

| 模块 | 输出参数 | 需要提供什么 |
|------|---------|-------------|
| `-desc_stats` | 可选 `-o` | 文件路径（文本文件）|
| `-clean` | **必需** `-o` | 输出CSV文件 |
| `-visualize` | **必需** `-o` | 图表文件(.png/.svg/.jpg) |
| `-split` | 自动生成 | 添加`_train.csv`, `_val.csv`, `_test.csv` |
| `-train` | **必需** `-o` | 输出文件夹 |
| `-predict` | **必需** `-o` | 输出CSV文件 |
| `-sample` | **必需** `-o` | 输出CSV文件 |

### 常见场景完整示例

#### 场景1：基于SMILES的分类

**目标：** 从SMILES字符串预测活性

**数据示例：**
```csv
smiles,activity
CCO,active
c1ccccc1,inactive
CC(C)O,active
```

**命令：**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -o results/
```

**发生了什么：**
1. 加载`data.csv`
2. 从`smiles`列提取分子描述符
3. 检测分类任务（分类标签）
4. 训练随机森林模型
5. 保存模型和报告到`results/`文件夹

**可选增强：**
```bash
# 添加超参数搜索
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -search random \
  -search_depth deep \
  -o results/

# 使用不同模型
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -model xgb \
  -o results/

# 使用指纹代替描述符
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -fp_bits 1024 \
  -o results/
```

#### 场景2：基于SMILES的回归与自定义特征

**目标：** 使用Morgan指纹预测IC50值

**数据示例：**
```csv
compound_id,smiles,ic50
COMP001,CCO,10.5
COMP002,CC(C)O,15.2
COMP003,c1ccccc1,45.8
```

**命令：**
```bash
moltrainer -i data.csv -train \
  -target ic50 \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -fp_bits 2048 \
  -task regression \
  -o ic50_model/
```

**发生了什么：**
1. 从SMILES提取Morgan指纹（2048位）
2. 训练回归模型预测IC50
3. 保存模型到`ic50_model/`文件夹

#### 场景3：使用预先计算的数值特征

**目标：** 使用现有分子描述符训练模型

**数据示例：**
```csv
compound_id,mw,logp,tpsa,activity
COMP001,46.07,0.23,20.23,active
COMP002,60.10,0.65,20.23,active
COMP003,78.11,2.05,0.00,inactive
```

**命令：**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -features "mw,logp,tpsa" \
  -o results/
```

**重要提示：**
- ✅ 对数值列使用`-features`
- ❌ 不要使用`-features smiles` - SMILES是文本！
- ✅ 多列：使用逗号分隔的字符串
- ❌ 不要使用空格：`"mw, logp"`可能导致问题

#### 场景4：训练前的数据清洗

**目标：** 清洗数据，然后训练模型

**步骤1：清洗数据**
```bash
moltrainer -i raw_data.csv -clean \
  -remove_duplicates \
  -validate_smiles \
  -handle_missing drop \
  -o cleaned_data.csv
```

**步骤2：训练模型**
```bash
moltrainer -i cleaned_data.csv -train \
  -target activity \
  -smiles smiles \
  -o model_results/
```

#### 场景5：完整工作流程（含数据分割）

**目标：** 清洗 → 分割 → 训练 → 预测

**步骤1：清洗**
```bash
moltrainer -i raw.csv -clean \
  -validate_smiles \
  -remove_duplicates \
  -o cleaned.csv
```

**步骤2：分割**
```bash
moltrainer -i cleaned.csv -split \
  -train_ratio 0.7 \
  -val_ratio 0.15 \
  -test_ratio 0.15
```
生成：`cleaned_train.csv`, `cleaned_val.csv`, `cleaned_test.csv`

**步骤3：训练**
```bash
moltrainer -i cleaned_train.csv -train \
  -target activity \
  -smiles smiles \
  -val cleaned_val.csv \
  -test cleaned_test.csv \
  -o final_model/
```

**步骤4：预测**
```bash
moltrainer -predict \
  -load_model final_model/model.pkl \
  -i new_compounds.csv \
  -o predictions.csv
```

#### 场景6：超参数优化

**目标：** 找到最佳模型超参数

**快速搜索（快）：**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -search random \
  -search_iter 20 \
  -o quick_search/
```

**深度搜索（彻底）：**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -model xgb \
  -search random \
  -search_depth deep \
  -search_iter 100 \
  -o deep_search/
```

#### 场景7：特征工程优化

**目标：** 找到最优指纹长度

**步骤1：优化指纹**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -optimize_fp \
  -fp_start 64 \
  -fp_step 64 \
  -fp_max 1024 \
  -o fp_optimization/
```

**结果：** 报告最佳指纹长度（例如，512位）

**步骤2：使用优化后的长度训练**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -feat_spec "desc:extended+fp:morgan:512" \
  -o final_model/
```

#### 场景8：高级特征组合

**目标：** 组合多个描述符集和指纹

**命令：**
```bash
moltrainer -i data.csv -train \
  -target activity \
  -smiles smiles \
  -feat_spec "desc:basic+desc:extended+fp:morgan:1024+fp:maccs" \
  -model xgb \
  -search random \
  -o advanced_model/
```

**特征顺序：** [基础描述符, 扩展描述符, Morgan 1024, MACCS]

#### 场景9：比较多个模型

**目标：** 用不同算法训练相同数据

```bash
# 随机森林
moltrainer -i data.csv -train -target activity -smiles smiles \
  -model rf -o models/rf/

# XGBoost
moltrainer -i data.csv -train -target activity -smiles smiles \
  -model xgb -o models/xgb/

# LightGBM
moltrainer -i data.csv -train -target activity -smiles smiles \
  -model lgb -o models/lgb/

# SVM
moltrainer -i data.csv -train -target activity -smiles smiles \
  -model svm -o models/svm/
```

#### 场景10：使用配置文件

**目标：** 用配置文件管理复杂实验

**创建配置：**
```bash
moltrainer -create_config my_experiment.yaml
```

**编辑`my_experiment.yaml`：**
```yaml
input_file: "data.csv"
output_folder: "experiment_results/"
target_column: "activity"
smiles_column: "smiles"
model_type: "xgb"
task: "classification"
search_method: "random"
search_depth: "deep"
feature_type: "combined"
descriptor_set: "extended"
fingerprint_type: "morgan"
fingerprint_bits: 1024
```

**运行实验：**
```bash
moltrainer -config my_experiment.yaml
```

### 常见错误与解决方案

#### 错误1："Target column must be specified"

**原因：** 缺少`-target`参数

**解决方案：**
```bash
# ❌ 错误
moltrainer -i data.csv -train -smiles smiles -o results/

# ✅ 正确
moltrainer -i data.csv -train -target activity -smiles smiles -o results/
```

#### 错误2："Must specify either -smiles or -features"

**原因：** 忘记指定特征来源

**解决方案：**
```bash
# ❌ 错误
moltrainer -i data.csv -train -target activity -o results/

# ✅ 正确（选项A：SMILES）
moltrainer -i data.csv -train -target activity -smiles smiles -o results/

# ✅ 正确（选项B：数值特征）
moltrainer -i data.csv -train -target activity -features "mw,logp" -o results/
```

#### 错误3："could not convert string to float: 'CCO'"

**原因：** 使用了`-features smiles`而不是`-smiles smiles`

**解决方案：**
```bash
# ❌ 错误：将SMILES当作数值特征
moltrainer -i data.csv -train -target activity -features smiles -o results/

# ✅ 正确：特征化SMILES
moltrainer -i data.csv -train -target activity -smiles smiles -o results/
```

#### 错误4："Output file is required"

**原因：** 缺少需要输出的操作的`-o`

**解决方案：**
```bash
# ❌ 错误
moltrainer -i data.csv -clean -validate_smiles

# ✅ 正确
moltrainer -i data.csv -clean -validate_smiles -o cleaned.csv
```

#### 错误5："SMILES column not found"

**原因：** 列名不匹配

**解决方案：**
```bash
# 首先检查CSV表头
head -1 data.csv
# 输出：compound_id,SMILES,activity

# 使用准确的列名（区分大小写）
moltrainer -i data.csv -train -target activity -smiles SMILES -o results/
```

### 参数快速参考

#### 输入/输出
```bash
-i, -input FILE           # 输入CSV文件（始终必需）
-o, -output FILE/FOLDER   # 输出文件或文件夹（部分操作必需）
-v, --verbose             # 显示详细进度
```

#### 训练核心
```bash
-train                    # 启用训练模式
-target COLUMN            # 目标列名（必需）
-smiles COLUMN            # SMILES列用于特征化
-features "col1,col2"     # 数值特征列（-smiles的替代）
-task TYPE                # auto/classification/regression（默认：auto）
```

#### 模型选择
```bash
-model TYPE               # rf/svm/xgb/lgb/lr（默认：rf）
```

#### 特征工程
```bash
-feat_type TYPE           # descriptors/fingerprints/combined（默认：descriptors）
-desc_set SET             # basic/extended/all（默认：basic）
-fp_type TYPE             # morgan/maccs/rdk/atompair/topological（默认：morgan）
-fp_bits N                # 指纹位数（默认：2048）
-fp_radius N              # Morgan半径（默认：2）
-feat_spec "..."          # 自定义特征组合
```

#### 优化
```bash
-search METHOD            # none/grid/random（默认：none）
-search_depth LEVEL       # shallow/deep（默认：shallow）
-optimize_fp              # 优化指纹长度
```

#### 数据分割
```bash
-val FILE                 # 验证集文件
-test FILE                # 测试集文件
-auto_split MODE          # 3way/2way/none（默认：3way如果没有val/test）
```

#### 预测
```bash
-predict                  # 启用预测模式
-load_model FILE          # 训练好的模型文件（.pkl）
-model_info FILE          # 显示模型信息
```

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

##### 3. 自动指纹长度优化 ⭐

**重要提示：** 为获得最佳结果，应先优化指纹长度，再与描述符组合。

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

##### 4. 组合特征

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

**特征拼接顺序：** `[描述符, 指纹]`

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

##### 5. 自定义特征组合（高级）⭐

使用 `-feat_spec` 组合多个描述符集和指纹，并自定义顺序：

**格式：** `"desc:<集合>+fp:<类型>:<位数>:<半径>+..."`

**示例 - 两个描述符集 + 两个指纹：**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_spec "desc:basic+desc:extended+fp:morgan:1024+fp:maccs" \
  -o results/
```

按顺序生成特征：**[基础描述符, 扩展描述符, Morgan 1024, MACCS]**

**更多示例：**

```bash
# 三个不同参数的指纹
-feat_spec "fp:morgan:512:2+fp:morgan:1024:3+fp:maccs"

# 全部描述符 + 多个指纹
-feat_spec "desc:all+fp:morgan:2048+fp:rdk:1024+fp:maccs"

# 指纹夹着描述符
-feat_spec "fp:maccs+desc:extended+fp:morgan:1024"
```

**推荐工作流程：**

1. 首先，单独优化指纹长度
2. 然后，将优化后的指纹与描述符组合
3. 使用 `-feat_spec` 进行精细控制

```bash
# 步骤1：找到最优Morgan指纹长度
moltrainer -i train.csv -train -target activity -smiles smiles \
  -feat_type fingerprints -fp_type morgan -optimize_fp -o step1/

# 结果：最佳指纹长度 = 512位

# 步骤2：使用优化后的长度与描述符组合
moltrainer -i train.csv -train -target activity -smiles smiles \
  -feat_spec "desc:extended+fp:morgan:512+fp:maccs" -o step2/
```

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
  url = {https://github.com/Marissapy/MolTrainer}
}
```

---

## 支持

- **文档**：本文件和 `help.md`
- **问题反馈**：https://github.com/Marissapy/MolTrainer/issues
- **讨论**：https://github.com/Marissapy/MolTrainer/discussions

---

## 许可证

MIT许可证 - 详见LICENSE文件

---

**最后更新：** 2025年10月  
**版本：** 0.1.0

