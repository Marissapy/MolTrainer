# MolTrainer User Manual

**Version:** 0.1.0  
**Author:** MolTrainer Development Team  
**Last Updated:** October 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Format](#data-format)
5. [Core Modules](#core-modules)
   - [Descriptive Statistics](#descriptive-statistics)
   - [Data Cleaning](#data-cleaning)
   - [Data Visualization](#data-visualization)
   - [Data Splitting](#data-splitting)
   - [Model Training](#model-training)
   - [Model Prediction](#model-prediction)
6. [Advanced Features](#advanced-features)
7. [Configuration Files](#configuration-files)
8. [Output Files](#output-files)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## Introduction

MolTrainer is a comprehensive command-line tool designed for machine learning on molecular data. It provides an integrated workflow from data cleaning to model training and prediction, with special support for SMILES-based molecular descriptors.

### Key Features

- **Decoupled modular architecture** for easy extension
- **SMILES featurization** using RDKit descriptors
- **Multiple ML algorithms**: Random Forest, SVM, XGBoost, LightGBM, Logistic/Linear Regression
- **Automated hyperparameter search** with Grid Search and Random Search
- **Academic-quality visualizations** following Nature/Science standards
- **Comprehensive reporting** with automatic timestamp and storage
- **Model metadata management** for reproducible predictions

---

## Installation

### Prerequisites

- Python 3.8 or higher
- conda (recommended for RDKit installation)

### Step 1: Install RDKit

RDKit is required for SMILES processing:

```bash
conda install -c conda-forge rdkit
```

Alternatively, using pip:

```bash
pip install rdkit-pypi
```

### Step 2: Install MolTrainer

**Editable Installation (Recommended for Development):**

```bash
git clone https://github.com/yourusername/moltrainer.git
cd moltrainer
pip install -e .
```

**Standard Installation:**

```bash
pip install .
```

### Step 3: Install Optional Dependencies

For XGBoost and LightGBM support:

```bash
pip install xgboost lightgbm
```

### Step 4: Verify Installation

```bash
moltrainer -h
```

---

## Quick Start

```bash
# 1. Explore your data
moltrainer -i data.csv -desc_stats

# 2. Clean your data
moltrainer -i data.csv -clean -validate_smiles -remove_duplicates -o clean.csv

# 3. Split into train/val/test sets
moltrainer -i clean.csv -split -stratify activity

# 4. Train a model
moltrainer -i clean_train.csv -train -target activity -smiles smiles -o results/

# 5. Make predictions
moltrainer -predict -load_model results/clean_train_model.pkl -i new_data.csv -o predictions.csv
```

---

## Data Format

### CSV Requirements

- **File Format**: CSV (comma-separated values)
- **Encoding**: UTF-8 recommended
- **Headers**: First row must contain column names
- **No index column** required (will be ignored if present)

### Supported Data Types

#### 1. SMILES-Based Classification

```csv
compound_id,smiles,activity
COMP001,CCO,active
COMP002,CC(C)O,active
COMP003,c1ccccc1,inactive
```

#### 2. SMILES-Based Regression

```csv
compound_id,smiles,ic50
COMP001,CCO,10.5
COMP002,CC(C)O,15.2
COMP003,c1ccccc1,45.8
```

#### 3. Numeric Features

```csv
compound_id,logp,mw,tpsa,activity
COMP001,0.23,46.07,20.23,active
COMP002,0.65,60.10,20.23,active
```

### Data Quality Requirements

- **SMILES**: Should be valid and standardized
- **Missing values**: Clearly marked (empty, NaN, or NA)
- **Target column**: 
  - Classification: categorical labels (e.g., "active", "inactive")
  - Regression: numeric values
- **Feature columns**: Numeric values only

---

## Core Modules

### Descriptive Statistics

Generate comprehensive statistical summaries of your dataset.

**Usage:**

```bash
moltrainer -i data.csv -desc_stats
```

**Output:**

- Dataset shape and memory usage
- Data types for each column
- Missing value analysis
- Numeric features: mean, std, min, max, quartiles, skewness, kurtosis
- Categorical features: unique values, top values, frequencies
- Automatic report saved to `reports/YYYYMMDD_HHMMSS_descriptive_stats.txt`

**Example:**

```bash
moltrainer -i compounds.csv -desc_stats -v
```

---

### Data Cleaning

Interactive or batch data cleaning with multiple operations.

#### Interactive Mode

```bash
moltrainer -i data.csv -clean -o cleaned.csv
```

User will be prompted to select cleaning operations.

#### Batch Mode

Specify cleaning operations directly:

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

#### Cleaning Operations

**1. Remove Duplicates**

```bash
-remove_duplicates                     # Remove duplicate rows
-duplicate_subset "col1,col2"          # Only consider specific columns
```

**2. Handle Missing Values**

```bash
-handle_missing                        # Enable missing value handling
-missing_method drop                   # drop, fill
-fill_method mean                      # mean, median, mode (for fill)
-fill_value 0                          # Or specify a value
```

**3. Remove Outliers**

```bash
-remove_outliers                       # Enable outlier removal
-outlier_method iqr                    # iqr or zscore
-outlier_threshold 1.5                 # Threshold (1.5 for IQR, 3 for z-score)
-outlier_columns "col1,col2"           # Columns to check
```

**4. Validate SMILES**

```bash
-validate_smiles                       # Remove invalid SMILES
-smiles_column smiles                  # SMILES column name
```

**5. Filter by Value**

```bash
-filter_value "column > value"         # Comparison operators: >, <, >=, <=, ==, !=
```

Examples:

```bash
-filter_value "ic50 < 100"
-filter_value "activity == active"
-filter_value "logp >= 0"
```

**6. Drop Columns**

```bash
-drop_columns                          # Enable column dropping
-columns_to_drop "col1,col2"           # Columns to remove
```

#### Output

- Cleaned CSV file
- Detailed cleaning report (console + `reports/` directory)
- Shows rows removed at each step

---

### Data Visualization

Generate publication-quality plots following academic standards.

**Usage:**

```bash
moltrainer -i data.csv -visualize -plot_type TYPE -o output.png
```

#### Plot Types

**1. Distribution Plots**

```bash
moltrainer -i data.csv -visualize \
  -plot_type distribution \
  -columns "logp,mw,tpsa" \
  -o distribution.svg
```

**2. Correlation Heatmap**

```bash
moltrainer -i data.csv -visualize \
  -plot_type correlation \
  -columns "logp,mw,tpsa" \
  -o correlation.png
```

**3. Boxplot**

```bash
moltrainer -i data.csv -visualize \
  -plot_type boxplot \
  -columns "logp,mw" \
  -title "LogP and MW Distribution" \
  -o boxplot.jpg
```

#### Options

```bash
-plot_type TYPE          # distribution, correlation, boxplot, all
-columns "col1,col2"     # Columns to plot (optional, defaults to all numeric)
-sample_size N           # Sample size: absolute number or percentage (e.g., "50%")
-title "My Title"        # Custom plot title
-xlabel "X Label"        # Custom x-axis label
-ylabel "Y Label"        # Custom y-axis label
```

#### Output Formats

- `.svg` - Vector format (recommended for publications)
- `.png` - Raster format (DPI=600 for high quality)
- `.jpg` / `.jpeg` - Compressed format

#### Visualization Standards

- Font: Arial, 8-10pt
- Style: Nature/Science publication standards
- Color palette: Colorblind-friendly
- DPI: 600 for raster formats
- No top/right spines
- Clear legends and labels

---

### Data Splitting

Split dataset into train, validation, and test sets.

**Basic Usage:**

```bash
moltrainer -i data.csv -split
```

This creates three files:
- `data_train.csv` (70%)
- `data_val.csv` (15%)
- `data_test.csv` (15%)

#### Custom Ratios

```bash
moltrainer -i data.csv -split \
  -train_ratio 0.8 \
  -val_ratio 0.1 \
  -test_ratio 0.1
```

#### 2-Way Split (Train/Test Only)

```bash
moltrainer -i data.csv -split \
  -train_ratio 0.8 \
  -val_ratio 0.0 \
  -test_ratio 0.2
```

#### Stratified Split

For classification tasks:

```bash
moltrainer -i data.csv -split \
  -stratify activity
```

This ensures balanced class distribution across splits.

#### Options

```bash
-train_ratio 0.7         # Training set ratio (default: 0.7)
-val_ratio 0.15          # Validation set ratio (default: 0.15)
-test_ratio 0.15         # Test set ratio (default: 0.15)
-stratify COLUMN         # Column for stratified sampling
-shuffle                 # Shuffle before split (default: True)
```

**Note:** Ratios must sum to 1.0

#### Output

- Three CSV files with automatic naming
- Split report with class distributions (if stratified)
- Report saved to `reports/` directory

---

### Model Training

Train machine learning models with automated workflows.

#### Basic Training with SMILES

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -o results/
```

#### Training with Numeric Features

```bash
moltrainer -i train.csv -train \
  -target ic50 \
  -features "logp,mw,tpsa" \
  -task regression \
  -o results/
```

#### Model Types

```bash
-model rf                # Random Forest (default)
-model svm               # Support Vector Machine
-model xgb               # XGBoost
-model lgb               # LightGBM
-model lr                # Logistic/Linear Regression
```

#### Task Types

```bash
-task auto               # Auto-detect (default)
-task classification     # Classification
-task regression         # Regression
```

#### Feature Engineering Options ⭐ NEW

MolTrainer supports comprehensive molecular feature generation from SMILES, including **200+ physicochemical descriptors** and **5 types of molecular fingerprints**.

**Available Feature Types:**

1. **Descriptors Only** (default)
2. **Fingerprints Only**
3. **Combined** (Descriptors + Fingerprints)

##### 1. Physicochemical Descriptors

Three descriptor sets are available:

```bash
-feat_type descriptors -desc_set basic      # 10 basic descriptors (fast)
-feat_type descriptors -desc_set extended   # ~30 descriptors (moderate)
-feat_type descriptors -desc_set all        # 200+ descriptors (comprehensive)
```

**Basic Descriptors (10):**
- Molecular Weight, LogP
- H-bond Donors/Acceptors
- TPSA (Topological Polar Surface Area)
- Rotatable Bonds
- Ring Counts (Aromatic, Saturated, Aliphatic)

**Extended Descriptors (~30):**
- Basic + Molecular Refractivity
- Topological indices (BertzCT, Chi, Kappa)
- Electronic properties
- Structural features (CSP3 fraction, Heteroatoms)
- Pharmacophore features (LabuteASA, PEOE_VSA, SMR_VSA)

**All Descriptors (200+):**
- Complete RDKit 2D descriptor set
- Comprehensive molecular characterization

**Example:**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type descriptors \
  -desc_set extended \
  -o results/
```

##### 2. Molecular Fingerprints

Five fingerprint types supported:

```bash
-feat_type fingerprints -fp_type morgan       # Morgan (circular) fingerprint
-feat_type fingerprints -fp_type maccs        # MACCS keys (167 bits, fixed)
-feat_type fingerprints -fp_type rdk          # RDKit fingerprint
-feat_type fingerprints -fp_type atompair     # Atom pair fingerprint
-feat_type fingerprints -fp_type topological  # Topological torsion fingerprint
```

**Fingerprint Options:**

```bash
-fp_bits 2048           # Fingerprint bit size (default: 2048)
-fp_radius 2            # Morgan fingerprint radius (default: 2)
```

**Common Bit Sizes:**
- Small: 256-512 bits (faster, less memory)
- Medium: 1024 bits (balanced)
- Large: 2048-4096 bits (more information)

**Example - Morgan Fingerprint:**

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

**Example - MACCS Keys:**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type maccs \
  -o results/
```

##### 3. Combined Features

Combine descriptors and fingerprints for maximum information:

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

This generates: **10 descriptors + 512 fingerprint bits = 522 features**

**Popular Combinations:**

```bash
# Basic descriptors + MACCS (fast, interpretable)
-feat_type combined -desc_set basic -fp_type maccs

# Extended descriptors + Morgan 1024 (balanced)
-feat_type combined -desc_set extended -fp_type morgan -fp_bits 1024

# All descriptors + Morgan 2048 (comprehensive)
-feat_type combined -desc_set all -fp_type morgan -fp_bits 2048
```

##### 4. Automatic Fingerprint Length Optimization ⭐

Automatically find the optimal fingerprint length by training models at different bit sizes:

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

**Optimization Parameters:**

```bash
-optimize_fp            # Enable fingerprint length optimization
-fp_start 16            # Starting bit size (default: 16)
-fp_step 16             # Step size for increments (default: 16)
-fp_max 2048            # Maximum bit size (default: 2048)
```

**How it works:**
1. Tests fingerprint lengths from 16 to 2048 bits (step=16)
2. Trains a model at each bit size with cross-validation
3. Reports the optimal length with best performance
4. Trains final model with optimal fingerprint length

**Example output:**

```
Testing 16 bits... Score: 0.75 (+/- 0.05)
Testing 32 bits... Score: 0.82 (+/- 0.04)
Testing 48 bits... Score: 0.85 (+/- 0.03)
...
Testing 512 bits... Score: 0.91 (+/- 0.02)  ← Best
Testing 1024 bits... Score: 0.90 (+/- 0.03)
...

Best fingerprint length: 512 bits
Best score: 0.9100
```

**Tips for Optimization:**

- **Quick search**: `-fp_start 64 -fp_step 64 -fp_max 1024`
- **Fine search**: `-fp_start 256 -fp_step 32 -fp_max 768`
- **Comprehensive**: `-fp_start 16 -fp_step 16 -fp_max 2048` (default, slower)

##### Feature Engineering Best Practices

**For Quick Experiments:**
```bash
-feat_type descriptors -desc_set basic      # Fastest
```

**For Publication-Quality Models:**
```bash
-feat_type combined -desc_set extended -fp_type morgan -fp_bits 1024
```

**For Maximum Performance:**
```bash
-feat_type combined -desc_set all -fp_type morgan -optimize_fp
```

**Memory Considerations:**
- Large fingerprints (2048+ bits) on big datasets may use significant memory
- Consider sampling data first if memory is limited
- Use `-desc_set basic` instead of `all` for large datasets

##### Feature Type Selection Guide

| Task | Recommended Features | Rationale |
|------|---------------------|-----------|
| ADMET prediction | Combined (extended + Morgan) | Needs both structure and property info |
| Activity classification | Fingerprints (Morgan/MACCS) | Structure-activity relationship |
| Property regression | Descriptors (extended/all) | Direct property calculation |
| Similarity search | MACCS or Morgan | Fast comparison |
| Interpretable models | Descriptors only | Named, interpretable features |

#### Hyperparameter Search

**Random Search (Recommended):**

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

**Grid Search:**

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -search grid \
  -search_depth shallow \
  -search_cv 3 \
  -o results/
```

**Search Options:**

```bash
-search METHOD           # none, grid, random (default: none)
-search_depth LEVEL      # shallow, deep (default: shallow)
-search_iter N           # Iterations for random search (default: 10)
-search_cv N             # CV folds for search (default: 3)
-search_timeout SECONDS  # Maximum search time
```

**Search Depth:**

- `shallow`: Faster, fewer parameters (good for quick experiments)
- `deep`: Thorough, more parameters (better performance, longer time)

#### Automatic Data Splitting

If validation/test sets are not provided:

```bash
# 3-way split (train/val/test)
-auto_split 3way         # Default

# 2-way split (train/test)
-auto_split 2way

# No auto-split (must provide -val and -test)
-auto_split none
```

Custom split ratios:

```bash
-train_split 0.6         # Use 60% for training (3-way: 60/20/20, 2-way: 60/40)
```

#### Cross-Validation

```bash
-cv 5                    # 5-fold CV (default)
-cv 10                   # 10-fold CV
-no_cv                   # Disable CV
```

#### Using Separate Validation/Test Sets

```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -val validation.csv \
  -test test.csv \
  -auto_split none \
  -o results/
```

#### Training Output

All files are saved to the specified output folder:

1. **Model file**: `{basename}_model.pkl`
   - Contains model, metadata, and label encoder
   - Self-contained for predictions

2. **Training log**: `{basename}_training_log.txt`
   - Detailed training information
   - Hyperparameters, timing, results

3. **Plots** (SVG + PNG):
   - Feature importance (for tree-based models)
   - Confusion matrix (classification)
   - Prediction scatter plot (regression)

4. **Plot data** (CSV):
   - Data used to generate each plot
   - For custom visualization

5. **Training report**: Saved to `reports/` directory

#### Invalid SMILES Handling

Invalid SMILES are automatically skipped with warnings:

```
Warning: 5 invalid SMILES found and skipped
Recommendation: Use -clean -validate_smiles before training
```

---

### Model Prediction

Make predictions using trained models.

#### View Model Information

```bash
moltrainer -model_info results/model.pkl
```

**Output:**
- Model type and task
- Training date
- Feature information (SMILES or numeric columns)
- Class labels (classification)
- Hyperparameters
- Cross-validation scores
- Usage instructions

#### Make Predictions

```bash
moltrainer -predict \
  -load_model results/model.pkl \
  -i new_data.csv \
  -o predictions.csv
```

#### Requirements

The input CSV must contain:
- **For SMILES-based models**: The same SMILES column
- **For numeric feature models**: The same feature columns

#### Prediction Output

**Classification:**
- `predicted_{target}`: Predicted class label
- `probability_{class1}`, `probability_{class2}`, ...: Class probabilities

**Regression:**
- `predicted_{target}`: Predicted numeric value

**Invalid Data:**
- Rows with invalid SMILES or missing features are marked with NaN in prediction columns
- Original data is preserved

#### Example Output

```csv
compound_id,smiles,activity,predicted_activity,probability_active,probability_inactive
C001,CCO,active,active,0.94,0.06
C002,INVALID,active,NaN,NaN,NaN
C003,c1ccccc1,inactive,inactive,0.01,0.99
```

#### Prediction Report

- Console output and saved report
- Number of successful predictions
- Warnings for invalid data
- Model information summary

---

## Advanced Features

### Configuration Files

For complex training setups, use YAML or JSON configuration files.

#### Create Example Config

```bash
moltrainer -create_config my_config.yaml
```

#### Config File Structure

```yaml
# Input/Output
input_file: data/train.csv
output_folder: results/experiment_001
target_column: activity

# Feature Specification
smiles_column: smiles
# OR
# feature_columns:
#   - logp
#   - molecular_weight

# Optional: Validation/Test Data
validation_file: data/val.csv
test_file: data/test.csv

# Model Settings
model_type: rf
task: auto

# Hyperparameters
n_estimators: 100
max_depth: null
random_state: 42

# Cross-Validation
cv_folds: 5
no_cv: false

# Auto Data Splitting
auto_split_mode: 3way
train_split_ratio: null

# Hyperparameter Search
search_method: random
search_depth: deep
search_iterations: 20
search_cv_folds: 5

# Output
verbose: true
```

#### Run Training with Config

```bash
moltrainer -config my_config.yaml
```

#### Override Config with CLI Arguments

CLI arguments take precedence:

```bash
moltrainer -config my_config.yaml -n_estimators 200 -search_iter 30
```

---

## Output Files

### Automatic Report Directory

All reports are saved to `reports/` with timestamp:

```
reports/
├── 20251024_143000_descriptive_stats.txt
├── 20251024_143100_data_cleaning.txt
├── 20251024_143200_visualization.txt
├── 20251024_143300_data_split.txt
├── 20251024_143400_training.txt
└── 20251024_143500_prediction.txt
```

### Training Output Structure

```
results/
├── model_name_model.pkl                      # Model + metadata
├── model_name_training_log.txt               # Detailed log
├── model_name_feature_importance.png         # Plot (PNG)
├── model_name_feature_importance.svg         # Plot (SVG)
├── model_name_feature_importance_data.csv    # Plot data
├── model_name_confusion_matrix.png           # Classification
├── model_name_confusion_matrix.svg
├── model_name_confusion_matrix_data.csv
├── model_name_predictions.png                # Regression
├── model_name_predictions.svg
└── model_name_predictions_data.csv
```

---

## Troubleshooting

### Common Issues

#### 1. RDKit Import Error

```
ImportError: RDKit is required for SMILES featurization
```

**Solution:**

```bash
conda install -c conda-forge rdkit
```

#### 2. Invalid SMILES

```
Warning: 10 invalid SMILES found
```

**Solution:** Clean data first:

```bash
moltrainer -i data.csv -clean -validate_smiles -smiles_column smiles -o clean.csv
```

#### 3. Model File Not Compatible

```
Error: Model metadata not found. This model may be from an older version.
```

**Solution:** Retrain the model with the current version.

#### 4. Feature Column Mismatch

```
ValueError: Feature columns not found in input data: logp, mw
```

**Solution:** Ensure prediction data has the same columns used during training. Check model info:

```bash
moltrainer -model_info model.pkl
```

#### 5. Memory Error with Large Datasets

**Solutions:**
- Use sampling for visualization: `-sample_size 1000` or `-sample_size 10%`
- Reduce search iterations: `-search_iter 5`
- Use shallow search: `-search_depth shallow`

#### 6. XGBoost/LightGBM Not Found

```
ImportError: XGBoost not installed
```

**Solution:**

```bash
pip install xgboost lightgbm
```

---

## FAQ

### General Questions

**Q: What file formats are supported?**  
A: Currently only CSV format is supported. Ensure UTF-8 encoding.

**Q: Can I use my own descriptors instead of SMILES?**  
A: Yes, use `-features "col1,col2,col3"` with numeric feature columns.

**Q: How do I update MolTrainer?**  
A: For editable installation: `git pull` in the MolTrainer directory.

### Data Questions

**Q: How many samples do I need for training?**  
A: Minimum 50-100 samples, but 500+ recommended for robust models. Use cross-validation for small datasets.

**Q: Should I normalize my features?**  
A: For SVM and Logistic/Linear Regression, yes (automatic). For tree-based models (RF, XGBoost, LightGBM), not necessary.

**Q: Can I train on imbalanced data?**  
A: Yes, use stratified splitting (`-stratify`) and consider adjusting class weights (future feature).

### Training Questions

**Q: Which model should I use?**  
A: 
- Start with Random Forest (`rf`) - good default
- Try XGBoost (`xgb`) or LightGBM (`lgb`) for better performance
- Use SVM (`svm`) for small datasets
- Use Logistic/Linear Regression (`lr`) for baseline

**Q: Should I use Grid Search or Random Search?**  
A: Random Search is recommended for most cases (faster, good results). Use Grid Search for final fine-tuning on a narrow parameter range.

**Q: How long does hyperparameter search take?**  
A: Depends on:
- Search method: Random < Grid
- Search depth: Shallow < Deep
- Dataset size
- Model type

Use `-search_timeout SECONDS` to limit search time.

**Q: What's the difference between shallow and deep search?**  
A:
- **Shallow**: 3-5 parameters, 4-6 values each, faster
- **Deep**: 6-10 parameters, 5-12 values each, thorough

### Prediction Questions

**Q: Can I use a model on data from a different source?**  
A: Yes, as long as:
1. The same features/SMILES column exists
2. Feature distributions are similar (model may perform poorly otherwise)

**Q: How do I interpret prediction probabilities?**  
A: For classification, probabilities indicate model confidence:
- >0.8: High confidence
- 0.5-0.8: Moderate confidence
- <0.5: Low confidence

**Q: What if I get NaN predictions?**  
A: NaN indicates invalid input data (invalid SMILES or missing features). Check the input data quality.

### Output Questions

**Q: Where are reports saved?**  
A: All reports are automatically saved to the `reports/` directory with timestamps.

**Q: Can I change the output directory?**  
A: For training: Use `-o <folder>`. For reports: Currently fixed to `reports/` (customization coming soon).

**Q: What format should I use for plots?**  
A: 
- Publications: SVG (vector, scalable)
- Presentations: PNG (high DPI, widely compatible)
- Web: JPG (smaller file size)

---

## Citation

If you use MolTrainer in your research, please cite:

```bibtex
@software{moltrainer2025,
  title = {MolTrainer: Machine Learning Tool for Molecular Data},
  author = {MolTrainer Development Team},
  year = {2025},
  url = {https://github.com/yourusername/moltrainer}
}
```

---

## Support

- **Documentation**: This file and `help_Chinese.md`
- **Issues**: https://github.com/yourusername/moltrainer/issues
- **Discussions**: https://github.com/yourusername/moltrainer/discussions

---

## License

MIT License - See LICENSE file for details

---

**Last Updated:** October 2025  
**Version:** 0.1.0
