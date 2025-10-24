# MolTrainer

<div align="center">

**Professional Machine Learning Tool for Molecular Data**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAQcRMwGorYznAAAAQUlEQVQI12NgAAMhZUEGBMCEA0swIYBJQA==")](https://www.rdkit.org/)

*From data cleaning to model deployment - A complete ML pipeline for chemoinformatics*

[**Documentation**](help.md) • [**中文文档**](help_Chinese.md) • [**Quick Start**](QUICKSTART.md)

</div>

---

## ✨ Features

### 🔬 Comprehensive Molecular Features
- **200+ Physicochemical Descriptors** (basic, extended, all RDKit 2D descriptors)
- **5 Types of Molecular Fingerprints** (Morgan, MACCS, RDKit, AtomPair, Topological)
- **Automatic Fingerprint Optimization** (auto-search optimal bit length)
- **Combined Features** (descriptors + fingerprints)

### 🤖 Machine Learning
- **5 ML Algorithms**: Random Forest, SVM, XGBoost, LightGBM, Logistic/Linear Regression
- **Hyperparameter Search**: Grid Search & Random Search with shallow/deep modes
- **Auto Data Splitting**: Smart 2-way/3-way splits with stratification
- **Cross-Validation**: Configurable k-fold CV
- **Model Metadata**: Self-contained models with complete training context

### 🛠️ Data Processing
- **Descriptive Statistics**: Comprehensive data profiling
- **Data Cleaning**: 7 cleaning operations (duplicates, missing, outliers, SMILES validation, filtering, etc.)
- **Data Visualization**: Publication-quality plots (Nature/Science standards)
- **Data Splitting**: Stratified train/val/test splits
- **Data Sampling**: 5 sampling methods (random, stratified, systematic, head, tail)

### 🎯 Prediction & Deployment
- **Model Prediction**: Load trained models for new predictions
- **Model Information**: View complete model details and training history
- **Automatic SMILES Handling**: Auto-featurization with same parameters as training

### 📊 Professional Output
- **Academic-Quality Plots**: SVG + PNG (600 DPI) with publication standards
- **Comprehensive Reports**: Auto-saved timestamped reports
- **GROMACS-Style Interface**: Professional CLI with structured output
- **Configuration Files**: YAML/JSON support for complex experiments

---

## 🚀 Quick Start

### Installation

#### Step 1: Install RDKit (Required)

RDKit **must** be installed **before** MolTrainer:

```bash
# Method A: Using Conda (Recommended)
conda install -c conda-forge rdkit

# Method B: Using pip
pip install rdkit-pypi
```

#### Step 2: Install MolTrainer

**Option 1: Clone from GitHub (Recommended for development)**
```bash
git clone https://github.com/Marissapy/MolTrainer.git
cd MolTrainer
pip install -e .
```

**Option 2: Direct Installation**
```bash
pip install git+https://github.com/Marissapy/MolTrainer.git
```

**Option 3: In a Conda Environment (Cleanest)**
```bash
# Create and activate environment
conda create -n moltrainer python=3.9
conda activate moltrainer

# Install RDKit
conda install -c conda-forge rdkit

# Install MolTrainer (will auto-install other dependencies)
pip install git+https://github.com/Marissapy/MolTrainer.git
```

#### Step 3: Optional Dependencies
```bash
# For advanced ML models (XGBoost, LightGBM)
pip install xgboost lightgbm

# Or install with extras
pip install git+https://github.com/Marissapy/MolTrainer.git[advanced]
```

### Basic Usage

```bash
# View help
moltrainer -h
# Or if command not found: python -m moltrainer -h

# Descriptive statistics
moltrainer -i data.csv -desc_stats

# Clean data
moltrainer -i data.csv -clean -validate_smiles -remove_duplicates -o clean.csv

# Train a model with Morgan fingerprints
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -fp_bits 1024 \
  -o results/

# Make predictions
moltrainer -predict -load_model results/model.pkl -i new_data.csv -o predictions.csv
```

> **Note**: If `moltrainer` command is not found, use `python -m moltrainer` instead.
> See [troubleshooting](#-troubleshooting) below.

---

## 📦 Update MolTrainer

### If Installed with Git Clone (Editable Mode)
```bash
cd MolTrainer
git pull origin main
pip install -e . --upgrade
```

### If Installed Directly from GitHub
```bash
pip install --upgrade --force-reinstall git+https://github.com/Marissapy/MolTrainer.git
```

### Check Version
```bash
moltrainer -h  # Version shown in header
```

---

## 💡 Examples

### Example 1: Extended Descriptors + Hyperparameter Search
```bash
moltrainer -i train.csv -train \
  -target ic50 \
  -smiles smiles \
  -feat_type descriptors \
  -desc_set extended \
  -model xgb \
  -search random \
  -search_depth deep \
  -o results/
```

### Example 2: Combined Features (Descriptors + MACCS)
```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type combined \
  -desc_set basic \
  -fp_type maccs \
  -o results/
```

### Example 3: Automatic Fingerprint Optimization
```bash
moltrainer -i train.csv -train \
  -target activity \
  -smiles smiles \
  -feat_type fingerprints \
  -fp_type morgan \
  -optimize_fp \
  -fp_start 64 \
  -fp_step 64 \
  -fp_max 1024 \
  -o results/
```

### Example 4: Using Configuration File
```bash
# Create example config
moltrainer -create_config my_config.yaml

# Edit my_config.yaml, then run
moltrainer -config my_config.yaml
```

---

## 📚 Documentation

- **[Complete User Manual](help.md)** (English)
- **[完整用户手册](help_Chinese.md)** (中文)
- **[Quick Start Guide](QUICKSTART.md)**

---

## 🗂️ Project Structure

```
MolTrainer/
├── moltrainer/
│   ├── core/
│   │   ├── descriptive_stats.py      # Descriptive statistics
│   │   ├── data_cleaning.py          # Data cleaning operations
│   │   ├── visualization.py          # Academic-quality plots
│   │   ├── data_splitter.py          # Train/val/test splitting
│   │   ├── data_sampler.py           # Data sampling methods
│   │   ├── feature_engineering.py    # Molecular features ⭐ NEW
│   │   ├── model_trainer.py          # ML model training
│   │   └── predictor.py              # Model prediction ⭐ NEW
│   ├── utils/
│   │   ├── config_loader.py          # YAML/JSON config support
│   │   └── report_manager.py         # Report generation
│   ├── cli.py                        # Command-line interface
│   └── output.py                     # Formatted output
├── help.md                           # Full documentation (EN)
├── help_Chinese.md                   # Full documentation (中文)
├── QUICKSTART.md                     # Quick start guide
└── README.md                         # This file
```

---

## 🎯 Feature Comparison

| Feature | Basic Mode | Advanced Mode |
|---------|-----------|---------------|
| **Descriptors** | 10 basic | 200+ comprehensive |
| **Fingerprints** | Fixed bits | Optimized bits |
| **Features** | Descriptors only | Combined (desc + fp) |
| **Models** | Random Forest | 5 algorithms + search |
| **Validation** | Simple split | Stratified + CV |
| **Output** | Console | Reports + Plots + Logs |

---

## 🔬 Supported Feature Types

### Physicochemical Descriptors
- **Basic**: 10 descriptors (MW, LogP, H-donors/acceptors, TPSA, etc.)
- **Extended**: ~30 descriptors (+ MR, topology, electronic properties)
- **All**: 200+ RDKit 2D descriptors

### Molecular Fingerprints
- **Morgan**: Circular fingerprints (configurable radius & bits)
- **MACCS**: 167 structural keys
- **RDKit**: Daylight-like fingerprints
- **AtomPair**: Atom pair fingerprints
- **Topological**: Topological torsion fingerprints

---

## 🔧 Troubleshooting

### Command Not Found: `moltrainer`

If you get `command not found` error after installation:

**Quick Fix (Use Python module directly):**
```bash
python -m moltrainer -h
```

**Permanent Fix (Add to PATH):**
```bash
# Linux/Mac
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Or create an alias
echo 'alias moltrainer="python -m moltrainer"' >> ~/.bashrc
source ~/.bashrc
```

**Best Solution (Use Conda environment):**
```bash
conda create -n moltrainer python=3.9
conda activate moltrainer
conda install -c conda-forge rdkit
pip install git+https://github.com/Marissapy/MolTrainer.git
# Now moltrainer command works directly
```

### Missing Dependencies After Installation

The current version auto-installs: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyyaml`, `art`, `colorama`

If you encounter import errors:
```bash
# Install missing dependencies
pip install matplotlib seaborn pyyaml

# Or reinstall with upgraded dependencies
pip install --upgrade --force-reinstall git+https://github.com/Marissapy/MolTrainer.git
```

### RDKit Import Error

RDKit must be installed separately:
```bash
# Recommended
conda install -c conda-forge rdkit

# Alternative
pip install rdkit-pypi
```

For more troubleshooting, see [INSTALL_GUIDE.md](INSTALL_GUIDE.md).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use MolTrainer in your research, please cite:

```bibtex
@software{moltrainer2025,
  title = {MolTrainer: Machine Learning Tool for Molecular Data},
  author = {MolTrainer Development Team},
  year = {2025},
  url = {https://github.com/Marissapy/MolTrainer},
  version = {0.1.0}
}
```

---

## 🙏 Acknowledgments

- Built with [RDKit](https://www.rdkit.org/) - Open-source cheminformatics
- Inspired by [GROMACS](https://www.gromacs.org/) - Professional output formatting
- Machine learning powered by [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [LightGBM](https://lightgbm.readthedocs.io/)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ for the chemoinformatics community

[Report Bug](https://github.com/Marissapy/MolTrainer/issues) • [Request Feature](https://github.com/Marissapy/MolTrainer/issues)

</div>
