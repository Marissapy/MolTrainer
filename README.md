# MolTrainer

<div align="center">

**Machine Learning Tool for Molecular Data**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAQcRMwGorYznAAAAQUlEQVQI12NgAAMhZUEGBMCEA0swIYBJQA==")](https://www.rdkit.org/)

*Complete ML pipeline for SMILES-based chemoinformatics: data processing, training, and prediction*

[**Documentation**](help.md) • [**中文文档**](help_Chinese.md)

</div>

---

## 🚀 Installation

```bash
# Create and activate conda environment
conda create -n moltrainer python=3.9
conda activate moltrainer

# Install RDKit (required for SMILES processing)
conda install -c conda-forge rdkit

# Install MolTrainer
pip install git+https://github.com/Marissapy/MolTrainer.git

# Optional: Install advanced ML models
pip install xgboost lightgbm
```

> **Note**: If `moltrainer` command is not found, use `python -m moltrainer` instead.

---

## 💡 Quick Examples

```bash
# View help
moltrainer -h

# Descriptive statistics
moltrainer -i data.csv -desc_stats

# Clean data
moltrainer -i data.csv -clean -validate_smiles -remove_duplicates -o clean.csv

# Train a classification model
moltrainer -i train.csv -train -target activity -smiles smiles -o results/

# Train with Morgan fingerprints + hyperparameter search
moltrainer -i train.csv -train \
  -target activity -smiles smiles \
  -feat_type fingerprints -fp_type morgan -fp_bits 1024 \
  -model xgb -search random -search_depth deep \
  -o results/

# Make predictions
moltrainer -predict -load_model results/model.pkl -i new_data.csv -o predictions.csv
```

---

## 📦 Update

```bash
# Update to latest version
pip install --upgrade --force-reinstall git+https://github.com/Marissapy/MolTrainer.git
```

---

## ✨ Key Features

**Data Processing**: Descriptive statistics • Data cleaning (7 operations) • SMILES validation • Publication-quality plots (SVG + PNG 600DPI) • Train/val/test splitting • Data sampling (5 methods)

**Feature Engineering**: 200+ RDKit descriptors (basic/extended/all) • 5 fingerprint types (Morgan, MACCS, RDKit, AtomPair, Topological) • Automatic fingerprint length optimization • Combined features

**Machine Learning**: 5 algorithms (RF, SVM, XGBoost, LightGBM, Logistic/Linear Regression) • Grid/Random hyperparameter search • Auto data splitting • Cross-validation • Self-contained model metadata

**Prediction**: Load trained models • Automatic feature reconstruction • Batch prediction with probabilities

---

## 📚 Documentation

- **[Complete Manual](help.md)** (English)
- **[完整手册](help_Chinese.md)** (中文)
- **[Installation Guide](INSTALL_GUIDE.md)** (Troubleshooting)

---

## 🔧 Troubleshooting

**Command not found?** Use `python -m moltrainer` instead, or see [Installation Guide](INSTALL_GUIDE.md).

**Import errors?** Make sure RDKit is installed: `conda install -c conda-forge rdkit`

---

## 📄 License & Citation

MIT License • If you use MolTrainer in research, please cite: `https://github.com/Marissapy/MolTrainer`

Built with [RDKit](https://www.rdkit.org/) • [scikit-learn](https://scikit-learn.org/) • [XGBoost](https://xgboost.readthedocs.io/) • [LightGBM](https://lightgbm.readthedocs.io/)

---

<div align="center">

**⭐ If you find MolTrainer useful, please star this repo! ⭐**

[Report Bug](https://github.com/Marissapy/MolTrainer/issues) • [Request Feature](https://github.com/Marissapy/MolTrainer/issues) • [Contribute](https://github.com/Marissapy/MolTrainer/pulls)

</div>
