# Changelog

All notable changes to MolTrainer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-24

### Added
- **Core Features**
  - Descriptive statistics analysis
  - Data cleaning with 7 operations (duplicates, missing values, outliers, SMILES validation, filtering, column operations)
  - Data visualization (distribution, correlation, boxplot) with publication-quality output (SVG, PNG 600DPI)
  - Data splitting (train/val/test) with stratification
  - Data sampling (5 methods: random, systematic, stratified, head, tail)
  
- **Feature Engineering**
  - 200+ RDKit physicochemical descriptors (basic, extended, all)
  - 5 molecular fingerprint types (Morgan, MACCS, RDKit, AtomPair, Topological)
  - Automatic fingerprint length optimization
  - Custom feature combinations via `-feat_spec`
  - Combined features (descriptors + fingerprints)

- **Machine Learning**
  - 5 ML algorithms: Random Forest, SVM, XGBoost, LightGBM, Logistic/Linear Regression
  - Hyperparameter search (Grid Search, Random Search) with shallow/deep modes
  - Automatic data splitting (2-way, 3-way)
  - Cross-validation
  - Model metadata for reproducibility

- **Prediction & Deployment**
  - Load trained models for prediction
  - Automatic feature reconstruction from model metadata
  - Model information display
  - Batch prediction with probabilities

- **Configuration Management**
  - YAML/JSON configuration file support
  - Example config generation

- **Documentation**
  - Comprehensive user manual (English & Chinese)
  - 10 common usage scenarios with examples
  - Parameter guide with rules and requirements
  - 5 common errors with solutions

### Fixed
- **Data Preparation Error** (Commit: 9531849)
  - Fixed "truth value of array with more than one element" error
  - Improved NaN handling in `_prepare_data` method
  - Now uses consistent pandas DataFrame/Series conversion

### Technical Details
- **Dependencies**: pandas, numpy, scikit-learn, RDKit, matplotlib, seaborn, pyyaml, art, colorama
- **Python**: 3.8+
- **License**: MIT

---

## Version History Summary

| Version | Date | Key Features |
|---------|------|--------------|
| 0.1.0 | 2025-10-24 | Initial release with full ML pipeline |

---

## Upcoming Features (Planned)

- Deep learning support (TensorFlow, PyTorch)
- Additional molecular descriptors (3D, quantum)
- Model ensemble methods
- Feature importance analysis
- SHAP values for model interpretation
- Web interface for easier access
- Docker containerization
- Automated model deployment

---

## Bug Reports and Feature Requests

Please report bugs and request features via [GitHub Issues](https://github.com/Marissapy/MolTrainer/issues).

---

## Contributors

MolTrainer Development Team

---

[0.1.0]: https://github.com/Marissapy/MolTrainer/releases/tag/v0.1.0

