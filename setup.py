"""
Setup script for MolTrainer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="moltrainer",
    version="0.1.0",
    author="MolTrainer Team",
    description="A Machine Learning Tool for Molecular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core Data Processing
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        
        # Machine Learning
        "scikit-learn>=1.3.0",
        
        # Cheminformatics
        # Note: RDKit must be installed separately
        # Recommended: conda install -c conda-forge rdkit
        # Alternative: pip install rdkit-pypi
        # RDKit is NOT auto-installed to avoid platform issues
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # Configuration Files
        "pyyaml>=6.0",
        
        # CLI and Output Formatting
        "art>=6.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "advanced": [
            "xgboost>=2.0.0",
            "lightgbm>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moltrainer=moltrainer.cli:main",
        ],
    },
)

