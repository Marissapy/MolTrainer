# Quick Start Guide

## Installation

1. Install dependencies using conda:
```bash
conda install pandas numpy scikit-learn
pip install art colorama
```

2. Install MolTrainer in development mode:
```bash
pip install -e .
```

## First Run

Try the descriptive statistics on your data:

```bash
moltrainer -i 8gjc_single_C2.csv -desc_stats
```

## Command Reference

### Get Help
```bash
moltrainer -h
```

### Descriptive Statistics
```bash
# Print to console
moltrainer -i data.csv -desc_stats

# Save to file
moltrainer -i data.csv -desc_stats -o stats_report.txt

# Verbose mode
moltrainer -i data.csv -desc_stats -v
```

### Future Commands (Coming Soon)

```bash
# Train a model
moltrainer -i data.csv -train -model rf

# Make predictions
moltrainer -i data.csv -predict -model trained_model.pkl

# Validate model
moltrainer -i data.csv -validate -model trained_model.pkl
```

## Output Format

MolTrainer provides GROMACS-style professional output:

1. **ASCII Art Logo** - Shows "MolTrainer" on startup
2. **Input Parameters** - Displays all user inputs
3. **Analysis Results** - Shows the analysis output
4. **Status** - SUCCESS or FAILED
5. **Inspirational Quote** - A random quote from famous scientists and thinkers
6. **Timestamp** - When the analysis completed

## Project Structure

```
moltrainer/
├── cli.py              # Command-line interface
├── output.py           # Output formatting module
├── core/
│   ├── base.py         # Base analyzer class
│   └── descriptive_stats.py  # Descriptive statistics
└── utils/
    └── quotes.py       # Inspirational quotes
```

## Adding Custom Modules

See README.md for instructions on adding new analysis modules.

