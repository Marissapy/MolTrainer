"""
Command Line Interface for MolTrainer
"""

import argparse
import sys
from moltrainer.output import OutputFormatter
from moltrainer.core.descriptive_stats import DescriptiveStatsAnalyzer
from moltrainer.core.data_cleaning import DataCleaner
from moltrainer.core.visualization import DataVisualizer
from moltrainer.core.data_splitter import DataSplitter
from moltrainer.core.data_sampler import DataSampler
from moltrainer.core.model_trainer import ModelTrainer
from moltrainer.core.predictor import ModelPredictor
from moltrainer.utils.config_loader import ConfigLoader


class MolTrainerCLI:
    """Main CLI controller for MolTrainer"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.output = OutputFormatter()
        self.modules = {
            'desc_stats': DescriptiveStatsAnalyzer(),
            'clean': DataCleaner(),
            'visualize': DataVisualizer(),
            'split': DataSplitter(),
            'sample': DataSampler(),
            'train': ModelTrainer(),
            'predict': ModelPredictor(),
            # Add more modules here as needed
            # 'validate': ModelValidator(),
        }
    
    def _create_parser(self):
        """Create argument parser with all options"""
        parser = argparse.ArgumentParser(
            prog='moltrainer',
            description='MolTrainer - Machine Learning Tool for Molecular Data',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,
            epilog="""
Examples:
  moltrainer -i data.csv -desc_stats
  moltrainer -i data.csv -clean -o cleaned.csv
  moltrainer -i data.csv -visualize -o plot.png
  moltrainer -i data.csv -split

For detailed documentation, see help.md
            """
        )
        
        # Add custom help
        parser.add_argument('-h', '--help', action='store_true',
                          help='Show this help message')
        
        # Input/Output options
        parser.add_argument('-i', '-input', dest='input',
                          help='Input CSV file path')
        
        # Analysis options
        parser.add_argument('-desc_stats', '--descriptive-statistics',
                          dest='desc_stats',
                          action='store_true',
                          help='Perform descriptive statistics analysis')
        
        # Data cleaning options
        parser.add_argument('-clean', '--clean',
                          action='store_true',
                          help='Clean data (interactive mode if no specific options provided)')
        
        parser.add_argument('-remove_duplicates', '--remove-duplicates',
                          action='store_true',
                          help='Remove duplicate rows')
        
        parser.add_argument('-duplicate_subset', '--duplicate-subset',
                          help='Column name to check for duplicates')
        
        parser.add_argument('-handle_missing', '--handle-missing',
                          action='store_true',
                          help='Handle missing values')
        
        parser.add_argument('-missing_method', '--missing-method',
                          choices=['drop', 'fill'],
                          default='drop',
                          help='Method to handle missing values: drop or fill')
        
        parser.add_argument('-fill_method', '--fill-method',
                          choices=['mean', 'median', 'mode'],
                          default='mean',
                          help='Method to fill missing values (for numeric columns)')
        
        parser.add_argument('-fill_value', '--fill-value',
                          help='Specific value to fill missing values')
        
        parser.add_argument('-remove_outliers', '--remove-outliers',
                          action='store_true',
                          help='Remove outliers from numeric columns')
        
        parser.add_argument('-outlier_method', '--outlier-method',
                          choices=['iqr', 'zscore'],
                          default='iqr',
                          help='Method for outlier detection: iqr or zscore')
        
        parser.add_argument('-outlier_threshold', '--outlier-threshold',
                          type=float,
                          help='Threshold for outlier detection (default: 1.5 for IQR, 3 for Z-score)')
        
        parser.add_argument('-outlier_columns', '--outlier-columns',
                          help='Comma-separated column names for outlier detection')
        
        parser.add_argument('-validate_smiles', '--validate-smiles',
                          action='store_true',
                          help='Validate SMILES strings and remove invalid ones')
        
        parser.add_argument('-smiles_column', '--smiles-column',
                          default='smiles',
                          help='Column name containing SMILES strings (default: smiles)')
        
        parser.add_argument('-filter_value', '--filter-value',
                          action='append',
                          help='Filter rows by condition (e.g., "ic50 > 0", "activity != 0"). Can be used multiple times.')
        
        parser.add_argument('-drop_columns', '--drop-columns',
                          action='store_true',
                          help='Drop specified columns')
        
        parser.add_argument('-columns_to_drop', '--columns',
                          help='Comma-separated column names to drop')
        
        # Visualization options
        parser.add_argument('-visualize', '--visualize',
                          action='store_true',
                          help='Generate data visualizations')
        
        parser.add_argument('-plot_type', '--plot-type',
                          help='Plot type: distribution, correlation, boxplot, scatter, histogram, or "all" (comma-separated for multiple)')
        
        parser.add_argument('-columns', '--vis-columns',
                          dest='vis_columns',
                          help='Comma-separated column names to visualize')
        
        parser.add_argument('-xlabel', '--xlabel',
                          help='X-axis label for plots')
        
        parser.add_argument('-ylabel', '--ylabel',
                          help='Y-axis label for plots')
        
        parser.add_argument('-title', '--plot-title',
                          dest='plot_title',
                          help='Title for plots')
        
        parser.add_argument('-sample_size', '--sample-size',
                          help='Sample size for large datasets (e.g., 10000 or 50%%)')
        
        # Data splitting options
        parser.add_argument('-split', '--split',
                          action='store_true',
                          help='Split data into train/val/test sets')
        
        parser.add_argument('-train_ratio', '--train-ratio',
                          type=float, default=0.7,
                          help='Training set ratio (default: 0.7)')
        
        parser.add_argument('-val_ratio', '--val-ratio',
                          type=float, default=0.15,
                          help='Validation set ratio (default: 0.15)')
        
        parser.add_argument('-test_ratio', '--test-ratio',
                          type=float, default=0.15,
                          help='Test set ratio (default: 0.15)')
        
        parser.add_argument('-stratify', '--stratify',
                          help='Column name for stratified splitting')
        
        parser.add_argument('-shuffle', '--no-shuffle',
                          dest='shuffle', action='store_false',
                          help='Disable shuffling before split')
        
        # Sampling options
        parser.add_argument('-sample', '--sample',
                          action='store_true',
                          help='Sample data from dataset')
        
        parser.add_argument('-n', '--num-samples',
                          dest='num_samples',
                          help='Number of samples: absolute number (e.g., 100) or percentage (e.g., 50%%)')
        
        parser.add_argument('-sample_method', '--sample-method',
                          choices=['random', 'systematic', 'stratified', 'head', 'tail'],
                          default='random',
                          help='Sampling method (default: random)')
        
        parser.add_argument('-replace', '--with-replacement',
                          action='store_true',
                          help='Sample with replacement')
        
        # Training options
        parser.add_argument('-train', '--train',
                          action='store_true',
                          help='Train machine learning model')
        
        parser.add_argument('-model', '--model',
                          choices=['rf', 'svm', 'xgb', 'lgb', 'lr', 'all'],
                          default='rf',
                          help='Model type: rf (Random Forest), svm (SVM), xgb (XGBoost), lgb (LightGBM), lr (Logistic/Linear Regression), all (compare all models)')
        
        parser.add_argument('-target', '--target-column',
                          dest='target_column',
                          help='Target column name (required for training)')
        
        parser.add_argument('-smiles', '--smiles',
                          dest='train_smiles_column',
                          help='SMILES column name for featurization (training)')
        
        parser.add_argument('-features', '--feature-columns',
                          dest='features',
                          help='Comma-separated feature column names (if not using SMILES)')
        
        # Feature engineering options
        parser.add_argument('-feat_type', '--feature-type',
                          choices=['descriptors', 'fingerprints', 'combined'],
                          default='descriptors',
                          help='Feature type: descriptors, fingerprints, or combined (default: descriptors)')
        
        parser.add_argument('-desc_set', '--descriptor-set',
                          choices=['basic', 'extended', 'all'],
                          default='basic',
                          help='Descriptor set: basic (10), extended (~30), all (200+) (default: basic)')
        
        parser.add_argument('-fp_type', '--fingerprint-type',
                          choices=['morgan', 'maccs', 'rdk', 'atompair', 'topological'],
                          default='morgan',
                          help='Fingerprint type (default: morgan)')
        
        parser.add_argument('-fp_bits', '--fingerprint-bits',
                          type=int,
                          default=2048,
                          help='Fingerprint bit size (default: 2048, MACCS is fixed at 167)')
        
        parser.add_argument('-fp_radius', '--fingerprint-radius',
                          type=int,
                          default=2,
                          help='Morgan fingerprint radius (default: 2)')
        
        parser.add_argument('-optimize_fp', '--optimize-fingerprint-length',
                          action='store_true',
                          help='Optimize fingerprint length (search from 16 to 2048 bits)')
        
        parser.add_argument('-fp_start', '--fp-optimize-start',
                          type=int,
                          default=16,
                          help='Starting bits for fingerprint optimization (default: 16)')
        
        parser.add_argument('-fp_step', '--fp-optimize-step',
                          type=int,
                          default=16,
                          help='Step size for fingerprint optimization (default: 16)')
        
        parser.add_argument('-fp_max', '--fp-optimize-max',
                          type=int,
                          default=2048,
                          help='Maximum bits for fingerprint optimization (default: 2048)')
        
        parser.add_argument('-task', '--task-type',
                          choices=['auto', 'classification', 'regression'],
                          default='auto',
                          help='Task type (default: auto-detect)')
        
        parser.add_argument('-val', '--validation-file',
                          dest='val_file',
                          help='Validation data file')
        
        parser.add_argument('-test', '--test-file',
                          dest='test_file',
                          help='Test data file')
        
        parser.add_argument('-n_estimators', '--n-estimators',
                          type=int, default=100,
                          help='Number of trees (default: 100)')
        
        parser.add_argument('-max_depth', '--max-depth',
                          type=int,
                          help='Maximum tree depth (default: None)')
        
        parser.add_argument('-cv', '--cross-validation',
                          type=int, default=5,
                          dest='cv_folds',
                          help='Cross-validation folds (default: 5, 0 to disable)')
        
        parser.add_argument('-no_cv', '--no-cross-validation',
                          action='store_true',
                          help='Disable cross-validation')
        
        # Data splitting options for training
        parser.add_argument('-auto_split', '--auto-split-mode',
                          choices=['3way', '2way', 'none'],
                          default='3way',
                          help='Auto-split mode: 3way (train/val/test), 2way (train/test), none (no split)')
        
        parser.add_argument('-train_split', '--train-split-ratio',
                          type=float,
                          help='Training set ratio for auto-split (default: 0.7 for 3way, 0.8 for 2way)')
        
        # Hyperparameter search
        parser.add_argument('-search', '--hyperparameter-search',
                          choices=['grid', 'random', 'none'],
                          default='none',
                          help='Hyperparameter search method: grid (GridSearchCV), random (RandomizedSearchCV), none')
        
        parser.add_argument('-search_depth', '--search-depth',
                          choices=['shallow', 'deep'],
                          default='shallow',
                          help='Search depth: shallow (fewer params, faster) or deep (more params, thorough)')
        
        parser.add_argument('-search_iter', '--search-iterations',
                          type=int,
                          default=10,
                          help='Number of iterations for random search (default: 10)')
        
        parser.add_argument('-search_cv', '--search-cv-folds',
                          type=int,
                          default=3,
                          help='CV folds for hyperparameter search (default: 3)')
        
        parser.add_argument('-search_timeout', '--search-timeout',
                          type=int,
                          help='Maximum time (seconds) for hyperparameter search (default: no limit)')
        
        # Configuration file
        parser.add_argument('-config', '--config-file',
                          dest='config_file',
                          help='Path to configuration file (.yaml, .yml, or .json). Overrides command-line arguments.')
        
        parser.add_argument('-create_config', '--create-example-config',
                          nargs='?',
                          const='training_config_example.yaml',
                          dest='create_config',
                          metavar='FILENAME',
                          help='Create an example configuration file and exit')
        
        # Prediction options
        parser.add_argument('-predict', '--predict',
                          action='store_true',
                          help='Make predictions using trained model')
        
        parser.add_argument('-load_model', '--load-model-file',
                          dest='model_file',
                          help='Path to trained model (.pkl file) for prediction')
        
        parser.add_argument('-model_info', '--model-info',
                          dest='model_info',
                          help='Display comprehensive information about a trained model')
        
        # Validation options (placeholder for future)
        parser.add_argument('-validate', '--validate',
                          action='store_true',
                          help='Validate model performance')
        
        # Output options
        parser.add_argument('-o', '-output', dest='output',
                          help='Output file path')
        
        parser.add_argument('-v', '--verbose',
                          action='store_true',
                          help='Enable verbose output')
        
        return parser
    
    def run(self, args=None):
        """Execute the CLI with given arguments"""
        args = self.parser.parse_args(args)
        
        # Handle creating example config (before showing logo)
        if hasattr(args, 'create_config') and args.create_config:
            filename = args.create_config
            format_type = 'json' if filename.endswith('.json') else 'yaml'
            print("\nCreating example configuration file...")
            created_file = ConfigLoader.create_example_config(filename, format_type)
            print(f"\n✓ Example configuration file created: {created_file}")
            print(f"\nNext steps:")
            print(f"  1. Edit {created_file} with your training parameters")
            print(f"  2. Run training: moltrainer -config {created_file}")
            print()
            sys.exit(0)
        
        # Handle model info (before showing logo)
        if hasattr(args, 'model_info') and args.model_info:
            predictor = self.modules['predict']
            try:
                info = predictor.show_model_info(args.model_info)
                print(info)
                sys.exit(0)
            except Exception as e:
                print(f"\nError: {str(e)}")
                sys.exit(1)
        
        # Handle help
        if args.help:
            self._print_help()
            sys.exit(0)
        
        # Show logo
        self.output.print_header()
        
        # Load configuration from file if provided
        if hasattr(args, 'config_file') and args.config_file:
            config = ConfigLoader.load_config(args.config_file)
            args = self._merge_config_with_args(args, config)
        
        # Validate input
        if not args.input:
            self.output.print_error("Error: Input file is required. Use -h for help.")
            sys.exit(1)
        
        # Collect user inputs for display
        user_inputs = {
            'Input File': args.input,
            'Analysis Type': self._get_analysis_type(args),
            'Model Type': args.model if args.model else 'N/A',
            'Output File': args.output if args.output else 'stdout',
        }
        
        self.output.print_inputs(user_inputs)
        
        # Execute requested analysis
        try:
            result = self._execute_analysis(args)
            self.output.print_footer(success=True, additional_info=result)
        except Exception as e:
            self.output.print_footer(success=False, error_msg=str(e))
            sys.exit(1)
    
    def _merge_config_with_args(self, args, config):
        """Merge configuration file with command-line arguments
        
        Config file values take precedence over defaults but not over
        explicitly provided command-line arguments.
        """
        # For training, validate and merge config
        if 'input_file' in config:
            config = ConfigLoader.validate_training_config(config)
            
            # Map config keys to args attributes
            mapping = {
                'input_file': 'input',
                'output_folder': 'output',
                'target_column': 'target_column',
                'smiles_column': 'train_smiles_column',
                'feature_columns': 'features',
                'validation_file': 'val_file',
                'test_file': 'test_file',
                'model_type': 'model',
                'task': 'task',
                'n_estimators': 'n_estimators',
                'max_depth': 'max_depth',
                'cv_folds': 'cv_folds',
                'no_cv': 'no_cv',
                'verbose': 'verbose',
            }
            
            # Set training mode
            args.train = True
            
            # Merge config into args
            for config_key, arg_key in mapping.items():
                if config_key in config:
                    value = config[config_key]
                    
                    # Handle feature_columns specially (convert list to string)
                    if config_key == 'feature_columns' and isinstance(value, list):
                        value = ','.join(value)
                    
                    # Only override if not explicitly set on command line
                    # (Check if it's still the default value)
                    if not hasattr(args, arg_key) or getattr(args, arg_key, None) is None:
                        setattr(args, arg_key, value)
            
            print(f"  Loaded configuration from: {args.config_file}")
            if args.verbose:
                print(f"  Configuration keys: {list(config.keys())}")
        
        return args
    
    def _get_analysis_type(self, args):
        """Determine which analysis type is requested"""
        if args.desc_stats:
            return "Descriptive Statistics"
        elif args.clean:
            return "Data Cleaning"
        elif args.visualize:
            return "Data Visualization"
        elif args.split:
            return "Data Splitting"
        elif args.sample:
            return "Data Sampling"
        elif args.train:
            return "Model Training"
        elif args.predict:
            return "Prediction"
        elif args.validate:
            return "Model Validation"
        else:
            return "None (use -desc_stats, -clean, -visualize, -split, -sample, -train, -predict, or -validate)"
    
    def _execute_analysis(self, args):
        """Route to appropriate analysis module"""
        if args.desc_stats:
            analyzer = self.modules['desc_stats']
            return analyzer.analyze(args.input, args.output, args.verbose)
        
        elif args.clean:
            analyzer = self.modules['clean']
            # Prepare cleaning parameters
            cleaning_params = {
                'remove_duplicates': getattr(args, 'remove_duplicates', False),
                'duplicate_subset': getattr(args, 'duplicate_subset', None),
                'handle_missing': getattr(args, 'handle_missing', False),
                'missing_method': getattr(args, 'missing_method', 'drop'),
                'fill_method': getattr(args, 'fill_method', 'mean'),
                'fill_value': getattr(args, 'fill_value', None),
                'remove_outliers': getattr(args, 'remove_outliers', False),
                'outlier_method': getattr(args, 'outlier_method', 'iqr'),
                'outlier_threshold': getattr(args, 'outlier_threshold', None),
                'outlier_columns': getattr(args, 'outlier_columns', None),
                'validate_smiles': getattr(args, 'validate_smiles', False),
                'smiles_column': getattr(args, 'smiles_column', 'smiles'),
                'filter_value': getattr(args, 'filter_value', None),
                'drop_columns': getattr(args, 'drop_columns', False),
                'columns_to_drop': getattr(args, 'columns_to_drop', None),
            }
            # Process comma-separated columns
            if cleaning_params['outlier_columns']:
                cleaning_params['outlier_columns'] = cleaning_params['outlier_columns'].split(',')
            if cleaning_params['columns_to_drop']:
                cleaning_params['columns_to_drop'] = cleaning_params['columns_to_drop'].split(',')
            
            return analyzer.analyze(args.input, args.output, args.verbose, **cleaning_params)
        
        elif args.visualize:
            analyzer = self.modules['visualize']
            # Prepare visualization parameters
            vis_params = {
                'plot_type': getattr(args, 'plot_type', 'all'),
                'columns': getattr(args, 'vis_columns', '').split(',') if getattr(args, 'vis_columns', None) else None,
                'xlabel': getattr(args, 'xlabel', None),
                'ylabel': getattr(args, 'ylabel', None),
                'title': getattr(args, 'plot_title', None),
                'sample_size': getattr(args, 'sample_size', None),
            }
            return analyzer.analyze(args.input, args.output, args.verbose, **vis_params)
        
        elif args.split:
            analyzer = self.modules['split']
            # Prepare split parameters
            split_params = {
                'train_ratio': getattr(args, 'train_ratio', 0.7),
                'val_ratio': getattr(args, 'val_ratio', 0.15),
                'test_ratio': getattr(args, 'test_ratio', 0.15),
                'stratify_column': getattr(args, 'stratify', None),
                'shuffle': getattr(args, 'shuffle', True),
            }
            return analyzer.analyze(args.input, None, args.verbose, **split_params)
        
        elif args.sample:
            analyzer = self.modules['sample']
            # Prepare sampling parameters
            sample_params = {
                'sample_size': getattr(args, 'num_samples', None),
                'sample_method': getattr(args, 'sample_method', 'random'),
                'stratify_column': getattr(args, 'stratify', None),
                'replace': getattr(args, 'replace', False),
                'random_state': 42,
            }
            return analyzer.analyze(args.input, args.output, args.verbose, **sample_params)
        
        elif args.train:
            analyzer = self.modules['train']
            
            # Get features parameter
            features_str = getattr(args, 'features', None)
            feature_cols = None
            if features_str:
                feature_cols = [f.strip() for f in features_str.split(',') if f.strip()]
            
            # Prepare training parameters
            train_params = {
                'model_type': getattr(args, 'model', 'rf'),
                'target_column': getattr(args, 'target_column', None),
                'smiles_column': getattr(args, 'train_smiles_column', None),
                'feature_columns': feature_cols,
                'task': getattr(args, 'task', 'auto'),
                'val_file': getattr(args, 'val_file', None),
                'test_file': getattr(args, 'test_file', None),
                'n_estimators': getattr(args, 'n_estimators', 100),
                'max_depth': getattr(args, 'max_depth', None),
                'cv_folds': getattr(args, 'cv_folds', 5),
                'no_cv': getattr(args, 'no_cv', False),
                'auto_split_mode': getattr(args, 'auto_split', '3way'),
                'train_split_ratio': getattr(args, 'train_split', None),
                'search_method': getattr(args, 'search', 'none'),
                'search_depth': getattr(args, 'search_depth', 'shallow'),
                'search_iterations': getattr(args, 'search_iter', 10),
                'search_cv_folds': getattr(args, 'search_cv', 3),
                'search_timeout': getattr(args, 'search_timeout', None),
                # Feature engineering options
                'feature_type': getattr(args, 'feat_type', 'descriptors'),
                'descriptor_set': getattr(args, 'desc_set', 'basic'),
                'fingerprint_type': getattr(args, 'fp_type', 'morgan'),
                'fingerprint_bits': getattr(args, 'fp_bits', 2048),
                'fingerprint_radius': getattr(args, 'fp_radius', 2),
                'optimize_fp_length': getattr(args, 'optimize_fp', False),
                'fp_optimize_start': getattr(args, 'fp_start', 16),
                'fp_optimize_step': getattr(args, 'fp_step', 16),
                'fp_optimize_max': getattr(args, 'fp_max', 2048),
            }
            
            return analyzer.analyze(args.input, args.output, args.verbose, **train_params)
        
        elif args.predict:
            analyzer = self.modules['predict']
            # Prepare prediction parameters
            pred_params = {
                'model_file': getattr(args, 'model_file', None),
            }
            return analyzer.analyze(args.input, args.output, args.verbose, **pred_params)
        
        elif args.validate:
            # Placeholder for future implementation
            return "Validation module not yet implemented. Coming soon!"
        
        else:
            return "No analysis specified. Use -h for help."
    
    def _print_help(self):
        """Print concise help message"""
        help_text = """
MolTrainer - Machine Learning Tool for Molecular Data (v0.1.0)

USAGE:
  moltrainer -i <input.csv> [OPTIONS]

CORE MODULES:
  -desc_stats                   Descriptive statistics analysis
  -clean -o <output.csv>        Data cleaning (interactive/batch mode)
  -visualize -o <plot.png>      Generate academic-quality plots
  -split                        Split data into train/val/test sets
  -sample -o <output.csv>       Sample data from dataset
  -train -o <folder>            Train ML models with hyperparameter search
  -predict -load_model <.pkl>   Make predictions using trained model

COMMON OPTIONS:
  -i, -input FILE               Input CSV file (required for most modules)
  -o, -output FILE/FOLDER       Output file or folder (required for some modules)
  -v, --verbose                 Verbose output
  -h, --help                    Show this help

TRAINING OPTIONS (use with -train):
  -target COLUMN                Target column name (required)
  -smiles COLUMN                SMILES column for featurization
  -features "col1,col2"         Numeric feature columns (if not using SMILES)
  -model TYPE                   Model: rf, svm, xgb, lgb, lr (default: rf)
  -task TYPE                    Task: auto, classification, regression
  -search METHOD                Search: none, grid, random (default: none)
  -search_depth LEVEL           Search depth: shallow, deep (default: shallow)
  -auto_split MODE              Auto-split: 3way, 2way, none (default: 3way)
  -config FILE                  Load training config from YAML/JSON file

PREDICTION OPTIONS (use with -predict):
  -load_model FILE              Path to trained model (.pkl file)
  -model_info FILE              Display model information (no prediction)

UTILITIES:
  -create_config [file.yaml]    Create example training config file

EXAMPLES:
  # Descriptive statistics
  moltrainer -i data.csv -desc_stats

  # Data cleaning
  moltrainer -i data.csv -clean -remove_duplicates -validate_smiles -o clean.csv

  # Sample data (random 50%)
  moltrainer -i data.csv -sample -n 50% -o sampled.csv

  # Train classification model with SMILES
  moltrainer -i train.csv -train -target activity -smiles smiles -o results/

  # Train with hyperparameter search
  moltrainer -i train.csv -train -target activity -smiles smiles -model xgb \\
    -search random -search_depth deep -o results/

  # Train using config file
  moltrainer -config my_config.yaml

  # Make predictions
  moltrainer -predict -load_model results/model.pkl -i new_data.csv -o pred.csv

  # View model info
  moltrainer -model_info results/model.pkl

DOCUMENTATION:
  Detailed docs: help.md (English) or help_Chinese.md (中文)
  Reports: Automatically saved to reports/ directory
  GitHub: https://github.com/yourusername/moltrainer
"""
        print(help_text)


def main():
    """Main entry point"""
    cli = MolTrainerCLI()
    cli.run()


if __name__ == "__main__":
    main()

