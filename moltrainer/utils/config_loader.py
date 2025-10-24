"""
Configuration File Loader
Supports YAML and JSON configuration files for training
"""

import json
from pathlib import Path


class ConfigLoader:
    """Load and parse configuration files for training"""
    
    @staticmethod
    def load_config(config_file):
        """
        Load configuration from YAML or JSON file
        
        Args:
            config_file: Path to configuration file (.yaml, .yml, or .json)
            
        Returns:
            dict: Configuration dictionary
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Determine file format
        suffix = config_path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return ConfigLoader._load_yaml(config_path)
        elif suffix == '.json':
            return ConfigLoader._load_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")
    
    @staticmethod
    def _load_yaml(file_path):
        """Load YAML configuration file"""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. Install with: pip install pyyaml"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def _load_json(file_path):
        """Load JSON configuration file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    @staticmethod
    def validate_training_config(config):
        """
        Validate training configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            dict: Validated configuration with defaults
        """
        required_fields = ['input_file', 'target_column']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field missing in config: {field}")
        
        # Check feature specification
        if 'smiles_column' not in config and 'feature_columns' not in config:
            raise ValueError(
                "Must specify either 'smiles_column' or 'feature_columns' in config"
            )
        
        # Set defaults
        defaults = {
            'model_type': 'rf',
            'task': 'auto',
            'n_estimators': 100,
            'max_depth': None,
            'cv_folds': 5,
            'no_cv': False,
            'auto_split_mode': '3way',
            'train_split_ratio': None,
            'search_method': 'none',
            'search_iterations': 10,
            'search_cv_folds': 3,
            'random_state': 42,
            'verbose': False,
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    @staticmethod
    def create_example_config(output_file='training_config_example.yaml', format='yaml'):
        """
        Create an example configuration file
        
        Args:
            output_file: Output file path
            format: 'yaml' or 'json'
        """
        example_config = {
            '# Training Configuration Example': None,
            'input_file': 'data/train.csv',
            'output_folder': 'results/experiment_001',
            'target_column': 'activity',
            
            '# Feature Specification (choose one)': None,
            'smiles_column': 'smiles',
            '# feature_columns': ['logp', 'molecular_weight', 'tpsa'],
            
            '# Optional: Validation and Test Sets': None,
            'validation_file': 'data/val.csv',
            'test_file': 'data/test.csv',
            
            '# Model Configuration': None,
            'model_type': 'rf',
            'task': 'auto',
            
            '# Hyperparameters': None,
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            
            '# Cross-Validation': None,
            'cv_folds': 5,
            'no_cv': False,
            
            '# Output Options': None,
            'verbose': True,
        }
        
        # Remove comment keys for actual config
        clean_config = {k: v for k, v in example_config.items() if not k.startswith('#')}
        
        if format == 'yaml':
            try:
                import yaml
            except ImportError:
                raise ImportError("PyYAML required. Install with: pip install pyyaml")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# MolTrainer Training Configuration\n")
                f.write("# This is an example configuration file for training models\n\n")
                
                # Write sections with comments
                sections = [
                    ('# Input Files', ['input_file', 'output_folder', 'target_column']),
                    ('# Feature Specification (use either smiles_column OR feature_columns)', 
                     ['smiles_column']),
                    ('# Optional: Additional Data', ['validation_file', 'test_file']),
                    ('# Model Settings', ['model_type', 'task']),
                    ('# Hyperparameters', ['n_estimators', 'max_depth', 'random_state']),
                    ('# Cross-Validation', ['cv_folds', 'no_cv']),
                    ('# Output', ['verbose']),
                ]
                
                for comment, keys in sections:
                    f.write(f"{comment}\n")
                    section_data = {k: clean_config[k] for k in keys if k in clean_config}
                    yaml.dump(section_data, f, default_flow_style=False, sort_keys=False)
                    f.write("\n")
                
                # Add commented alternative
                f.write("# Alternative: Use numeric features instead of SMILES\n")
                f.write("# feature_columns:\n")
                f.write("#   - logp\n")
                f.write("#   - molecular_weight\n")
                f.write("#   - tpsa\n")
        
        elif format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file

