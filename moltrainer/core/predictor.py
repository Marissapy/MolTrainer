"""
Model Prediction Module
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class ModelPredictor(BaseAnalyzer):
    """Load trained model and make predictions on new data"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
        self.model = None
        self.metadata = None
        self.label_encoder = None
    
    def load_model(self, model_file, verbose=False):
        """
        Load model and metadata from pickle file
        
        Args:
            model_file: Path to model pickle file
            verbose: Enable verbose output
        
        Returns:
            dict: Model metadata
        """
        model_path = Path(model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Handle both old and new model formats
        if isinstance(model_package, dict):
            if 'model' in model_package:
                # New format with metadata
                self.model = model_package['model']
                self.metadata = model_package.get('metadata', {})
                self.label_encoder = model_package.get('label_encoder')
                
                # Handle old format where metadata fields were at top level
                if not self.metadata:
                    self.metadata = {
                        'task': model_package.get('task'),
                        'feature_names': model_package.get('feature_names'),
                        'target_column': model_package.get('target_column'),
                        'smiles_column': model_package.get('smiles_column'),
                    }
            else:
                # Very old format - just the model
                self.model = model_package
                self.metadata = {}
        else:
            # Direct model object (oldest format)
            self.model = model_package
            self.metadata = {}
        
        if verbose:
            print(f"   Model loaded from: {model_path.name}")
            print(f"   Task: {self.metadata.get('task', 'Unknown')}")
            print(f"   Features: {self.metadata.get('n_features', 'Unknown')}")
        
        return self.metadata
    
    def show_model_info(self, model_file):
        """
        Display comprehensive model information
        
        Args:
            model_file: Path to model pickle file
        
        Returns:
            str: Formatted model information
        """
        self.load_model(model_file, verbose=False)
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("MODEL INFORMATION")
        lines.append("="*80)
        
        lines.append("\nBASIC INFO:")
        lines.append("-"*80)
        lines.append(f"Model File: {Path(model_file).name}")
        lines.append(f"Model Type: {self.metadata.get('model_type', 'Unknown')}")
        lines.append(f"Task: {self.metadata.get('task', 'Unknown')}")
        lines.append(f"Version: {self.metadata.get('version', 'Unknown')}")
        lines.append(f"Training Date: {self.metadata.get('training_date', 'Unknown')}")
        
        lines.append("\nDATA INFO:")
        lines.append("-"*80)
        lines.append(f"Target Column: {self.metadata.get('target_column', 'Unknown')}")
        lines.append(f"Number of Features: {self.metadata.get('n_features', 'Unknown')}")
        lines.append(f"Training Samples: {self.metadata.get('n_samples_train', 'Unknown')}")
        
        lines.append("\nFEATURE INFO:")
        lines.append("-"*80)
        smiles_col = self.metadata.get('smiles_column')
        feature_cols = self.metadata.get('feature_columns')
        
        if smiles_col:
            lines.append(f"Feature Type: SMILES Descriptors")
            lines.append(f"SMILES Column: {smiles_col}")
            feature_names = self.metadata.get('feature_names', [])
            if feature_names:
                lines.append(f"Generated Descriptors ({len(feature_names)}):")
                for i, name in enumerate(feature_names[:10], 1):
                    lines.append(f"  {i}. {name}")
                if len(feature_names) > 10:
                    lines.append(f"  ... and {len(feature_names) - 10} more")
        elif feature_cols:
            lines.append(f"Feature Type: Numeric Columns")
            lines.append(f"Feature Columns ({len(feature_cols)}):")
            for i, col in enumerate(feature_cols, 1):
                lines.append(f"  {i}. {col}")
        else:
            lines.append("Feature info not available")
        
        if self.metadata.get('task') == 'classification' and self.label_encoder:
            lines.append("\nCLASS LABELS:")
            lines.append("-"*80)
            classes = self.label_encoder.classes_
            for i, cls in enumerate(classes):
                lines.append(f"  {i}: {cls}")
        
        hyperparams = self.metadata.get('hyperparameters')
        if hyperparams:
            lines.append("\nHYPERPARAMETERS:")
            lines.append("-"*80)
            for key, value in hyperparams.items():
                lines.append(f"  {key}: {value}")
        
        best_params = self.metadata.get('best_params')
        if best_params:
            lines.append("\nOPTIMIZED PARAMETERS:")
            lines.append("-"*80)
            lines.append(f"Search Method: {self.metadata.get('search_method', 'Unknown')}")
            for key, value in best_params.items():
                lines.append(f"  {key}: {value}")
        
        cv_scores = self.metadata.get('cv_scores')
        if cv_scores:
            lines.append("\nCROSS-VALIDATION SCORES:")
            lines.append("-"*80)
            lines.append(f"Mean: {np.mean(cv_scores):.4f}")
            lines.append(f"Std: {np.std(cv_scores):.4f}")
            lines.append(f"Scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        lines.append("\nUSAGE:")
        lines.append("-"*80)
        lines.append(f"To make predictions:")
        lines.append(f"  moltrainer -predict -model {Path(model_file).name} -i new_data.csv -o predictions.csv")
        
        lines.append("\n" + "="*80)
        
        report = '\n'.join(lines)
        return report
    
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Make predictions on new data
        
        Args:
            input_file: Path to input CSV file with new data
            output_file: Path to save predictions (required)
            verbose: Enable verbose output
            **kwargs: Additional parameters (model_file required)
        """
        model_file = kwargs.get('model_file')
        if not model_file:
            raise ValueError("Model file must be specified with -model parameter")
        
        if not output_file:
            raise ValueError("Output file must be specified with -o parameter")
        
        # Load model
        if verbose:
            print(f"\n   Loading model...")
        self.load_model(model_file, verbose=verbose)
        
        # Check metadata
        if not self.metadata:
            raise ValueError(
                "Model metadata not found. This model may be from an older version.\n"
                "Please retrain the model with the current version."
            )
        
        # Load new data
        if verbose:
            print(f"   Loading input data...")
        data = self._load_data(input_file)
        original_data = data.copy()
        
        if verbose:
            print(f"   Loaded: {len(data)} samples")
        
        # Prepare features
        X, invalid_indices = self._prepare_features(data, verbose)
        
        # Make predictions
        if verbose:
            print(f"   Making predictions...")
        
        predictions = self.model.predict(X)
        
        # Get probabilities for classification
        probabilities = None
        if self.metadata.get('task') == 'classification' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        
        # Decode predictions if classification
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        # Prepare output
        output_data = original_data.copy()
        
        # Add predictions
        pred_column = f"predicted_{self.metadata.get('target_column', 'target')}"
        # Initialize column with appropriate dtype
        if self.label_encoder:
            # For classification, use object dtype for string labels
            output_data[pred_column] = pd.Series(dtype='object')
        else:
            # For regression, use float dtype
            output_data[pred_column] = pd.Series(dtype='float64')
        output_data.loc[~output_data.index.isin(invalid_indices), pred_column] = predictions
        
        # Add probabilities if available
        if probabilities is not None and self.label_encoder:
            for i, class_name in enumerate(self.label_encoder.classes_):
                prob_column = f"probability_{class_name}"
                output_data[prob_column] = np.nan
                output_data.loc[~output_data.index.isin(invalid_indices), prob_column] = probabilities[:, i]
        
        # Save predictions
        output_path = Path(output_file)
        output_data.to_csv(output_path, index=False)
        
        if verbose:
            print(f"   Predictions saved to: {output_path.name}")
        
        # Generate report
        report_content = self._generate_prediction_report(
            input_file, model_file, len(data), len(invalid_indices),
            self.metadata, output_file
        )
        
        report_path = self.report_manager.save_report(report_content, 'prediction')
        
        # Print to console
        print("\n" + report_content)
        print("\n" + "="*80)
        print(f"Predictions saved to: {output_path.absolute()}")
        print(self.report_manager.get_report_message(report_path))
        print("="*80)
        
        return f"Prediction complete: {len(data) - len(invalid_indices)} samples predicted"
    
    def _prepare_features(self, data, verbose):
        """Prepare features based on model metadata"""
        smiles_column = self.metadata.get('smiles_column')
        feature_columns = self.metadata.get('feature_columns')
        invalid_indices = []
        
        if smiles_column:
            # SMILES-based features
            if smiles_column not in data.columns:
                raise ValueError(
                    f"SMILES column '{smiles_column}' not found in input data.\n"
                    f"Available columns: {', '.join(data.columns)}"
                )
            
            if verbose:
                print(f"   Computing molecular descriptors from '{smiles_column}' column...")
            
            X = self._compute_molecular_descriptors(data[smiles_column])
            
            # Track invalid SMILES
            if hasattr(self, 'invalid_smiles_list') and self.invalid_smiles_list:
                invalid_indices = [idx for idx, _, _ in self.invalid_smiles_list]
                if verbose:
                    print(f"   Warning: {len(invalid_indices)} invalid SMILES found (will be marked as NaN in output)")
            
            # Remove rows with NaN features
            valid_mask = ~np.isnan(X).any(axis=1)
            invalid_from_nan = np.where(~valid_mask)[0].tolist()
            invalid_indices.extend(invalid_from_nan)
            
            X = X[valid_mask]
            
        elif feature_columns:
            # Numeric features
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(
                    f"Feature columns not found in input data: {', '.join(missing_cols)}\n"
                    f"Available columns: {', '.join(data.columns)}"
                )
            
            if verbose:
                print(f"   Using feature columns: {', '.join(feature_columns)}")
            
            X = data[feature_columns].values
            
            # Check for missing values
            if np.isnan(X).any():
                valid_mask = ~np.isnan(X).any(axis=1)
                invalid_indices = np.where(~valid_mask)[0].tolist()
                if verbose:
                    print(f"   Warning: {len(invalid_indices)} rows with missing values (will be marked as NaN in output)")
                X = X[valid_mask]
        else:
            raise ValueError("Model metadata does not specify feature columns or SMILES column")
        
        # Validate feature count
        expected_features = self.metadata.get('n_features')
        if expected_features and X.shape[1] != expected_features:
            raise ValueError(
                f"Feature count mismatch: model expects {expected_features} features, "
                f"but got {X.shape[1]} from input data"
            )
        
        return X, invalid_indices
    
    def _compute_molecular_descriptors(self, smiles_series):
        """Compute molecular descriptors from SMILES (same as training)"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            raise ImportError("RDKit is required for SMILES featurization. Install with: conda install -c conda-forge rdkit")
        
        descriptors_list = []
        self.invalid_smiles_list = []
        
        # Use same descriptors as training
        descriptor_funcs = [
            ('MolWt', Descriptors.MolWt),
            ('LogP', Descriptors.MolLogP),
            ('NumHDonors', Descriptors.NumHDonors),
            ('NumHAcceptors', Descriptors.NumHAcceptors),
            ('TPSA', Descriptors.TPSA),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumAromaticRings', Descriptors.NumAromaticRings),
            ('NumSaturatedRings', Descriptors.NumSaturatedRings),
            ('NumAliphaticRings', Descriptors.NumAliphaticRings),
            ('RingCount', Descriptors.RingCount),
        ]
        
        for idx, smiles in enumerate(smiles_series):
            if pd.isna(smiles):
                descriptors_list.append([np.nan] * len(descriptor_funcs))
                self.invalid_smiles_list.append((idx, smiles, 'Missing SMILES'))
                continue
            
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                descriptors_list.append([np.nan] * len(descriptor_funcs))
                self.invalid_smiles_list.append((idx, str(smiles), 'Invalid SMILES'))
                continue
            
            desc_values = []
            for _, func in descriptor_funcs:
                try:
                    val = func(mol)
                    desc_values.append(val)
                except:
                    desc_values.append(np.nan)
            
            descriptors_list.append(desc_values)
        
        RDLogger.EnableLog('rdApp.*')
        
        return np.array(descriptors_list)
    
    def _generate_prediction_report(self, input_file, model_file, n_samples, n_invalid, metadata, output_file):
        """Generate prediction report"""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("PREDICTION REPORT")
        lines.append("="*80)
        
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append("\n" + "-"*80)
        lines.append("INPUT:")
        lines.append("-"*80)
        lines.append(f"Model: {Path(model_file).name}")
        lines.append(f"Input Data: {Path(input_file).name}")
        lines.append(f"Total Samples: {n_samples}")
        if n_invalid > 0:
            lines.append(f"Invalid/Missing Data: {n_invalid} samples")
            lines.append(f"Valid Predictions: {n_samples - n_invalid} samples")
        
        lines.append("\n" + "-"*80)
        lines.append("MODEL INFO:")
        lines.append("-"*80)
        lines.append(f"Model Type: {metadata.get('model_type', 'Unknown')}")
        lines.append(f"Task: {metadata.get('task', 'Unknown')}")
        lines.append(f"Target: {metadata.get('target_column', 'Unknown')}")
        lines.append(f"Training Date: {metadata.get('training_date', 'Unknown')}")
        
        lines.append("\n" + "-"*80)
        lines.append("OUTPUT:")
        lines.append("-"*80)
        lines.append(f"Predictions saved to: {Path(output_file).name}")
        
        task = metadata.get('task')
        if task == 'classification':
            lines.append(f"Output columns: predicted_{metadata.get('target_column', 'target')}, probability_*")
        else:
            lines.append(f"Output column: predicted_{metadata.get('target_column', 'target')}")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return '\n'.join(lines)

