"""
Model Training Module - Phase 1: Random Forest
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            mean_squared_error, mean_absolute_error, r2_score,
                            confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class ModelTrainer(BaseAnalyzer):
    """Train machine learning models on molecular data"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
        self.model = None
        self.task = None
        self.feature_names = None
        self.label_encoder = None
        self.invalid_smiles_list = []
        self.training_log = []
        
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Train a machine learning model
        
        Args:
            input_file: Path to training data CSV
            output_file: Not used (auto-generated)
            verbose: Enable verbose output
            **kwargs: Training parameters
        """
        # Extract parameters
        model_type = kwargs.get('model_type', 'rf')
        target_column = kwargs.get('target_column')
        smiles_column = kwargs.get('smiles_column')
        feature_columns = kwargs.get('feature_columns')
        val_file = kwargs.get('val_file')
        test_file = kwargs.get('test_file')
        task = kwargs.get('task', 'auto')
        cv_folds = kwargs.get('cv_folds', 5)
        no_cv = kwargs.get('no_cv', False)
        
        # Data splitting options
        auto_split_mode = kwargs.get('auto_split_mode', '3way')
        train_split_ratio = kwargs.get('train_split_ratio', None)
        
        # Hyperparameter search options
        search_method = kwargs.get('search_method', 'none')
        search_depth = kwargs.get('search_depth', 'shallow')
        search_iterations = kwargs.get('search_iterations', 10)
        search_cv_folds = kwargs.get('search_cv_folds', 3)
        search_timeout = kwargs.get('search_timeout', None)
        
        # Model hyperparameters
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', None)
        random_state = kwargs.get('random_state', 42)
        
        # Validate inputs
        if not target_column:
            raise ValueError("Target column must be specified with -target")
        
        if not smiles_column and (not feature_columns or len(feature_columns) == 0):
            raise ValueError(
                "Must specify either:\n"
                "  -smiles for SMILES featurization, or\n"
                "  -features for numeric features\n"
                f"Got: smiles={smiles_column}, features={feature_columns}"
            )
        
        # Load data
        if verbose:
            print("   Loading training data...")
        train_data = self._load_data(input_file)
        
        if verbose:
            print(f"   Loaded: {train_data.shape[0]} rows, {train_data.shape[1]} columns")
        
        # Prepare features and target
        X_train, y_train = self._prepare_data(
            train_data, target_column, smiles_column, feature_columns, verbose
        )
        
        # Show warnings for invalid SMILES
        if self.invalid_smiles_list:
            print(f"\n   WARNING: {len(self.invalid_smiles_list)} invalid/missing SMILES detected:")
            # Show first 5 examples
            for idx, smiles, reason in self.invalid_smiles_list[:5]:
                print(f"     Row {idx}: '{smiles}' - {reason}")
            if len(self.invalid_smiles_list) > 5:
                print(f"     ... and {len(self.invalid_smiles_list) - 5} more")
            print(f"   RECOMMENDATION: Use -clean -validate_smiles to filter invalid SMILES first")
            print()
        
        # Auto-detect task if needed
        if task == 'auto':
            self.task = self._detect_task(y_train, verbose)
        else:
            self.task = task
        
        if verbose:
            print(f"   Task detected: {self.task}")
            print(f"   Features: {X_train.shape[1]} columns")
            print(f"   Training samples: {len(X_train)}")
        
        # Auto-split flag
        auto_split = False
        split_info = None
        
        # Load validation data if provided, otherwise auto-split
        X_val, y_val = None, None
        if val_file:
            if verbose:
                print(f"   Loading validation data...")
            val_data = self._load_data(val_file)
            X_val, y_val = self._prepare_data(
                val_data, target_column, smiles_column, feature_columns, verbose
            )
            if verbose:
                print(f"   Validation samples: {len(X_val)}")
        
        # Load test data if provided
        X_test, y_test = None, None
        if test_file:
            if verbose:
                print(f"   Loading test data...")
            test_data = self._load_data(test_file)
            X_test, y_test = self._prepare_data(
                test_data, target_column, smiles_column, feature_columns, verbose
            )
            if verbose:
                print(f"   Test samples: {len(X_test)}")
        
        # Auto-split if no validation or test set provided (and mode is not 'none')
        if auto_split_mode != 'none' and (val_file is None or test_file is None):
            if verbose:
                mode_name = "3-way (train/val/test)" if auto_split_mode == '3way' else "2-way (train/test)"
                print(f"\n   Auto-splitting data using {mode_name} mode...")
            
            from sklearn.model_selection import train_test_split
            
            # Determine stratification
            stratify_y = y_train if self.task == 'classification' and len(np.unique(y_train)) < 20 else None
            
            # Determine split ratios based on mode
            if auto_split_mode == '2way':
                # Two-way split: train/test only
                if train_split_ratio is None:
                    train_ratio = 0.8
                else:
                    train_ratio = train_split_ratio
                val_ratio = 0.0
                test_ratio = 1.0 - train_ratio if test_file is None else 0.0
            else:  # '3way'
                # Three-way split: train/val/test
                if train_split_ratio is None:
                    train_ratio = 0.7
                else:
                    train_ratio = train_split_ratio
                val_ratio = (1.0 - train_ratio) / 2 if val_file is None else 0.0
                test_ratio = (1.0 - train_ratio) / 2 if test_file is None else 0.0
            
            if val_ratio + test_ratio > 0:
                # First split: train vs (val+test)
                X_train_new, X_temp, y_train_new, y_temp = train_test_split(
                    X_train, y_train,
                    test_size=(val_ratio + test_ratio),
                    random_state=42,
                    stratify=stratify_y
                )
                
                # Second split: val vs test
                if val_ratio > 0 and test_ratio > 0:
                    stratify_temp = y_temp if self.task == 'classification' and len(np.unique(y_temp)) < 20 else None
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp,
                        test_size=test_ratio / (val_ratio + test_ratio),
                        random_state=42,
                        stratify=stratify_temp
                    )
                elif val_ratio > 0:
                    X_val, y_val = X_temp, y_temp
                else:
                    X_test, y_test = X_temp, y_temp
                
                # Update training data
                original_size = len(X_train)
                X_train, y_train = X_train_new, y_train_new
                
                auto_split = True
                split_info = {
                    'original_size': original_size,
                    'train_size': len(X_train),
                    'val_size': len(X_val) if X_val is not None else 0,
                    'test_size': len(X_test) if X_test is not None else 0,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'stratified': stratify_y is not None,
                    'mode': auto_split_mode
                }
                
                if verbose:
                    print(f"   Split: {len(X_train)} train / {len(X_val) if X_val is not None else 0} val / {len(X_test) if X_test is not None else 0} test")
                    if stratify_y is not None:
                        print(f"   Stratified by target variable")
        
        # Handle 'all' model type - compare all models
        if model_type == 'all':
            return self._compare_all_models(
                input_file, X_train, y_train, X_val, y_val, X_test, y_test,
                output_file, target_column, smiles_column, feature_columns,
                search_method, search_iterations, search_cv_folds, search_timeout,
                cv_folds, no_cv, random_state, auto_split, split_info, verbose
            )
        
        # Build model
        if verbose:
            print(f"\n   Building {model_type.upper()} model...")
        
        # Hyperparameter search or direct training
        if search_method != 'none':
            if verbose:
                print(f"   Performing {search_method} hyperparameter search...")
            
            import time
            start_time = time.time()
            self.model, best_params = self._hyperparameter_search(
                model_type, self.task, X_train, y_train,
                search_method, search_depth, search_iterations, search_cv_folds, search_timeout,
                n_estimators, max_depth, random_state, verbose
            )
            training_time = time.time() - start_time
            
            if verbose:
                print(f"   Best parameters: {best_params}")
                print(f"   Search completed in {training_time:.2f} seconds")
        else:
            self.model = self._build_model(
                model_type, self.task, n_estimators, max_depth, random_state
            )
            
            # Train model
            if verbose:
                print("   Training model...")
            
            import time
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            best_params = None
        
        if verbose:
            print(f"   Training completed in {training_time:.2f} seconds")
        
        # Cross-validation
        cv_scores = None
        if not no_cv and cv_folds > 0:
            if verbose:
                print(f"   Performing {cv_folds}-fold cross-validation...")
            cv_scores = self._cross_validate(X_train, y_train, cv_folds, verbose)
        
        # Evaluate on all sets
        results = {}
        results['train'] = self._evaluate(X_train, y_train, 'Training')
        
        if X_val is not None:
            results['val'] = self._evaluate(X_val, y_val, 'Validation')
        
        if X_test is not None:
            results['test'] = self._evaluate(X_test, y_test, 'Test')
        
        # Prepare output directory
        input_path = Path(input_file)
        if output_file:
            output_dir = Path(output_file)
            output_dir.mkdir(parents=True, exist_ok=True)
            base_name = input_path.stem
        else:
            output_dir = input_path.parent
            base_name = input_path.stem
        
        # Save model with comprehensive metadata
        model_file = output_dir / f"{base_name}_model.pkl"
        
        model_package = {
            'model': self.model,
            'metadata': {
                'model_type': model_type,
                'task': self.task,
                'target_column': target_column,
                'smiles_column': smiles_column,
                'feature_columns': feature_columns,
                'feature_names': self.feature_names,
                'n_features': X_train.shape[1],
                'n_samples_train': len(X_train),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                },
                'search_method': search_method if search_method != 'none' else None,
                'best_params': best_params if best_params else None,
                'cv_scores': cv_scores if cv_scores else None,
            },
            'label_encoder': self.label_encoder,
            'version': '0.1.0'
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        if verbose:
            print(f"\n   Model saved to: {model_file.name}")
        
        # Generate visualizations
        plot_files, data_files = self._generate_plots(
            output_dir, base_name, X_train, y_train, X_test, y_test, results
        )
        
        # Generate training log
        log_file = output_dir / f"{base_name}_training_log.txt"
        self._save_training_log(log_file, input_file, model_type, self.task, target_column,
                               n_estimators, max_depth, training_time, results, cv_scores, 
                               auto_split, split_info)
        
        # Generate report
        report = self._generate_training_report(
            input_file, model_type, self.task, target_column,
            n_estimators, max_depth, training_time,
            results, cv_scores, model_file, plot_files, data_files,
            auto_split, split_info
        )
        
        report_path = self.report_manager.save_report(report, 'training')
        
        # Print report
        print("\n" + report)
        print("\n" + "="*80)
        print(f"Model saved to: {model_file.absolute()}")
        print(f"Training log saved to: {log_file.absolute()}")
        for plot_file in plot_files:
            print(f"Plot saved to: {plot_file.absolute()}")
        for data_file in data_files:
            print(f"Data saved to: {data_file.absolute()}")
        print(self.report_manager.get_report_message(report_path))
        print("="*80)
        
        test_metric = results.get('test', results.get('val', results['train']))
        metric_name = 'accuracy' if self.task == 'classification' else 'r2'
        metric_value = test_metric.get(metric_name, 0)
        
        return f"Training complete. {metric_name.upper()}: {metric_value:.4f}"
    
    def _prepare_data(self, data, target_column, smiles_column, feature_columns, verbose):
        """Prepare features and target from data"""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get target values and encode if necessary
        y = data[target_column].values
        
        # Encode categorical target for classification
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.astype(float)
        
        # Prepare features
        if smiles_column:
            if smiles_column not in data.columns:
                raise ValueError(f"SMILES column '{smiles_column}' not found in data")
            
            if verbose:
                print(f"   Computing molecular descriptors from {smiles_column}...")
            
            X = self._compute_molecular_descriptors(data[smiles_column])
            
        else:
            # Use specified feature columns
            if not all(col in data.columns for col in feature_columns):
                missing = [col for col in feature_columns if col not in data.columns]
                raise ValueError(f"Feature columns not found: {missing}")
            
            X = data[feature_columns].values
            self.feature_names = feature_columns
        
        # Remove rows with NaN in features or target
        # Convert X to DataFrame temporarily for consistent NaN handling
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        # Check for NaN/invalid values
        valid_mask = X_df.notna().all(axis=1) & y_series.notna()
        
        # Apply mask
        X = X[valid_mask.values]
        y = y[valid_mask.values]
        
        if verbose and (~valid_mask).any():
            removed = (~valid_mask).sum()
            print(f"   Removed {removed} rows with missing/invalid values")
        
        return X, y
    
    def _compute_molecular_descriptors(self, smiles_series):
        """Compute molecular descriptors from SMILES using RDKit"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit import RDLogger
            # Suppress RDKit warnings temporarily
            RDLogger.DisableLog('rdApp.*')
        except ImportError:
            raise ImportError("RDKit is required for SMILES featurization. Install with: conda install -c conda-forge rdkit")
        
        descriptors_list = []
        descriptor_names = []
        invalid_smiles = []
        
        # Define descriptors to compute
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
        
        descriptor_names = [name for name, _ in descriptor_funcs]
        self.feature_names = descriptor_names
        
        for idx, smiles in enumerate(smiles_series):
            if pd.isna(smiles):
                descriptors_list.append([np.nan] * len(descriptor_funcs))
                invalid_smiles.append((idx, smiles, 'Missing SMILES'))
                continue
            
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                descriptors_list.append([np.nan] * len(descriptor_funcs))
                invalid_smiles.append((idx, str(smiles), 'Invalid SMILES'))
                continue
            
            desc_values = []
            for _, func in descriptor_funcs:
                try:
                    val = func(mol)
                    desc_values.append(val)
                except:
                    desc_values.append(np.nan)
            
            descriptors_list.append(desc_values)
        
        # Store invalid SMILES for reporting
        if not hasattr(self, 'invalid_smiles_list'):
            self.invalid_smiles_list = []
        self.invalid_smiles_list.extend(invalid_smiles)
        
        # Re-enable RDKit logging
        RDLogger.EnableLog('rdApp.*')
        
        return np.array(descriptors_list)
    
    def _detect_task(self, y, verbose):
        """Automatically detect task type"""
        unique_values = len(np.unique(y))
        
        if unique_values < 20:
            task = 'classification'
        else:
            task = 'regression'
        
        if verbose:
            print(f"   Unique target values: {unique_values}")
        
        return task
    
    def _build_model(self, model_type, task, n_estimators=100, max_depth=None, random_state=42):
        """Build model based on type and task"""
        
        if model_type == 'rf':
            # Random Forest
            if task == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
        
        elif model_type == 'svm':
            # Support Vector Machine
            from sklearn.svm import SVC, SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            if task == 'classification':
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel='rbf', random_state=random_state))
                ])
            else:
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVR(kernel='rbf'))
                ])
        
        elif model_type == 'xgb':
            # XGBoost
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
            
            if task == 'classification':
                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 6,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else 6,
                    random_state=random_state,
                    n_jobs=-1
                )
        
        elif model_type == 'lgb':
            # LightGBM
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
            
            if task == 'classification':
                return lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else -1,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                return lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth if max_depth else -1,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1
                )
        
        elif model_type == 'lr':
            # Logistic/Linear Regression
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            if task == 'classification':
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(random_state=random_state, max_iter=1000))
                ])
            else:
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('ridge', Ridge(random_state=random_state))
                ])
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _hyperparameter_search(self, model_type, task, X, y, search_method, search_depth,
                               n_iter, cv_folds, timeout, n_estimators, max_depth, random_state, verbose):
        """Perform hyperparameter search with optional timeout"""
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        # Get base model
        base_model = self._build_model(model_type, task, n_estimators, max_depth, random_state)
        
        # Define parameter grids for each model type (use deep if specified)
        deep = (search_depth == 'deep')
        param_grids = self._get_param_grid(model_type, task, deep=deep)
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grids,
                cv=cv_folds,
                scoring='accuracy' if task == 'classification' else 'r2',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )
        else:  # random
            search = RandomizedSearchCV(
                base_model,
                param_grids,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='accuracy' if task == 'classification' else 'r2',
                n_jobs=-1,
                random_state=random_state,
                verbose=1 if verbose else 0
            )
        
        # Perform search with timeout if specified
        if timeout:
            import signal
            import time
            
            class TimeoutException(Exception):
                pass
            
            def timeout_handler(signum, frame):
                raise TimeoutException("Search timeout")
            
            try:
                # Set timeout (Unix/Linux)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                search.fit(X, y)
                signal.alarm(0)
            except (TimeoutException, AttributeError):
                # AttributeError on Windows (no SIGALRM)
                # Use threading-based timeout
                import threading
                result = [None]
                exception = [None]
                
                def search_thread():
                    try:
                        search.fit(X, y)
                        result[0] = True
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=search_thread)
                thread.daemon = True
                thread.start()
                thread.join(timeout)
                
                if thread.is_alive():
                    if verbose:
                        print(f"   Search timeout ({timeout}s) reached, using best found so far")
                elif exception[0]:
                    raise exception[0]
        else:
            search.fit(X, y)
        
        return search.best_estimator_, search.best_params_
    
    def _get_param_grid(self, model_type, task, deep=True):
        """Get parameter grid for hyperparameter search
        
        Args:
            model_type: Model type
            task: Task type (classification/regression)
            deep: If True, use extended parameter ranges for deeper search
        """
        
        if model_type == 'rf':
            if deep:
                return {
                    'n_estimators': [50, 100, 200, 300, 500, 800, 1000],
                    'max_depth': [None, 5, 10, 15, 20, 30, 40, 50],
                    'min_samples_split': [2, 4, 6, 8, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 3, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
                    'bootstrap': [True, False]
                }
            else:
                return {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        
        elif model_type == 'svm':
            if deep:
                return {
                    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
                    'svm__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
                    'svm__kernel': ['rbf', 'poly', 'sigmoid'],
                    'svm__degree': [2, 3, 4, 5]  # for poly kernel
                }
            else:
                return {
                    'svm__C': [0.1, 1, 10, 100],
                    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'svm__kernel': ['rbf', 'poly']
                }
        
        elif model_type == 'xgb':
            if deep:
                return {
                    'n_estimators': [50, 100, 200, 300, 500, 800],
                    'max_depth': [3, 4, 5, 6, 7, 8, 10, 12, 15],
                    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5, 7, 10],
                    'gamma': [0, 0.1, 0.2, 0.5, 1, 2],
                    'reg_alpha': [0, 0.01, 0.1, 1, 10],
                    'reg_lambda': [0, 0.01, 0.1, 1, 10, 100]
                }
            else:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
        
        elif model_type == 'lgb':
            if deep:
                return {
                    'n_estimators': [50, 100, 200, 300, 500, 800],
                    'max_depth': [3, 5, 7, 10, 15, 20, -1],
                    'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
                    'num_leaves': [15, 31, 50, 63, 100, 127, 200, 255],
                    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_child_samples': [10, 20, 30, 50, 100],
                    'reg_alpha': [0, 0.01, 0.1, 1, 10],
                    'reg_lambda': [0, 0.01, 0.1, 1, 10, 100]
                }
            else:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'num_leaves': [15, 31, 63, 127],
                    'subsample': [0.6, 0.8, 1.0]
                }
        
        elif model_type == 'lr':
            if task == 'classification':
                if deep:
                    return {
                        'lr__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000],
                        'lr__penalty': ['l1', 'l2', 'elasticnet'],
                        'lr__solver': ['lbfgs', 'saga', 'liblinear'],
                        'lr__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # for elasticnet
                    }
                else:
                    return {
                        'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
                        'lr__penalty': ['l2'],
                        'lr__solver': ['lbfgs', 'saga']
                    }
            else:
                if deep:
                    return {
                        'ridge__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000]
                    }
                else:
                    return {
                        'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                    }
        
        else:
            return {}
    
    def _compare_all_models(self, input_file, X_train, y_train, X_val, y_val, X_test, y_test,
                            output_file, target_column, smiles_column, feature_columns,
                            search_method, search_iterations, search_cv_folds, search_timeout,
                            cv_folds, no_cv, random_state, auto_split, split_info, verbose):
        """Compare all available models and select the best one"""
        
        model_types = ['rf', 'svm', 'xgb', 'lgb', 'lr']
        results_summary = []
        
        if verbose:
            print(f"\n   Comparing {len(model_types)} model types...")
            print(f"   Models: {', '.join([m.upper() for m in model_types])}\n")
        
        import time
        
        for model_type in model_types:
            try:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"   Training {model_type.upper()}...")
                    print(f"{'='*60}")
                
                start_time = time.time()
                
                # Build and train model
                if search_method != 'none':
                    model, best_params = self._hyperparameter_search(
                        model_type, self.task, X_train, y_train,
                        search_method, 'shallow', search_iterations, search_cv_folds, search_timeout,
                        100, None, random_state, verbose
                    )
                else:
                    model = self._build_model(model_type, self.task, 100, None, random_state)
                    model.fit(X_train, y_train)
                    best_params = None
                
                training_time = time.time() - start_time
                
                # Evaluate
                train_results = self._evaluate_model(model, X_train, y_train)
                val_results = self._evaluate_model(model, X_val, y_val) if X_val is not None else None
                test_results = self._evaluate_model(model, X_test, y_test) if X_test is not None else None
                
                # Get primary metric
                if self.task == 'classification':
                    metric_name = 'accuracy'
                    test_score = test_results['accuracy'] if test_results else (
                        val_results['accuracy'] if val_results else train_results['accuracy']
                    )
                else:
                    metric_name = 'r2'
                    test_score = test_results['r2'] if test_results else (
                        val_results['r2'] if val_results else train_results['r2']
                    )
                
                results_summary.append({
                    'model_type': model_type,
                    'model': model,
                    'best_params': best_params,
                    'training_time': training_time,
                    'train_results': train_results,
                    'val_results': val_results,
                    'test_results': test_results,
                    'test_score': test_score,
                    'metric_name': metric_name
                })
                
                if verbose:
                    print(f"   {model_type.upper()} {metric_name}: {test_score:.4f} (time: {training_time:.2f}s)")
            
            except Exception as e:
                if verbose:
                    print(f"   {model_type.upper()} failed: {str(e)}")
                continue
        
        if not results_summary:
            raise RuntimeError("All models failed to train")
        
        # Select best model
        results_summary.sort(key=lambda x: x['test_score'], reverse=True)
        best_result = results_summary[0]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"   BEST MODEL: {best_result['model_type'].upper()}")
            print(f"   {best_result['metric_name'].upper()}: {best_result['test_score']:.4f}")
            print(f"{'='*60}\n")
        
        # Set the best model
        self.model = best_result['model']
        model_type = best_result['model_type']
        
        # Generate comparison report and plots...
        # (使用best_result继续生成报告)
        
        return f"Model comparison complete. Best: {model_type.upper()} ({best_result['metric_name']}: {best_result['test_score']:.4f})"
    
    def _evaluate_model(self, model, X, y):
        """Helper method to evaluate a model"""
        if X is None or y is None:
            return None
        
        y_pred = model.predict(X)
        results = {'predictions': y_pred}
        
        if self.task == 'classification':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            results['accuracy'] = accuracy_score(y, y_pred)
            results['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            results['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            results['mae'] = mean_absolute_error(y, y_pred)
            results['r2'] = r2_score(y, y_pred)
        
        return results
    
    def _cross_validate(self, X, y, cv_folds, verbose):
        """Perform cross-validation"""
        if self.task == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        if verbose:
            print(f"   CV {scoring}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return scores
    
    def _evaluate(self, X, y, set_name):
        """Evaluate model on a dataset"""
        y_pred = self.model.predict(X)
        
        results = {'predictions': y_pred}
        
        if self.task == 'classification':
            results['accuracy'] = accuracy_score(y, y_pred)
            results['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            results['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            results['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            results['confusion_matrix'] = confusion_matrix(y, y_pred)
        else:
            results['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            results['mae'] = mean_absolute_error(y, y_pred)
            results['r2'] = r2_score(y, y_pred)
        
        return results
    
    def _generate_plots(self, output_dir, base_name, X_train, y_train, X_test, y_test, results):
        """Generate visualization plots in SVG and PNG formats with corresponding data files"""
        plot_files = []
        data_files = []
        
        # Feature importance plot
        if hasattr(self.model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            feature_names = self.feature_names if self.feature_names else \
                           [f"Feature_{i}" for i in range(len(importances))]
            
            ax.barh(range(len(indices)), importances[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 Feature Importances', fontweight='bold')
            ax.invert_yaxis()
            
            plt.tight_layout()
            
            # Save as PNG and SVG
            plot_png = output_dir / f"{base_name}_feature_importance.png"
            plot_svg = output_dir / f"{base_name}_feature_importance.svg"
            fig.savefig(plot_png, dpi=600, bbox_inches='tight')
            fig.savefig(plot_svg, format='svg', bbox_inches='tight')
            plt.close(fig)
            
            plot_files.extend([plot_png, plot_svg])
            
            # Save data
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            data_file = output_dir / f"{base_name}_feature_importance_data.csv"
            importance_df.to_csv(data_file, index=False)
            data_files.append(data_file)
        
        # Confusion matrix (classification only)
        if self.task == 'classification' and X_test is not None and 'test' in results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            cm = results['test']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix (Test Set)', fontweight='bold')
            
            plt.tight_layout()
            
            # Save as PNG and SVG
            plot_png = output_dir / f"{base_name}_confusion_matrix.png"
            plot_svg = output_dir / f"{base_name}_confusion_matrix.svg"
            fig.savefig(plot_png, dpi=600, bbox_inches='tight')
            fig.savefig(plot_svg, format='svg', bbox_inches='tight')
            plt.close(fig)
            
            plot_files.extend([plot_png, plot_svg])
            
            # Save data
            cm_df = pd.DataFrame(cm)
            data_file = output_dir / f"{base_name}_confusion_matrix_data.csv"
            cm_df.to_csv(data_file)
            data_files.append(data_file)
        
        # Prediction scatter plot (regression only)
        if self.task == 'regression' and X_test is not None and 'test' in results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            y_pred = results['test']['predictions']
            
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Prediction vs Actual (Test Set)', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save as PNG and SVG
            plot_png = output_dir / f"{base_name}_predictions.png"
            plot_svg = output_dir / f"{base_name}_predictions.svg"
            fig.savefig(plot_png, dpi=600, bbox_inches='tight')
            fig.savefig(plot_svg, format='svg', bbox_inches='tight')
            plt.close(fig)
            
            plot_files.extend([plot_png, plot_svg])
            
            # Save data
            pred_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'residual': y_test - y_pred
            })
            data_file = output_dir / f"{base_name}_predictions_data.csv"
            pred_df.to_csv(data_file, index=False)
            data_files.append(data_file)
        
        return plot_files, data_files
    
    def _save_training_log(self, log_file, input_file, model_type, task, target_column,
                           n_estimators, max_depth, training_time, results, cv_scores,
                           auto_split=False, split_info=None):
        """Save detailed training log"""
        from datetime import datetime
        
        lines = []
        lines.append("="*80)
        lines.append(f"MOLTRAINER TRAINING LOG")
        lines.append("="*80)
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Training Data: {input_file}")
        lines.append(f"Model Type: {model_type.upper()}")
        lines.append(f"Task: {task.capitalize()}")
        lines.append(f"Target Column: {target_column}")
        lines.append("")
        
        if auto_split and split_info:
            lines.append("AUTO DATA SPLITTING:")
            lines.append(f"  Original samples: {split_info['original_size']}")
            lines.append(f"  Train: {split_info['train_size']} ({split_info['train_ratio']*100:.1f}%)")
            lines.append(f"  Validation: {split_info['val_size']} ({split_info['val_ratio']*100:.1f}%)")
            lines.append(f"  Test: {split_info['test_size']} ({split_info['test_ratio']*100:.1f}%)")
            lines.append(f"  Stratified: {'Yes' if split_info['stratified'] else 'No'}")
            lines.append(f"  Random seed: 42")
            lines.append("")
        
        lines.append("HYPERPARAMETERS:")
        lines.append(f"  n_estimators: {n_estimators}")
        lines.append(f"  max_depth: {max_depth if max_depth else 'None (unlimited)'}")
        lines.append(f"  random_state: 42")
        lines.append("")
        
        lines.append("FEATURE INFORMATION:")
        lines.append(f"  Number of features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        if self.feature_names:
            lines.append(f"  Feature names: {', '.join(self.feature_names[:10])}")
            if len(self.feature_names) > 10:
                lines.append(f"    ... and {len(self.feature_names) - 10} more")
        lines.append("")
        
        if self.invalid_smiles_list:
            lines.append("INVALID SMILES WARNING:")
            lines.append(f"  Total invalid/missing SMILES: {len(self.invalid_smiles_list)}")
            lines.append("  First 10 examples:")
            for idx, smiles, reason in self.invalid_smiles_list[:10]:
                lines.append(f"    Row {idx}: '{smiles}' - {reason}")
            if len(self.invalid_smiles_list) > 10:
                lines.append(f"    ... and {len(self.invalid_smiles_list) - 10} more")
            lines.append("")
        
        lines.append("TRAINING METRICS:")
        lines.append(f"  Training time: {training_time:.4f} seconds")
        lines.append("")
        
        if cv_scores is not None:
            metric_name = 'Accuracy' if task == 'classification' else 'R²'
            lines.append("CROSS-VALIDATION RESULTS:")
            lines.append(f"  Number of folds: {len(cv_scores)}")
            lines.append(f"  {metric_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            lines.append(f"  Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
            lines.append("")
        
        lines.append("PERFORMANCE METRICS:")
        for set_name, set_results in results.items():
            lines.append(f"\n  {set_name.capitalize()} Set:")
            if task == 'classification':
                lines.append(f"    Accuracy:  {set_results['accuracy']:.4f}")
                lines.append(f"    Precision: {set_results['precision']:.4f}")
                lines.append(f"    Recall:    {set_results['recall']:.4f}")
                lines.append(f"    F1 Score:  {set_results['f1']:.4f}")
            else:
                lines.append(f"    RMSE: {set_results['rmse']:.4f}")
                lines.append(f"    MAE:  {set_results['mae']:.4f}")
                lines.append(f"    R²:   {set_results['r2']:.4f}")
        
        lines.append("\n" + "="*80)
        lines.append("END OF TRAINING LOG")
        lines.append("="*80)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_training_report(self, input_file, model_type, task, target_column,
                                  n_estimators, max_depth, training_time,
                                  results, cv_scores, model_file, plot_files, data_files,
                                  auto_split=False, split_info=None):
        """Generate training report"""
        from datetime import datetime
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("MODEL TRAINING REPORT")
        lines.append("="*80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append("\n" + "-"*80)
        lines.append("CONFIGURATION:")
        lines.append("-"*80)
        lines.append(f"Training Data: {input_file}")
        lines.append(f"Model Type: {model_type.upper()}")
        lines.append(f"Task: {task.capitalize()}")
        lines.append(f"Target Column: {target_column}")
        lines.append(f"Number of Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        
        if auto_split and split_info:
            lines.append("\n" + "-"*80)
            lines.append("DATA SPLITTING (AUTO):")
            lines.append("-"*80)
            lines.append(f"Original samples: {split_info['original_size']}")
            lines.append(f"Train: {split_info['train_size']} samples ({split_info['train_ratio']*100:.1f}%)")
            lines.append(f"Validation: {split_info['val_size']} samples ({split_info['val_ratio']*100:.1f}%)")
            lines.append(f"Test: {split_info['test_size']} samples ({split_info['test_ratio']*100:.1f}%)")
            lines.append(f"Stratified: {'Yes' if split_info['stratified'] else 'No'}")
            lines.append(f"Note: Data was automatically split as validation/test sets were not provided.")
        
        lines.append("\n" + "-"*80)
        lines.append("HYPERPARAMETERS:")
        lines.append("-"*80)
        lines.append(f"n_estimators: {n_estimators}")
        lines.append(f"max_depth: {max_depth if max_depth else 'None (unlimited)'}")
        lines.append(f"Training Time: {training_time:.2f} seconds")
        
        if cv_scores is not None:
            lines.append("\n" + "-"*80)
            lines.append("CROSS-VALIDATION:")
            lines.append("-"*80)
            metric_name = 'Accuracy' if task == 'classification' else 'R²'
            lines.append(f"Folds: {len(cv_scores)}")
            lines.append(f"{metric_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            lines.append(f"Individual scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        lines.append("\n" + "-"*80)
        lines.append("PERFORMANCE METRICS:")
        lines.append("-"*80)
        
        for set_name, set_results in results.items():
            lines.append(f"\n{set_name.capitalize()} Set:")
            
            if task == 'classification':
                lines.append(f"  Accuracy:  {set_results['accuracy']:.4f}")
                lines.append(f"  Precision: {set_results['precision']:.4f}")
                lines.append(f"  Recall:    {set_results['recall']:.4f}")
                lines.append(f"  F1 Score:  {set_results['f1']:.4f}")
            else:
                lines.append(f"  RMSE: {set_results['rmse']:.4f}")
                lines.append(f"  MAE:  {set_results['mae']:.4f}")
                lines.append(f"  R²:   {set_results['r2']:.4f}")
        
        if self.feature_names and hasattr(self.model, 'feature_importances_'):
            lines.append("\n" + "-"*80)
            lines.append("TOP 10 FEATURE IMPORTANCES:")
            lines.append("-"*80)
            
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            for i, idx in enumerate(indices, 1):
                lines.append(f"{i:2d}. {self.feature_names[idx]:30s} {importances[idx]:.4f}")
        
        if self.invalid_smiles_list:
            lines.append("\n" + "-"*80)
            lines.append("WARNINGS:")
            lines.append("-"*80)
            lines.append(f"Invalid/missing SMILES: {len(self.invalid_smiles_list)}")
            lines.append("These rows were skipped during training.")
            lines.append("Recommendation: Use -clean -validate_smiles to filter data first.")
        
        lines.append("\n" + "-"*80)
        lines.append("OUTPUT FILES:")
        lines.append("-"*80)
        lines.append(f"Model: {model_file.name}")
        for plot_file in plot_files:
            lines.append(f"Plot:  {plot_file.name}")
        for data_file in data_files:
            lines.append(f"Data:  {data_file.name}")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)

