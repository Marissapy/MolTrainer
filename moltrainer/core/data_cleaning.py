"""
Data Cleaning Module
"""

import pandas as pd
import numpy as np
from datetime import datetime
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class DataCleaner(BaseAnalyzer):
    """Comprehensive data cleaning for molecular datasets"""
    
    def __init__(self):
        super().__init__()
        self.cleaning_report = []
        self.original_shape = None
        self.final_shape = None
        self.report_manager = ReportManager()
        
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Main entry point for data cleaning
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save cleaned data (REQUIRED for data cleaning)
            verbose: Enable verbose output
            **kwargs: Cleaning parameters
        """
        # Require output file for data cleaning
        if not output_file:
            raise ValueError(
                "Data cleaning requires output file specification.\n"
                "Use: moltrainer -i input.csv -clean [options] -o output.csv"
            )
        
        # Load data
        data = self._load_data(input_file)
        self.original_shape = data.shape
        
        if verbose:
            print(f"   Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Check if interactive mode (no specific cleaning options provided)
        cleaning_options = self._extract_cleaning_options(kwargs)
        
        if not cleaning_options:
            # Interactive mode
            if verbose:
                print("   Entering interactive mode...")
            data, selected_operations = self._interactive_clean(data, verbose)
        else:
            # Command mode
            data = self._execute_cleaning(data, cleaning_options, verbose)
            selected_operations = cleaning_options
        
        self.final_shape = data.shape
        
        # Generate report
        report = self._generate_report(selected_operations)
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        
        # Save report automatically to reports directory
        report_path = self.report_manager.save_report(report, 'data_cleaning')
        report_message = self.report_manager.get_report_message(report_path)
        
        if verbose:
            print(f"   Cleaned data saved to: {output_file}")
        
        # Print report to console
        print("\n" + report)
        
        # Print file locations
        print("\n" + "="*80)
        print(f"Cleaned data saved to: {output_file}")
        print(report_message)
        print("="*80)
        
        summary = f"Cleaned data saved. Rows: {self.original_shape[0]} -> {self.final_shape[0]}"
        
        return summary
    
    def _extract_cleaning_options(self, kwargs):
        """Extract cleaning options from kwargs"""
        options = {}
        
        if kwargs.get('remove_duplicates'):
            options['remove_duplicates'] = {
                'subset': kwargs.get('duplicate_subset')
            }
        
        if kwargs.get('handle_missing'):
            options['handle_missing'] = {
                'method': kwargs.get('missing_method', 'drop'),
                'fill_value': kwargs.get('fill_value'),
                'fill_method': kwargs.get('fill_method', 'mean')
            }
        
        if kwargs.get('remove_outliers'):
            options['remove_outliers'] = {
                'method': kwargs.get('outlier_method', 'iqr'),
                'threshold': kwargs.get('outlier_threshold', 3),
                'columns': kwargs.get('outlier_columns')
            }
        
        if kwargs.get('validate_smiles'):
            options['validate_smiles'] = {
                'column': kwargs.get('smiles_column', 'smiles')
            }
        
        if kwargs.get('filter_value'):
            options['filter_value'] = kwargs.get('filter_value')
        
        if kwargs.get('drop_columns'):
            options['drop_columns'] = {
                'columns': kwargs.get('columns_to_drop')
            }
        
        return options
    
    def _interactive_clean(self, data, verbose):
        """Interactive mode for data cleaning"""
        print("\n" + "="*80)
        print("INTERACTIVE DATA CLEANING MODE")
        print("="*80)
        print(f"\nCurrent dataset: {data.shape[0]} rows x {data.shape[1]} columns")
        print(f"Columns: {', '.join(data.columns.tolist())}")
        
        # Show data preview
        print("\nData Preview (first 5 rows):")
        print(data.head(5).to_string())
        
        # Display menu
        print("\n" + "="*80)
        print("CLEANING OPTIONS:")
        print("="*80)
        print("0. Default Pipeline (duplicates + missing + outliers)")
        print("1. Remove duplicate rows")
        print("2. Handle missing values")
        print("3. Remove outliers")
        print("4. Validate SMILES")
        print("5. Filter by value conditions (>, <, ==, !=, >=, <=)")
        print("6. Drop columns")
        print("7. Custom pipeline (select multiple)")
        print("8. Skip cleaning (exit)")
        print("="*80)
        
        choice = input("\nSelect cleaning option (0-8): ").strip()
        
        selected_operations = {}
        
        if choice == '0':
            # Default pipeline
            print("\nApplying default cleaning pipeline...")
            selected_operations = self._configure_default_pipeline(data)
            
        elif choice == '1':
            selected_operations['remove_duplicates'] = self._configure_duplicates(data)
            
        elif choice == '2':
            selected_operations['handle_missing'] = self._configure_missing(data)
            
        elif choice == '3':
            selected_operations['remove_outliers'] = self._configure_outliers(data)
            
        elif choice == '4':
            selected_operations['validate_smiles'] = self._configure_smiles(data)
            
        elif choice == '5':
            selected_operations['filter_range'] = self._configure_range_filter(data)
            
        elif choice == '6':
            selected_operations['drop_columns'] = self._configure_drop_columns(data)
            
        elif choice == '7':
            selected_operations = self._configure_custom_pipeline(data)
            
        elif choice == '8':
            print("\nSkipping data cleaning.")
            return data, {}
        
        else:
            print("\nInvalid choice. Skipping data cleaning.")
            return data, {}
        
        # Execute cleaning
        data = self._execute_cleaning(data, selected_operations, verbose)
        
        return data, selected_operations
    
    def _configure_default_pipeline(self, data):
        """Configure default cleaning pipeline"""
        operations = {
            'remove_duplicates': {'subset': None},
            'handle_missing': {'method': 'drop'},
            'remove_outliers': {'method': 'iqr', 'threshold': 1.5, 'columns': None}
        }
        return operations
    
    def _configure_duplicates(self, data):
        """Configure duplicate removal"""
        print("\nDuplicate Removal Configuration:")
        subset = input("Enter column name to check duplicates (or press Enter for all columns): ").strip()
        return {'subset': subset if subset else None}
    
    def _configure_missing(self, data):
        """Configure missing value handling"""
        print("\nMissing Value Configuration:")
        print("1. Drop rows with missing values")
        print("2. Fill with mean (numeric columns)")
        print("3. Fill with median (numeric columns)")
        print("4. Fill with specific value")
        
        choice = input("Select method (1-4): ").strip()
        
        if choice == '1':
            return {'method': 'drop'}
        elif choice == '2':
            return {'method': 'fill', 'fill_method': 'mean'}
        elif choice == '3':
            return {'method': 'fill', 'fill_method': 'median'}
        elif choice == '4':
            value = input("Enter fill value: ").strip()
            return {'method': 'fill', 'fill_value': value}
        else:
            return {'method': 'drop'}
    
    def _configure_outliers(self, data):
        """Configure outlier removal"""
        print("\nOutlier Removal Configuration:")
        print("1. IQR method (recommended)")
        print("2. Z-score method")
        
        choice = input("Select method (1-2): ").strip()
        method = 'iqr' if choice == '1' else 'zscore'
        
        threshold = input(f"Enter threshold (default: {'1.5' if method == 'iqr' else '3'}): ").strip()
        threshold = float(threshold) if threshold else (1.5 if method == 'iqr' else 3)
        
        columns = input("Enter column names (comma-separated, or press Enter for all numeric): ").strip()
        columns = [c.strip() for c in columns.split(',')] if columns else None
        
        return {'method': method, 'threshold': threshold, 'columns': columns}
    
    def _configure_smiles(self, data):
        """Configure SMILES validation"""
        print("\nSMILES Validation Configuration:")
        col = input(f"Enter SMILES column name (default: 'smiles'): ").strip()
        return {'column': col if col else 'smiles'}
    
    def _configure_range_filter(self, data):
        """Configure value filtering with operators"""
        print("\nValue Filter Configuration:")
        print(f"Numeric columns: {', '.join(data.select_dtypes(include=[np.number]).columns.tolist())}")
        print("\nSupported operators: ==, !=, >, >=, <, <=")
        print("Examples:")
        print("  ic50 > 0        (remove rows where ic50 <= 0)")
        print("  ic50 != 0       (remove rows where ic50 == 0)")
        print("  activity == 1   (keep only rows where activity is 1)")
        
        filters = []
        while True:
            filter_expr = input("\nEnter filter (column operator value), or press Enter to finish: ").strip()
            if not filter_expr:
                break
            filters.append(filter_expr)
        
        return filters
    
    def _configure_drop_columns(self, data):
        """Configure column dropping"""
        print("\nDrop Columns Configuration:")
        print(f"Available columns: {', '.join(data.columns.tolist())}")
        
        cols = input("Enter column names to drop (comma-separated): ").strip()
        return {'columns': [c.strip() for c in cols.split(',')]}
    
    def _configure_custom_pipeline(self, data):
        """Configure custom pipeline"""
        print("\nCustom Pipeline Configuration:")
        print("Enter operation numbers separated by commas (e.g., 1,2,3)")
        print("1. Remove duplicates")
        print("2. Handle missing values")
        print("3. Remove outliers")
        print("4. Validate SMILES")
        print("5. Filter by range")
        
        choices = input("Enter operations: ").strip().split(',')
        operations = {}
        
        for choice in choices:
            choice = choice.strip()
            if choice == '1':
                operations['remove_duplicates'] = self._configure_duplicates(data)
            elif choice == '2':
                operations['handle_missing'] = self._configure_missing(data)
            elif choice == '3':
                operations['remove_outliers'] = self._configure_outliers(data)
            elif choice == '4':
                operations['validate_smiles'] = self._configure_smiles(data)
            elif choice == '5':
                operations['filter_value'] = self._configure_range_filter(data)
        
        return operations
    
    def _execute_cleaning(self, data, operations, verbose):
        """Execute cleaning operations"""
        for op_name, op_config in operations.items():
            if verbose:
                print(f"   Executing: {op_name}...")
            
            before_rows = len(data)
            
            if op_name == 'remove_duplicates':
                data = self._remove_duplicates(data, op_config)
            elif op_name == 'handle_missing':
                data = self._handle_missing(data, op_config)
            elif op_name == 'remove_outliers':
                data = self._remove_outliers(data, op_config)
            elif op_name == 'validate_smiles':
                data = self._validate_smiles(data, op_config)
            elif op_name == 'filter_value':
                data = self._filter_value(data, op_config)
            elif op_name == 'drop_columns':
                data = self._drop_columns(data, op_config)
            
            after_rows = len(data)
            removed = before_rows - after_rows
            
            self.cleaning_report.append({
                'operation': op_name,
                'config': op_config,
                'rows_before': before_rows,
                'rows_after': after_rows,
                'rows_removed': removed
            })
            
            if verbose and removed > 0:
                print(f"      Removed {removed} rows")
        
        return data
    
    def _remove_duplicates(self, data, config):
        """Remove duplicate rows"""
        subset = config.get('subset')
        subset = [subset] if subset and subset in data.columns else None
        return data.drop_duplicates(subset=subset, keep='first')
    
    def _handle_missing(self, data, config):
        """Handle missing values"""
        method = config.get('method', 'drop')
        
        if method == 'drop':
            return data.dropna()
        elif method == 'fill':
            fill_method = config.get('fill_method')
            fill_value = config.get('fill_value')
            
            if fill_value is not None:
                return data.fillna(fill_value)
            elif fill_method == 'mean':
                return data.fillna(data.mean(numeric_only=True))
            elif fill_method == 'median':
                return data.fillna(data.median(numeric_only=True))
        
        return data
    
    def _remove_outliers(self, data, config):
        """Remove outliers using IQR or Z-score method"""
        method = config.get('method', 'iqr')
        threshold = config.get('threshold', 1.5 if method == 'iqr' else 3)
        columns = config.get('columns')
        
        # Select numeric columns
        if columns:
            numeric_cols = [c for c in columns if c in data.columns and pd.api.types.is_numeric_dtype(data[c])]
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return data
        
        mask = pd.Series([True] * len(data), index=data.index)
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask &= (data[col] >= lower) & (data[col] <= upper)
            
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                mask &= z_scores <= threshold
        
        return data[mask]
    
    def _validate_smiles(self, data, config):
        """Validate SMILES strings using RDKit"""
        try:
            from rdkit import Chem
        except ImportError:
            print("   Warning: RDKit not installed. Skipping SMILES validation.")
            print("   Install with: conda install -c conda-forge rdkit")
            return data
        
        smiles_col = config.get('column', 'smiles')
        
        if smiles_col not in data.columns:
            print(f"   Warning: Column '{smiles_col}' not found. Skipping SMILES validation.")
            return data
        
        def is_valid_smiles(smiles):
            if pd.isna(smiles):
                return False
            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None
        
        valid_mask = data[smiles_col].apply(is_valid_smiles)
        return data[valid_mask]
    
    def _filter_value(self, data, filters):
        """Filter data by value conditions with operators"""
        if not filters:
            return data
        
        # Support both list and single string
        if isinstance(filters, str):
            filters = [filters]
        
        mask = pd.Series([True] * len(data), index=data.index)
        
        for filter_expr in filters:
            # Parse filter expression: "column operator value"
            parsed = self._parse_filter_expression(filter_expr)
            if not parsed:
                print(f"   Warning: Invalid filter expression '{filter_expr}'. Skipping.")
                continue
            
            column, operator, value = parsed
            
            if column not in data.columns:
                print(f"   Warning: Column '{column}' not found. Skipping filter.")
                continue
            
            # Apply operator
            try:
                if operator == '==':
                    mask &= data[column] == value
                elif operator == '!=':
                    mask &= data[column] != value
                elif operator == '>':
                    mask &= data[column] > value
                elif operator == '>=':
                    mask &= data[column] >= value
                elif operator == '<':
                    mask &= data[column] < value
                elif operator == '<=':
                    mask &= data[column] <= value
            except Exception as e:
                print(f"   Warning: Error applying filter '{filter_expr}': {str(e)}. Skipping.")
                continue
        
        return data[mask]
    
    def _parse_filter_expression(self, expr):
        """Parse filter expression like 'column > 0' or 'column != 0'"""
        import re
        
        # Try to match: column operator value
        # Support operators: ==, !=, >=, <=, >, <
        pattern = r'^\s*(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$'
        match = re.match(pattern, expr)
        
        if not match:
            return None
        
        column = match.group(1)
        operator = match.group(2)
        value_str = match.group(3)
        
        # Try to convert value to numeric
        try:
            value = float(value_str)
        except ValueError:
            # Keep as string for string comparison
            value = value_str
        
        return (column, operator, value)
    
    def _drop_columns(self, data, config):
        """Drop specified columns"""
        columns = config.get('columns', [])
        columns_to_drop = [c for c in columns if c in data.columns]
        return data.drop(columns=columns_to_drop)
    
    def _generate_report(self, operations):
        """Generate cleaning report"""
        lines = []
        lines.append("\n" + "="*80)
        lines.append("DATA CLEANING REPORT")
        lines.append("="*80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\nOriginal Dataset: {self.original_shape[0]} rows x {self.original_shape[1]} columns")
        lines.append(f"Final Dataset: {self.final_shape[0]} rows x {self.final_shape[1]} columns")
        lines.append(f"Total Rows Removed: {self.original_shape[0] - self.final_shape[0]}")
        lines.append(f"Retention Rate: {self.final_shape[0] / self.original_shape[0] * 100:.2f}%")
        
        lines.append("\n" + "-"*80)
        lines.append("OPERATIONS PERFORMED:")
        lines.append("-"*80)
        
        for i, entry in enumerate(self.cleaning_report, 1):
            lines.append(f"\n{i}. {entry['operation'].upper().replace('_', ' ')}")
            lines.append(f"   Configuration: {entry['config']}")
            lines.append(f"   Rows before: {entry['rows_before']}")
            lines.append(f"   Rows after: {entry['rows_after']}")
            lines.append(f"   Rows removed: {entry['rows_removed']}")
            lines.append(f"   Removal rate: {entry['rows_removed'] / entry['rows_before'] * 100:.2f}%")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)

