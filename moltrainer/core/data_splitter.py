"""
Data Splitting Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class DataSplitter(BaseAnalyzer):
    """Split data into train/validation/test sets"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
    
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Split data into train/validation/test sets
        
        Args:
            input_file: Path to input CSV file
            output_file: Not used (auto-generated filenames)
            verbose: Enable verbose output
            **kwargs: Splitting parameters
        """
        # Load data
        data = self._load_data(input_file)
        
        if verbose:
            print(f"   Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Extract parameters
        train_ratio = kwargs.get('train_ratio', 0.7)
        val_ratio = kwargs.get('val_ratio', 0.15)
        test_ratio = kwargs.get('test_ratio', 0.15)
        stratify_column = kwargs.get('stratify_column')
        random_state = kwargs.get('random_state', 42)
        shuffle = kwargs.get('shuffle', True)
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio}\n"
                f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}"
            )
        
        # Perform split
        if verbose:
            print(f"   Splitting data: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
            if stratify_column:
                print(f"   Stratifying by column: {stratify_column}")
        
        train_data, val_data, test_data = self._split_data(
            data, train_ratio, val_ratio, test_ratio,
            stratify_column, random_state, shuffle, verbose
        )
        
        # Generate output filenames
        input_path = Path(input_file)
        base_name = input_path.stem
        output_dir = input_path.parent
        
        train_file = output_dir / f"{base_name}_train.csv"
        val_file = output_dir / f"{base_name}_val.csv"
        test_file = output_dir / f"{base_name}_test.csv"
        
        # Save splits
        train_data.to_csv(train_file, index=False)
        val_data.to_csv(val_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        if verbose:
            print(f"   Saved train set: {train_file.name}")
            print(f"   Saved val set: {val_file.name}")
            print(f"   Saved test set: {test_file.name}")
        
        # Generate report
        report = self._generate_split_report(
            data, train_data, val_data, test_data,
            train_ratio, val_ratio, test_ratio,
            stratify_column, train_file, val_file, test_file
        )
        
        report_path = self.report_manager.save_report(report, 'data_split')
        
        # Print report
        print("\n" + report)
        print("\n" + "="*80)
        print(f"Train set: {train_file.absolute()}")
        print(f"Val set:   {val_file.absolute()}")
        print(f"Test set:  {test_file.absolute()}")
        print(self.report_manager.get_report_message(report_path))
        print("="*80)
        
        return f"Data split complete: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    
    def _split_data(self, data, train_ratio, val_ratio, test_ratio,
                   stratify_column, random_state, shuffle, verbose):
        """Perform stratified or random split"""
        
        # Prepare stratification
        stratify_array = None
        if stratify_column:
            if stratify_column not in data.columns:
                raise ValueError(f"Stratify column '{stratify_column}' not found in data")
            stratify_array = data[stratify_column]
            
            # Check if column is suitable for stratification
            unique_values = stratify_array.nunique()
            if unique_values > len(data) * 0.5:
                print(f"   Warning: Stratify column has many unique values ({unique_values}). "
                      f"Consider using a categorical column.")
        
        # Handle edge cases: if test_ratio is 0
        if test_ratio <= 0:
            train_val_data = data
            test_data = pd.DataFrame(columns=data.columns)
        else:
            # First split: separate test set
            train_val_data, test_data = train_test_split(
                data,
                test_size=test_ratio,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_array if stratify_column else None
            )
        
        # Handle edge cases: if val_ratio is 0
        if val_ratio <= 0:
            train_data = train_val_data
            val_data = pd.DataFrame(columns=data.columns)
        else:
            # Second split: separate train and validation from train_val
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            
            if stratify_column and len(train_val_data) > 0:
                stratify_train_val = train_val_data[stratify_column]
            else:
                stratify_train_val = None
            
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_ratio_adjusted,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify_train_val
            )
        
        return train_data, val_data, test_data
    
    def _generate_split_report(self, original_data, train_data, val_data, test_data,
                               train_ratio, val_ratio, test_ratio, stratify_column,
                               train_file, val_file, test_file):
        """Generate splitting report"""
        from datetime import datetime
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("DATA SPLITTING REPORT")
        lines.append("="*80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append(f"\nOriginal Dataset: {len(original_data)} rows x {original_data.shape[1]} columns")
        
        lines.append("\n" + "-"*80)
        lines.append("SPLIT CONFIGURATION:")
        lines.append("-"*80)
        lines.append(f"Train ratio: {train_ratio:.2%} ({train_ratio:.2f})")
        lines.append(f"Validation ratio: {val_ratio:.2%} ({val_ratio:.2f})")
        lines.append(f"Test ratio: {test_ratio:.2%} ({test_ratio:.2f})")
        
        if stratify_column:
            lines.append(f"Stratification: {stratify_column}")
        else:
            lines.append("Stratification: None (random split)")
        
        lines.append(f"Random seed: 42")
        lines.append(f"Shuffle: Yes")
        
        lines.append("\n" + "-"*80)
        lines.append("RESULTING SETS:")
        lines.append("-"*80)
        
        lines.append(f"\nTraining Set:")
        lines.append(f"  Rows: {len(train_data)} ({len(train_data)/len(original_data):.2%})")
        lines.append(f"  File: {train_file.name}")
        
        lines.append(f"\nValidation Set:")
        lines.append(f"  Rows: {len(val_data)} ({len(val_data)/len(original_data):.2%})")
        lines.append(f"  File: {val_file.name}")
        
        lines.append(f"\nTest Set:")
        lines.append(f"  Rows: {len(test_data)} ({len(test_data)/len(original_data):.2%})")
        lines.append(f"  File: {test_file.name}")
        
        # If stratified, show class distribution
        if stratify_column and stratify_column in original_data.columns:
            lines.append("\n" + "-"*80)
            lines.append("CLASS DISTRIBUTION (if stratified):")
            lines.append("-"*80)
            
            for set_name, set_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
                dist = set_data[stratify_column].value_counts(normalize=True).sort_index()
                lines.append(f"\n{set_name} Set:")
                for cls, prop in dist.items():
                    count = set_data[stratify_column].value_counts()[cls]
                    lines.append(f"  {cls}: {count} ({prop:.2%})")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)

