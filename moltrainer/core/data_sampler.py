"""
Data Sampling Module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class DataSampler(BaseAnalyzer):
    """Sample data from dataset with various strategies"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
    
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Sample data from dataset
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save sampled data (required)
            verbose: Enable verbose output
            **kwargs: Sampling parameters
        """
        # Check output file
        if not output_file:
            raise ValueError("Output file is required for data sampling. Use: moltrainer -sample -i input.csv -o output.csv")
        
        # Load data
        data = self._load_data(input_file)
        original_shape = data.shape
        
        if verbose:
            print(f"   Loaded data: {original_shape[0]} rows, {original_shape[1]} columns")
        
        # Extract sampling parameters
        sample_size = kwargs.get('sample_size')
        sample_method = kwargs.get('sample_method', 'random')
        stratify_column = kwargs.get('stratify_column')
        replace = kwargs.get('replace', False)
        random_state = kwargs.get('random_state', 42)
        
        # Perform sampling
        if verbose:
            print(f"   Sampling method: {sample_method}")
        
        sampled_data = self._perform_sampling(
            data, sample_size, sample_method, stratify_column, replace, random_state, verbose
        )
        
        # Save sampled data
        output_path = Path(output_file)
        sampled_data.to_csv(output_path, index=False)
        
        if verbose:
            print(f"   Saved sampled data: {output_path.name}")
        
        # Generate report
        report_content = self._generate_sampling_report(
            input_file, output_file, original_shape, sampled_data.shape,
            sample_size, sample_method, stratify_column, replace, random_state
        )
        
        report_path = self.report_manager.save_report(report_content, 'data_sampling')
        
        # Print to console
        print("\n" + report_content)
        print("\n" + "="*80)
        print(f"Sampled data saved to: {output_path.absolute()}")
        print(self.report_manager.get_report_message(report_path))
        print("="*80)
        
        return f"Sampling complete: {original_shape[0]} -> {sampled_data.shape[0]} rows ({sampled_data.shape[0]/original_shape[0]*100:.1f}%)"
    
    def _perform_sampling(self, data, sample_size, method, stratify_column, replace, random_state, verbose):
        """Perform sampling based on specified method"""
        
        # Parse sample size
        n_samples = self._parse_sample_size(data, sample_size)
        
        if n_samples >= len(data) and not replace:
            if verbose:
                print(f"   Warning: Sample size ({n_samples}) >= dataset size ({len(data)})")
                print(f"   Returning entire dataset")
            return data
        
        # Sampling methods
        if method == 'random':
            sampled_data = self._random_sampling(data, n_samples, stratify_column, replace, random_state, verbose)
        elif method == 'systematic':
            sampled_data = self._systematic_sampling(data, n_samples, verbose)
        elif method == 'stratified':
            if not stratify_column:
                raise ValueError("Stratified sampling requires -stratify COLUMN parameter")
            sampled_data = self._stratified_sampling(data, n_samples, stratify_column, replace, random_state, verbose)
        elif method == 'head':
            sampled_data = data.head(n_samples)
            if verbose:
                print(f"   Selected first {n_samples} rows")
        elif method == 'tail':
            sampled_data = data.tail(n_samples)
            if verbose:
                print(f"   Selected last {n_samples} rows")
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return sampled_data
    
    def _parse_sample_size(self, data, sample_size):
        """Parse sample size (absolute number or percentage)"""
        if sample_size is None:
            raise ValueError("Sample size is required. Use -sample_size N or -sample_size 50%")
        
        # Check if percentage
        if isinstance(sample_size, str) and '%' in sample_size:
            percentage = float(sample_size.rstrip('%'))
            if not 0 < percentage <= 100:
                raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")
            n_samples = int(len(data) * percentage / 100)
        else:
            n_samples = int(sample_size)
            if n_samples <= 0:
                raise ValueError(f"Sample size must be positive, got {n_samples}")
        
        return n_samples
    
    def _random_sampling(self, data, n_samples, stratify_column, replace, random_state, verbose):
        """Random sampling"""
        if stratify_column:
            if stratify_column not in data.columns:
                raise ValueError(f"Stratify column '{stratify_column}' not found in data")
            
            if verbose:
                print(f"   Performing stratified random sampling on '{stratify_column}'")
            
            # Check if we have enough samples per stratum
            value_counts = data[stratify_column].value_counts()
            min_count = value_counts.min()
            
            if not replace and n_samples > len(data):
                n_samples = len(data)
                if verbose:
                    print(f"   Adjusted sample size to {n_samples} (max without replacement)")
            
            # Use sklearn for stratified sampling if possible
            try:
                from sklearn.model_selection import train_test_split
                sampled_data, _ = train_test_split(
                    data,
                    train_size=n_samples,
                    stratify=data[stratify_column],
                    random_state=random_state
                )
            except ValueError as e:
                # Fall back to proportional sampling
                if verbose:
                    print(f"   Warning: {e}")
                    print(f"   Using proportional sampling instead")
                sampled_data = data.groupby(stratify_column, group_keys=False, as_index=False).apply(
                    lambda x: x.sample(n=max(1, int(n_samples * len(x) / len(data))), 
                                     replace=replace, random_state=random_state),
                    include_groups=False
                ).reset_index(drop=True)
        else:
            if verbose:
                print(f"   Performing random sampling: {n_samples} rows")
            sampled_data = data.sample(n=n_samples, replace=replace, random_state=random_state)
        
        return sampled_data
    
    def _systematic_sampling(self, data, n_samples, verbose):
        """Systematic sampling (every k-th row)"""
        if n_samples >= len(data):
            return data
        
        k = len(data) // n_samples
        if k < 1:
            k = 1
        
        indices = np.arange(0, len(data), k)[:n_samples]
        sampled_data = data.iloc[indices].reset_index(drop=True)
        
        if verbose:
            print(f"   Systematic sampling: every {k}-th row, selected {len(sampled_data)} rows")
        
        return sampled_data
    
    def _stratified_sampling(self, data, n_samples, stratify_column, replace, random_state, verbose):
        """Stratified sampling"""
        if stratify_column not in data.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found in data")
        
        if verbose:
            print(f"   Performing stratified sampling on '{stratify_column}'")
        
        # Calculate proportional sample sizes for each stratum
        value_counts = data[stratify_column].value_counts()
        if verbose:
            print(f"   Original distribution:")
            for val, count in value_counts.items():
                print(f"     {val}: {count} ({count/len(data)*100:.1f}%)")
        
        # Sample proportionally from each stratum
        sampled_data = data.groupby(stratify_column, group_keys=False, as_index=False).apply(
            lambda x: x.sample(n=max(1, int(n_samples * len(x) / len(data))),
                             replace=replace, random_state=random_state),
            include_groups=False
        ).reset_index(drop=True)
        
        # Adjust to exact sample size if needed
        if len(sampled_data) < n_samples and not replace:
            # Add more samples randomly
            remaining = n_samples - len(sampled_data)
            additional = data[~data.index.isin(sampled_data.index)].sample(n=remaining, random_state=random_state)
            sampled_data = pd.concat([sampled_data, additional]).reset_index(drop=True)
        elif len(sampled_data) > n_samples:
            # Remove excess samples randomly
            sampled_data = sampled_data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        
        if verbose:
            print(f"   Sampled distribution:")
            sampled_counts = sampled_data[stratify_column].value_counts()
            for val, count in sampled_counts.items():
                print(f"     {val}: {count} ({count/len(sampled_data)*100:.1f}%)")
        
        return sampled_data
    
    def _generate_sampling_report(self, input_file, output_file, original_shape, sampled_shape,
                                  sample_size, method, stratify_column, replace, random_state):
        """Generate sampling report"""
        from datetime import datetime
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("DATA SAMPLING REPORT")
        lines.append("="*80)
        
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        lines.append("\n" + "-"*80)
        lines.append("INPUT:")
        lines.append("-"*80)
        lines.append(f"Input File: {Path(input_file).name}")
        lines.append(f"Original Size: {original_shape[0]} rows x {original_shape[1]} columns")
        
        lines.append("\n" + "-"*80)
        lines.append("SAMPLING CONFIGURATION:")
        lines.append("-"*80)
        lines.append(f"Sampling Method: {method}")
        lines.append(f"Sample Size: {sample_size}")
        lines.append(f"Sampling Rate: {sampled_shape[0]/original_shape[0]*100:.2f}%")
        if stratify_column:
            lines.append(f"Stratify Column: {stratify_column}")
        lines.append(f"With Replacement: {'Yes' if replace else 'No'}")
        lines.append(f"Random Seed: {random_state}")
        
        lines.append("\n" + "-"*80)
        lines.append("OUTPUT:")
        lines.append("-"*80)
        lines.append(f"Output File: {Path(output_file).name}")
        lines.append(f"Sampled Size: {sampled_shape[0]} rows x {sampled_shape[1]} columns")
        lines.append(f"Rows Sampled: {sampled_shape[0]}")
        lines.append(f"Rows Removed: {original_shape[0] - sampled_shape[0]}")
        
        lines.append("\n" + "="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return '\n'.join(lines)

