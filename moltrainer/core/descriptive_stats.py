"""
Descriptive Statistics Module
"""

import pandas as pd
import numpy as np
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class DescriptiveStatsAnalyzer(BaseAnalyzer):
    """Perform descriptive statistics analysis on molecular data"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
    
    def analyze(self, input_file, output_file=None, verbose=False):
        """
        Compute descriptive statistics for the dataset
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output file (optional)
            verbose: Enable verbose output
            
        Returns:
            Summary string of the analysis
        """
        # Load data
        data = self._load_data(input_file)
        
        if verbose:
            print(f"   Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Compute statistics
        results = self._compute_statistics(data, verbose)
        
        # Always save report automatically
        report_path = self.report_manager.save_report(results, 'descriptive_stats')
        report_message = self.report_manager.get_report_message(report_path)
        
        # Print to console
        print("\n" + results)
        print("\n" + "="*80)
        print(report_message)
        print("="*80)
        
        summary = f"Statistics computed for {data.shape[0]} samples, {data.shape[1]} features"
        
        return summary
    
    def _compute_statistics(self, data, verbose=False):
        """Compute detailed statistics"""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("DESCRIPTIVE STATISTICS REPORT")
        lines.append("=" * 80 + "\n")
        
        # Basic info
        lines.append(f"Dataset Shape: {data.shape[0]} rows x {data.shape[1]} columns")
        lines.append(f"Columns: {', '.join(data.columns.tolist())}\n")
        
        # Memory usage
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        lines.append(f"Memory Usage: {memory_mb:.2f} MB\n")
        
        # Data types
        lines.append("Data Types:")
        for col, dtype in data.dtypes.items():
            lines.append(f"  {col}: {dtype}")
        lines.append("")
        
        # Missing values
        lines.append("Missing Values:")
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        for col in data.columns:
            if missing[col] > 0:
                lines.append(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        if missing.sum() == 0:
            lines.append("  No missing values detected")
        lines.append("")
        
        # Numeric columns statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            lines.append("Numeric Features Statistics:")
            lines.append("-" * 80)
            
            stats_df = data[numeric_cols].describe()
            lines.append(stats_df.to_string())
            lines.append("")
            
            # Additional statistics
            lines.append("Additional Statistics:")
            for col in numeric_cols:
                skew = data[col].skew()
                kurt = data[col].kurtosis()
                lines.append(f"  {col}:")
                lines.append(f"    Skewness: {skew:.4f}")
                lines.append(f"    Kurtosis: {kurt:.4f}")
                lines.append(f"    Unique values: {data[col].nunique()}")
            lines.append("")
        
        # Categorical columns
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            lines.append("Categorical Features:")
            lines.append("-" * 80)
            for col in cat_cols:
                lines.append(f"  {col}:")
                lines.append(f"    Unique values: {data[col].nunique()}")
                top_values = data[col].value_counts().head(5)
                lines.append(f"    Top 5 values:")
                for val, count in top_values.items():
                    lines.append(f"      {val}: {count} ({count/len(data)*100:.2f}%)")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

