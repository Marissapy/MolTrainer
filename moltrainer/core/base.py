"""
Base classes for analysis modules
"""

from abc import ABC, abstractmethod


class BaseAnalyzer(ABC):
    """Base class for all analysis modules"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, input_file, output_file=None, verbose=False):
        """
        Perform analysis on input data
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output file (optional)
            verbose: Enable verbose output
            
        Returns:
            Result summary string
        """
        pass
    
    def _load_data(self, input_file):
        """Load data from CSV file"""
        import pandas as pd
        try:
            data = pd.read_csv(input_file)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load data from {input_file}: {str(e)}")
    
    def _save_results(self, results, output_file):
        """Save results to file"""
        if output_file:
            with open(output_file, 'w') as f:
                f.write(results)

