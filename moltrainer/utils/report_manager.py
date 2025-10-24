"""
Report Manager - Centralized report handling
"""

import os
from datetime import datetime
from pathlib import Path


class ReportManager:
    """Manage automatic report generation and storage"""
    
    def __init__(self, reports_dir='reports'):
        """
        Initialize report manager
        
        Args:
            reports_dir: Directory to store reports (default: 'reports')
        """
        self.reports_dir = Path(reports_dir)
        self._ensure_reports_dir()
    
    def _ensure_reports_dir(self):
        """Create reports directory if it doesn't exist"""
        if not self.reports_dir.exists():
            self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report_filename(self, operation_type):
        """
        Generate report filename with timestamp
        
        Args:
            operation_type: Type of operation (e.g., 'descriptive_stats', 'data_cleaning')
            
        Returns:
            Full path to report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{operation_type}.txt"
        return self.reports_dir / filename
    
    def save_report(self, content, operation_type):
        """
        Save report to file
        
        Args:
            content: Report content as string
            operation_type: Type of operation
            
        Returns:
            Path to saved report file
        """
        report_path = self.generate_report_filename(operation_type)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return report_path
    
    def get_report_message(self, report_path):
        """
        Generate message about report location
        
        Args:
            report_path: Path to report file
            
        Returns:
            Formatted message string
        """
        abs_path = Path(report_path).absolute()
        return f"Report saved to: {abs_path}"

