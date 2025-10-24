"""
Data Visualization Module - Academic-quality plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from moltrainer.core.base import BaseAnalyzer
from moltrainer.utils.report_manager import ReportManager


class DataVisualizer(BaseAnalyzer):
    """Generate academic-quality visualizations for molecular data"""
    
    def __init__(self):
        super().__init__()
        self.report_manager = ReportManager()
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Setup academic publication style following Nature/Science standards"""
        # Set clean style without grid
        sns.set_style("white")
        sns.set_context("paper")
        
        # Use professional color palettes suitable for colorblind readers
        # Nature recommends these colors
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Diverging colormap for correlation (colorblind-friendly)
        self.cmap = "RdBu_r"
        
        # Set matplotlib defaults for publication quality (Nature guidelines)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 600
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['axes.titlesize'] = 10
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 8
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
    
    def analyze(self, input_file, output_file=None, verbose=False, **kwargs):
        """
        Generate visualizations for the dataset
        
        Args:
            input_file: Path to input CSV file
            output_file: Output file path (with extension: .svg, .png, .jpg, .jpeg)
            verbose: Enable verbose output
            **kwargs: Visualization parameters
        """
        if not output_file:
            raise ValueError(
                "Visualization requires output file specification.\n"
                "Use: moltrainer -i input.csv -visualize -o output.png"
            )
        
        # Load data
        data = self._load_data(input_file)
        original_size = len(data)
        
        if verbose:
            print(f"   Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Apply sampling if specified
        sample_size = kwargs.get('sample_size')
        if sample_size:
            data = self._apply_sampling(data, sample_size, verbose)
        
        # Extract parameters
        plot_type = kwargs.get('plot_type', 'all')
        columns = kwargs.get('columns')
        xlabel = kwargs.get('xlabel')
        ylabel = kwargs.get('ylabel')
        title = kwargs.get('title')
        
        # Parse output format from file extension
        output_path = Path(output_file)
        output_format = output_path.suffix.lower().lstrip('.')
        
        if output_format not in ['svg', 'png', 'jpg', 'jpeg']:
            raise ValueError(f"Unsupported format: {output_format}. Use .svg, .png, .jpg, or .jpeg")
        
        # Generate plots
        if verbose:
            print(f"   Generating {plot_type} plot(s)...")
        
        plot_info = self._generate_plots(
            data, output_path, plot_type, output_format,
            columns, xlabel, ylabel, title, verbose
        )
        
        # Generate report
        report = self._generate_visualization_report(data, plot_info, original_size, sample_size)
        report_path = self.report_manager.save_report(report, 'visualization')
        
        # Print report
        print("\n" + report)
        print("\n" + "="*80)
        print(f"Visualization saved to: {output_path.absolute()}")
        print(self.report_manager.get_report_message(report_path))
        print("="*80)
        
        return f"Visualization complete: {len(plot_info)} plot(s) generated"
    
    def _apply_sampling(self, data, sample_size, verbose):
        """Apply sampling to large datasets"""
        if isinstance(sample_size, str) and '%' in sample_size:
            # Percentage sampling
            percentage = float(sample_size.rstrip('%'))
            n_samples = int(len(data) * percentage / 100)
        else:
            # Absolute number sampling
            n_samples = int(sample_size)
        
        if n_samples >= len(data):
            if verbose:
                print(f"   Sample size ({n_samples}) >= dataset size ({len(data)}), using full dataset")
            return data
        
        sampled_data = data.sample(n=n_samples, random_state=42)
        
        if verbose:
            print(f"   Sampled {n_samples} rows from {len(data)} ({n_samples/len(data)*100:.1f}%)")
        
        return sampled_data
    
    def _generate_plots(self, data, output_path, plot_type, output_format,
                       columns, xlabel, ylabel, title, verbose):
        """Generate requested plots"""
        
        plot_info = []
        
        if plot_type == 'all':
            # Generate multiple plot types
            types = ['distribution', 'correlation', 'boxplot']
        else:
            types = [t.strip() for t in plot_type.split(',')]
        
        for ptype in types:
            if ptype == 'distribution':
                info = self._plot_distribution(data, output_path, output_format, 
                                              columns, xlabel, ylabel, title)
                plot_info.append(info)
                
            elif ptype == 'correlation':
                info = self._plot_correlation(data, output_path, output_format, title)
                plot_info.append(info)
                
            elif ptype == 'boxplot':
                info = self._plot_boxplot(data, output_path, output_format,
                                         columns, xlabel, ylabel, title)
                plot_info.append(info)
                
            elif ptype == 'scatter':
                info = self._plot_scatter(data, output_path, output_format,
                                         columns, xlabel, ylabel, title)
                plot_info.append(info)
                
            elif ptype == 'histogram':
                info = self._plot_histogram(data, output_path, output_format,
                                           columns, xlabel, ylabel, title)
                plot_info.append(info)
            
            if verbose:
                print(f"      Created: {info['filename']}")
        
        return plot_info
    
    def _get_output_filename(self, base_path, plot_type, output_format):
        """Generate output filename for multi-plot scenarios"""
        base = base_path.stem
        parent = base_path.parent
        
        if plot_type:
            filename = f"{base}_{plot_type}.{output_format}"
        else:
            filename = f"{base}.{output_format}"
        
        return parent / filename
    
    def _save_figure(self, fig, filepath, output_format):
        """Save figure in specified format with high quality"""
        if output_format in ['png', 'jpg', 'jpeg']:
            fig.savefig(filepath, format=output_format, dpi=600, bbox_inches='tight')
        else:  # svg
            fig.savefig(filepath, format='svg', bbox_inches='tight')
        
        plt.close(fig)
    
    def _plot_distribution(self, data, output_path, output_format, 
                          columns, xlabel, ylabel, title):
        """Plot distribution of numeric features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            cols_to_plot = [c for c in columns if c in numeric_cols]
        else:
            cols_to_plot = numeric_cols[:6]  # Limit to first 6 columns
        
        if not cols_to_plot:
            return {'type': 'distribution', 'status': 'skipped', 'reason': 'No numeric columns'}
        
        n_cols = len(cols_to_plot)
        n_rows = (n_cols + 2) // 3
        n_plot_cols = min(3, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5*n_plot_cols, 4*n_rows))
        if n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_cols > 1 else axes
        
        for i, col in enumerate(cols_to_plot):
            ax = axes[i] if n_cols > 1 else axes
            
            # Plot histogram with KDE (academic style)
            color = self.color_palette[i % len(self.color_palette)]
            sns.histplot(data[col].dropna(), kde=True, ax=ax, 
                        color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(xlabel if xlabel else col)
            ax.set_ylabel(ylabel if ylabel else 'Frequency')
            ax.set_title(col, fontweight='bold', pad=10)
            
            # Remove top and right spines (already done globally)
            # Add subtle grid on y-axis only
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Remove empty subplots
        if n_cols > 1:
            for i in range(n_cols, len(axes)):
                fig.delaxes(axes[i])
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = self._get_output_filename(output_path, 'distribution', output_format)
        self._save_figure(fig, output_file, output_format)
        
        return {
            'type': 'distribution',
            'filename': output_file.name,
            'columns': cols_to_plot,
            'status': 'success'
        }
    
    def _plot_correlation(self, data, output_path, output_format, title):
        """Plot correlation heatmap"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'type': 'correlation', 'status': 'skipped', 'reason': 'Less than 2 numeric columns'}
        
        corr_matrix = data[numeric_cols].corr()
        
        # Calculate appropriate figure size
        n_vars = len(numeric_cols)
        figsize = (max(8, n_vars * 0.6), max(6, n_vars * 0.5))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create academic-style heatmap
        # Use full matrix for better readability in publications
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap=self.cmap, 
                   center=0,
                   vmin=-1, vmax=1,
                   square=True, 
                   linewidths=0.5,
                   linecolor='white',
                   cbar_kws={
                       "shrink": 0.8,
                       "label": "Pearson Correlation Coefficient"
                   },
                   annot_kws={"size": 7},
                   ax=ax)
        
        # Improve labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        ax.set_title(title if title else 'Correlation Matrix', 
                    fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        output_file = self._get_output_filename(output_path, 'correlation', output_format)
        self._save_figure(fig, output_file, output_format)
        
        return {
            'type': 'correlation',
            'filename': output_file.name,
            'columns': numeric_cols,
            'status': 'success'
        }
    
    def _plot_boxplot(self, data, output_path, output_format,
                     columns, xlabel, ylabel, title):
        """Plot boxplots for numeric features"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            cols_to_plot = [c for c in columns if c in numeric_cols]
        else:
            cols_to_plot = numeric_cols[:8]  # Limit to first 8 columns
        
        if not cols_to_plot:
            return {'type': 'boxplot', 'status': 'skipped', 'reason': 'No numeric columns'}
        
        fig, ax = plt.subplots(figsize=(max(10, len(cols_to_plot)*1.5), 6))
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        for col in cols_to_plot:
            plot_data.append(data[col].dropna().values)
            labels.append(col)
        
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Style the boxplot for publication quality
        for i, (patch, color) in enumerate(zip(bp['boxes'], 
                                                self.color_palette * (len(cols_to_plot) // len(self.color_palette) + 1))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(0.8)
        
        # Style whiskers, caps, medians
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        # Style outliers
        plt.setp(bp['fliers'], marker='o', markersize=3, alpha=0.5)
        
        ax.set_xlabel(xlabel if xlabel else 'Features')
        ax.set_ylabel(ylabel if ylabel else 'Values')
        ax.set_title(title if title else 'Distribution Analysis', fontweight='bold', pad=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = self._get_output_filename(output_path, 'boxplot', output_format)
        self._save_figure(fig, output_file, output_format)
        
        return {
            'type': 'boxplot',
            'filename': output_file.name,
            'columns': cols_to_plot,
            'status': 'success'
        }
    
    def _plot_scatter(self, data, output_path, output_format,
                     columns, xlabel, ylabel, title):
        """Plot scatter plot matrix"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns and len(columns) >= 2:
            cols_to_plot = [c for c in columns if c in numeric_cols][:4]
        else:
            cols_to_plot = numeric_cols[:4]  # Limit to first 4 columns
        
        if len(cols_to_plot) < 2:
            return {'type': 'scatter', 'status': 'skipped', 'reason': 'Less than 2 numeric columns'}
        
        # Create pairplot
        plot_data = data[cols_to_plot].dropna()
        
        g = sns.pairplot(plot_data, diag_kind='kde', plot_kws={'alpha': 0.6},
                        palette=self.color_palette)
        
        if title:
            g.fig.suptitle(title, y=1.01, fontsize=16, fontweight='bold')
        
        output_file = self._get_output_filename(output_path, 'scatter', output_format)
        self._save_figure(g.fig, output_file, output_format)
        
        return {
            'type': 'scatter',
            'filename': output_file.name,
            'columns': cols_to_plot,
            'status': 'success'
        }
    
    def _plot_histogram(self, data, output_path, output_format,
                       columns, xlabel, ylabel, title):
        """Plot histograms"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            cols_to_plot = [c for c in columns if c in numeric_cols]
        else:
            cols_to_plot = numeric_cols[:1]  # Single column for detailed histogram
        
        if not cols_to_plot:
            return {'type': 'histogram', 'status': 'skipped', 'reason': 'No numeric columns'}
        
        col = cols_to_plot[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = ax.hist(data[col].dropna(), bins=30, 
                                   color=self.color_palette[0], alpha=0.7, edgecolor='black')
        
        ax.set_xlabel(xlabel if xlabel else col, fontsize=12)
        ax.set_ylabel(ylabel if ylabel else 'Count', fontsize=12)
        ax.set_title(title if title else f'Histogram: {col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = data[col].mean()
        median_val = data[col].median()
        std_val = data[col].std()
        
        textstr = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        output_file = self._get_output_filename(output_path, 'histogram', output_format)
        self._save_figure(fig, output_file, output_format)
        
        return {
            'type': 'histogram',
            'filename': output_file.name,
            'columns': [col],
            'status': 'success'
        }
    
    def _generate_visualization_report(self, data, plot_info, original_size, sample_size):
        """Generate visualization report"""
        from datetime import datetime
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append("VISUALIZATION REPORT")
        lines.append("="*80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\nOriginal Dataset: {original_size} rows")
        
        if sample_size:
            lines.append(f"Visualized Dataset: {data.shape[0]} rows (sampled)")
            lines.append(f"Sampling Rate: {data.shape[0]/original_size*100:.1f}%")
        else:
            lines.append(f"Visualized Dataset: {data.shape[0]} rows (full dataset)")
        
        lines.append(f"Columns: {data.shape[1]}")
        
        lines.append("\n" + "-"*80)
        lines.append("PLOTS GENERATED:")
        lines.append("-"*80)
        
        for i, info in enumerate(plot_info, 1):
            lines.append(f"\n{i}. {info['type'].upper()}")
            lines.append(f"   File: {info['filename']}")
            lines.append(f"   Status: {info['status']}")
            if 'columns' in info:
                lines.append(f"   Columns: {', '.join(info['columns'])}")
            if 'reason' in info:
                lines.append(f"   Reason: {info['reason']}")
        
        lines.append("\n" + "="*80)
        lines.append("Plot Settings:")
        lines.append(f"  Format: High-resolution (DPI 600 for raster formats)")
        lines.append(f"  Style: Nature/Science publication standards")
        lines.append(f"  Font: Arial, 8-10pt")
        lines.append(f"  Color scheme: Colorblind-friendly palette")
        lines.append(f"  Reference: Nature Methods figure guidelines")
        lines.append("="*80)
        
        return "\n".join(lines)

