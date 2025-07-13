"""
exporter.py: Export datasets and reports for balens
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Union, Optional, Tuple
from datetime import datetime


class DataExporter:
    """Export balanced datasets and generate reports."""
    
    def __init__(self, output_dir: str = "balens_output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_output_dir()
    
    def _create_output_dir(self):
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def export_balanced_data(self, 
                           X_train: Union[pd.DataFrame, np.ndarray],
                           y_train: Union[pd.Series, np.ndarray],
                           X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                           y_test: Optional[Union[pd.Series, np.ndarray]] = None,
                           feature_names: Optional[list] = None,
                           target_name: str = "target",
                           prefix: str = "balanced") -> Dict:
        """
        Export balanced train/test datasets to CSV files.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            feature_names: Names for features (auto-generated if None)
            target_name: Name for target column
            prefix: Prefix for output files
            
        Returns:
            Dictionary with file paths and export info
        """
        # Convert to DataFrame if needed
        if isinstance(X_train, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
        else:
            X_train_df = X_train.copy()
            feature_names = X_train_df.columns.tolist()
        
        if isinstance(y_train, np.ndarray):
            y_train_df = pd.Series(y_train, name=target_name)
        else:
            y_train_df = y_train.copy()
            y_train_df.name = target_name
        
        # Create training dataset
        train_df = pd.concat([X_train_df, y_train_df], axis=1)
        
        # Export training data
        train_filename = f"{prefix}_train_{self.timestamp}.csv"
        train_path = os.path.join(self.output_dir, train_filename)
        train_df.to_csv(train_path, index=False)
        
        export_info = {
            "train_file": train_path,
            "train_shape": train_df.shape,
            "feature_names": feature_names,
            "target_name": target_name
        }
        
        # Export test data if provided
        if X_test is not None and y_test is not None:
            if isinstance(X_test, np.ndarray):
                X_test_df = pd.DataFrame(X_test, columns=feature_names)
            else:
                X_test_df = X_test.copy()
            
            if isinstance(y_test, np.ndarray):
                y_test_df = pd.Series(y_test, name=target_name)
            else:
                y_test_df = y_test.copy()
                y_test_df.name = target_name
            
            test_df = pd.concat([X_test_df, y_test_df], axis=1)
            
            test_filename = f"{prefix}_test_{self.timestamp}.csv"
            test_path = os.path.join(self.output_dir, test_filename)
            test_df.to_csv(test_path, index=False)
            
            export_info.update({
                "test_file": test_path,
                "test_shape": test_df.shape
            })
        
        return export_info
    
    def generate_summary_report(self, 
                              imbalance_info: Dict,
                              resampling_info: Optional[Dict] = None,
                              binning_info: Optional[Dict] = None,
                              weights_info: Optional[Dict] = None,
                              export_info: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            imbalance_info: Information from imbalance detection
            resampling_info: Information from resampling (optional)
            binning_info: Information from binning (optional)
            weights_info: Information from class weights (optional)
            export_info: Information from data export (optional)
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("BALENS - DATA BALANCING REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Imbalance Detection Section
        report_lines.append("1. IMBALANCE DETECTION")
        report_lines.append("-" * 30)
        report_lines.append(f"Target Column: {imbalance_info.get('target_column', 'N/A')}")
        report_lines.append(f"Total Samples: {imbalance_info.get('total_samples', 'N/A')}")
        report_lines.append(f"Number of Classes: {imbalance_info.get('n_classes', 'N/A')}")
        report_lines.append(f"Imbalance Ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.4f}")
        report_lines.append(f"Severity: {imbalance_info.get('severity', 'N/A').upper()}")
        report_lines.append("")
        
        # Class Distribution
        class_counts = imbalance_info.get('class_counts', {})
        if class_counts:
            report_lines.append("Class Distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / imbalance_info['total_samples']) * 100
                report_lines.append(f"  {class_name}: {count} ({percentage:.2f}%)")
        report_lines.append("")
        
        # Resampling Section
        if resampling_info:
            report_lines.append("2. RESAMPLING")
            report_lines.append("-" * 30)
            report_lines.append(f"Method: {resampling_info.get('method', 'N/A').upper()}")
            report_lines.append(f"Sampling Strategy: {resampling_info.get('sampling_strategy', 'N/A')}")
            report_lines.append(f"Original Shape: {resampling_info.get('original_shape', 'N/A')}")
            report_lines.append(f"Resampled Shape: {resampling_info.get('resampled_shape', 'N/A')}")
            report_lines.append(f"Samples Added: {resampling_info.get('samples_added', 'N/A')}")
            report_lines.append(f"Samples Removed: {resampling_info.get('samples_removed', 'N/A')}")
            report_lines.append("")
            
            # Resampled Class Distribution
            resampled_counts = resampling_info.get('resampled_counts', {})
            if resampled_counts:
                report_lines.append("Resampled Class Distribution:")
                total_resampled = sum(resampled_counts.values())
                for class_name, count in resampled_counts.items():
                    percentage = (count / total_resampled) * 100
                    report_lines.append(f"  {class_name}: {count} ({percentage:.2f}%)")
            report_lines.append("")
        
        # Binning Section
        if binning_info:
            report_lines.append("3. BINNING")
            report_lines.append("-" * 30)
            report_lines.append(f"Method: {binning_info.get('method', 'N/A').upper()}")
            report_lines.append(f"Number of Bins: {binning_info.get('n_bins', 'N/A')}")
            report_lines.append(f"Bin Labels: {binning_info.get('bin_labels', 'N/A')}")
            report_lines.append("")
            
            # Binned Class Distribution
            bin_counts = binning_info.get('bin_counts', {})
            if bin_counts:
                report_lines.append("Binned Class Distribution:")
                total_binned = sum(bin_counts.values())
                for bin_name, count in bin_counts.items():
                    percentage = (count / total_binned) * 100
                    report_lines.append(f"  {bin_name}: {count} ({percentage:.2f}%)")
            report_lines.append("")
        
        # Class Weights Section
        if weights_info:
            report_lines.append("4. CLASS WEIGHTS")
            report_lines.append("-" * 30)
            report_lines.append(f"Method: {weights_info.get('method', 'N/A')}")
            report_lines.append(f"Min Weight: {weights_info.get('min_weight', 'N/A'):.4f}")
            report_lines.append(f"Max Weight: {weights_info.get('max_weight', 'N/A'):.4f}")
            report_lines.append(f"Weight Ratio: {weights_info.get('weight_ratio', 'N/A'):.4f}")
            report_lines.append("")
            
            # Individual Class Weights
            class_weights = weights_info.get('class_weights', {})
            if class_weights:
                report_lines.append("Class Weights:")
                for class_name, weight in class_weights.items():
                    report_lines.append(f"  {class_name}: {weight:.4f}")
            report_lines.append("")
        
        # Export Section
        if export_info:
            report_lines.append("5. EXPORT")
            report_lines.append("-" * 30)
            report_lines.append(f"Output Directory: {self.output_dir}")
            report_lines.append(f"Training File: {export_info.get('train_file', 'N/A')}")
            report_lines.append(f"Training Shape: {export_info.get('train_shape', 'N/A')}")
            if 'test_file' in export_info:
                report_lines.append(f"Test File: {export_info.get('test_file', 'N/A')}")
                report_lines.append(f"Test Shape: {export_info.get('test_shape', 'N/A')}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_report(self, 
                   report_content: str,
                   filename: Optional[str] = None) -> str:
        """
        Save the summary report to a text file.
        
        Args:
            report_content: Report content string
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            filename = f"balens_report_{self.timestamp}.txt"
        
        report_path = os.path.join(self.output_dir, filename)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path


def export_balanced_data(X_train: Union[pd.DataFrame, np.ndarray],
                        y_train: Union[pd.Series, np.ndarray],
                        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
                        output_dir: str = "balens_output",
                        feature_names: Optional[list] = None,
                        target_name: str = "target",
                        prefix: str = "balanced") -> Dict:
    """
    Convenience function to export balanced data.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features (optional)
        y_test: Test targets (optional)
        output_dir: Output directory
        feature_names: Feature names
        target_name: Target column name
        prefix: File prefix
        
    Returns:
        Dictionary with export information
    """
    exporter = DataExporter(output_dir)
    return exporter.export_balanced_data(
        X_train, y_train, X_test, y_test, 
        feature_names, target_name, prefix
    )


def generate_and_save_report(imbalance_info: Dict,
                           resampling_info: Optional[Dict] = None,
                           binning_info: Optional[Dict] = None,
                           weights_info: Optional[Dict] = None,
                           export_info: Optional[Dict] = None,
                           output_dir: str = "balens_output",
                           filename: Optional[str] = None) -> str:
    """
    Convenience function to generate and save a report.
    
    Args:
        imbalance_info: Imbalance detection information
        resampling_info: Resampling information (optional)
        binning_info: Binning information (optional)
        weights_info: Class weights information (optional)
        export_info: Export information (optional)
        output_dir: Output directory
        filename: Report filename (optional)
        
    Returns:
        Path to saved report file
    """
    exporter = DataExporter(output_dir)
    report_content = exporter.generate_summary_report(
        imbalance_info, resampling_info, binning_info, weights_info, export_info
    )
    return exporter.save_report(report_content, filename) 