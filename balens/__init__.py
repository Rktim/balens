"""
balens: A universal Python toolkit for detecting and fixing data imbalance
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Import core modules
from .detector import detect_imbalance, ImbalanceDetector
from .binning import bin_regression_target, SmartBinner
from .resampler import resample_data, DataResampler, get_available_methods, validate_method
from .weights import compute_class_weights, get_sample_weights, get_sklearn_class_weights, ClassWeightsComputer
from .exporter import export_balanced_data, generate_and_save_report, DataExporter

__version__ = "0.1.0"
__author__ = "Your Name"


def clean_dataset(df: pd.DataFrame, target: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Clean dataset by handling missing values, duplicates, and data quality issues.
    
    Args:
        df: Input DataFrame
        target: Target column name (auto-detected if None)
        
    Returns:
        Tuple of (cleaned_df, target_column)
    """
    print("🔧 Cleaning dataset...")
    
    # Auto-detect target if not provided
    if target is None:
        # Try to find a reasonable target column
        potential_targets = []
        for col in df.columns:
            # Skip columns with too many unique values (likely not categorical targets)
            if df[col].nunique() <= 50 and df[col].dtype in ['object', 'category', 'int64']:
                potential_targets.append(col)
        
        if potential_targets:
            # Choose the column with the most balanced distribution
            best_target = None
            best_balance = float('inf')
            for col in potential_targets:
                value_counts = df[col].value_counts()
                if len(value_counts) >= 2:
                    balance = value_counts.max() / value_counts.min()
                    if balance < best_balance:
                        best_balance = balance
                        best_target = col
            target = best_target or potential_targets[0]
        else:
            # Fallback to last column
            target = df.columns[-1]
    
    print(f"📊 Target column: {target}")
    
    # Remove columns that are all missing
    all_missing_cols = df.columns[df.isnull().all()].tolist()
    if all_missing_cols:
        print(f"🗑️  Removing columns with all missing values: {all_missing_cols}")
        df = df.drop(columns=all_missing_cols)
    
    # Remove duplicate rows
    original_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < original_rows:
        print(f"🗑️  Removed {original_rows - len(df)} duplicate rows")
    
    # Handle missing values in target column
    target_missing = df[target].isnull().sum()
    if target_missing > 0:
        print(f"🗑️  Removing {target_missing} rows with missing target values")
        df = df.dropna(subset=[target])
    
    # Handle missing values in features
    feature_cols = [col for col in df.columns if col != target]
    missing_counts = df[feature_cols].isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"🔧 Imputing missing values in {len(cols_with_missing)} columns...")
        
        # Separate numeric and categorical columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric columns with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            print(f"  📈 Imputed {len(numeric_cols)} numeric columns with median")
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            print(f"  📝 Imputed {len(categorical_cols)} categorical columns with mode")
    
    # Handle non-numeric columns that could cause issues during resampling
    print("🔧 Converting non-numeric columns for resampling...")
    feature_cols = [col for col in df.columns if col != target]
    
    # Identify problematic columns (dates, strings that can't be converted to numeric)
    problematic_cols = []
    for col in feature_cols:
        if col in df.columns:  # Check if column still exists
            sample_values = df[col].dropna().head(10)
            if len(sample_values) > 0:
                # Check if it looks like a date
                if any(str(val).count('-') >= 2 or str(val).count('/') >= 2 for val in sample_values):
                    problematic_cols.append(col)
                # Check if it's a string that can't be converted to numeric
                elif df[col].dtype == 'object':
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        problematic_cols.append(col)
    
    # Remove problematic columns
    if problematic_cols:
        print(f"🗑️  Removing non-numeric columns that could cause issues: {problematic_cols}")
        df = df.drop(columns=problematic_cols)
    
    print(f"✅ Dataset cleaned: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, target


def auto_balance(data: Union[pd.DataFrame, str],
                target: Optional[str] = None,
                method: str = "smote",
                auto_bin: bool = False,
                bin_method: str = "quantile",
                n_bins: int = 3,
                test_size: float = 0.2,
                random_state: Optional[int] = None,
                export: bool = False,
                output_dir: str = "balens_output",
                **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Main function to automatically balance a dataset.
    
    Args:
        data: DataFrame or path to CSV file
        target: Target column name (auto-detected if None)
        method: Resampling method ('smote', 'adasyn', 'random_oversample', etc.)
        auto_bin: Whether to automatically bin regression targets
        bin_method: Binning method if auto_bin is True
        n_bins: Number of bins for binning
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        export: Whether to export results
        output_dir: Output directory for exports
        **kwargs: Additional arguments for resampling
        
    Returns:
        Tuple of (X_resampled, y_resampled, info_dict)
    """
    # Load data if string path provided
    if isinstance(data, str):
        print(f"📁 Loading data from: {data}")
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Clean the dataset
    df, target = clean_dataset(df, target)
    
    # Detect imbalance
    imbalance_info = detect_imbalance(df, target)
    print(f"⚖️  Imbalance detected: {imbalance_info['severity']} (ratio: {imbalance_info['imbalance_ratio']:.4f})")
    
    # Split data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Determine if we should use stratified splitting
    use_stratify = len(np.unique(y)) <= 10 and not auto_bin
    
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Handle regression targets with binning if requested
    binning_info = None
    if auto_bin and imbalance_info['n_classes'] > 10:  # Assume regression if many classes
        print("📊 Converting regression target to categorical using binning...")
        y_train_binned, binning_info = bin_regression_target(
            y_train, method=bin_method, n_bins=n_bins
        )
        y_train = pd.Series(y_train_binned, index=y_train.index)
        y_test_binned, _ = bin_regression_target(
            y_test, method=bin_method, n_bins=n_bins
        )
        y_test = pd.Series(y_test_binned, index=y_test.index)
        
        # Remove any remaining NaN values after binning
        train_mask = ~pd.isnull(y_train)
        test_mask = ~pd.isnull(y_test)
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"✅ Binning complete: {len(np.unique(y_train))} classes")
    
    # Resample training data only
    print(f"🔄 Applying {method.upper()} resampling...")
    X_resampled, y_resampled, resampling_info = resample_data(
        X_train, y_train, method=method, random_state=random_state, **kwargs
    )
    
    # Compute class weights
    weights_info = compute_class_weights(y_resampled)
    
    # Prepare return information
    info = {
        "imbalance_info": imbalance_info,
        "resampling_info": resampling_info,
        "binning_info": binning_info,
        "weights_info": weights_info,
        "test_data": {
            "X_test": X_test,
            "y_test": y_test
        },
        "preprocessing": {
            "original_shape": df.shape if isinstance(data, str) else data.shape,
            "final_shape": df.shape,
            "target_column": target
        }
    }
    
    # Export if requested
    if export:
        print(f"💾 Exporting results to {output_dir}...")
        export_info = export_balanced_data(
            X_resampled, y_resampled, X_test, y_test,
            output_dir=output_dir,
            feature_names=X.columns.tolist(),
            target_name=target
        )
        
        # Generate and save report
        report_path = generate_and_save_report(
            imbalance_info, resampling_info, binning_info, weights_info, export_info,
            output_dir=output_dir
        )
        
        info["export_info"] = export_info
        info["report_path"] = report_path
        print(f"📄 Report saved to: {report_path}")
    
    print(f"🎉 Balancing complete! Original: {X_train.shape}, Resampled: {X_resampled.shape}")
    
    return X_resampled, y_resampled, info


def detect_imbalance_only(data: Union[pd.DataFrame, str], target: Optional[str] = None) -> Dict:
    """Detect imbalance without fixing it."""
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()
    
    # Clean dataset first
    df, target = clean_dataset(df, target)
    
    return detect_imbalance(df, target)


def resample_only(X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray],
                 method: str = "smote",
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Resample data without preprocessing."""
    return resample_data(X, y, method=method, **kwargs)


def get_weights_only(y: Union[pd.Series, np.ndarray], method: str = "balanced") -> Dict:
    """Compute class weights only."""
    return compute_class_weights(y, method=method)


# Export main functions
__all__ = [
    "auto_balance",
    "detect_imbalance",
    "bin_regression_target", 
    "resample_data",
    "compute_class_weights",
    "export_balanced_data",
    "generate_and_save_report",
    "detect_imbalance_only",
    "resample_only",
    "get_weights_only",
    "get_available_methods",
    "validate_method",
    "ImbalanceDetector",
    "SmartBinner", 
    "DataResampler",
    "ClassWeightsComputer",
    "DataExporter"
] 