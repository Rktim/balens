"""
binning.py: Smart binning for regression targets in balens
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import KBinsDiscretizer


class SmartBinner:
    """Smart binning for regression targets."""
    
    def __init__(self):
        self.bin_edges = None
        self.bin_labels = None
        self.n_bins = None
        
    def bin_regression_target(self, 
                             y: Union[pd.Series, np.ndarray],
                             method: str = "quantile",
                             n_bins: int = 3,
                             labels: Optional[List[str]] = None,
                             custom_thresholds: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Convert regression target to categorical labels.
        
        Args:
            y: Target variable (regression values)
            method: Binning method ('quantile', 'uniform', 'custom')
            n_bins: Number of bins (ignored if custom_thresholds provided)
            labels: Custom labels for bins (e.g., ['Low', 'Medium', 'High'])
            custom_thresholds: Custom threshold values for binning
            
        Returns:
            Tuple of (binned_values, binning_info)
        """
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Handle missing values in target
        if np.isnan(y_array).any():
            print(f"Removing {np.isnan(y_array).sum()} samples with missing target values")
            y_array = y_array[~np.isnan(y_array)]
        
        if custom_thresholds is not None:
            return self._custom_binning(y_array, custom_thresholds, labels)
        else:
            return self._auto_binning(y_array, method, n_bins, labels)
    
    def _custom_binning(self, 
                       y: np.ndarray,
                       thresholds: List[float],
                       labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """Apply custom threshold-based binning."""
        # Sort thresholds and add min/max if needed
        sorted_thresholds = sorted(thresholds)
        bin_edges = [y.min()] + sorted_thresholds + [y.max()]
        
        # Create labels if not provided
        if labels is None:
            labels = [f"Bin_{i}" for i in range(len(bin_edges) - 1)]
        elif len(labels) != len(bin_edges) - 1:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of bins ({len(bin_edges) - 1})")
        
        # Apply binning
        binned = np.digitize(y, bin_edges[1:-1], right=True)
        binned_labels = np.array([labels[i] for i in binned])
        
        self.bin_edges = bin_edges
        self.bin_labels = labels
        self.n_bins = len(bin_edges) - 1
        
        binning_info = {
            "method": "custom",
            "bin_edges": bin_edges,
            "bin_labels": labels,
            "n_bins": self.n_bins,
            "bin_counts": pd.Series(binned_labels).value_counts().to_dict()
        }
        
        return binned_labels, binning_info
    
    def _auto_binning(self, 
                     y: np.ndarray,
                     method: str,
                     n_bins: int,
                     labels: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """Apply automatic binning using sklearn's KBinsDiscretizer."""
        if method not in ['quantile', 'uniform']:
            raise ValueError("Method must be 'quantile' or 'uniform'")
        
        # Create discretizer
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy=method
        )
        
        # Fit and transform
        y_reshaped = y.reshape(-1, 1)
        binned_indices = discretizer.fit_transform(y_reshaped).flatten().astype(int)
        
        # Get bin edges
        bin_edges = discretizer.bin_edges_[0]
        
        # Create labels
        if labels is None:
            if n_bins == 3:
                labels = ['Low', 'Medium', 'High']
            elif n_bins == 4:
                labels = ['Low', 'Medium', 'High', 'Very_High']
            elif n_bins == 5:
                labels = ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']
            else:
                labels = [f"Bin_{i}" for i in range(n_bins)]
        elif len(labels) != n_bins:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of bins ({n_bins})")
        
        # Convert indices to labels
        binned_labels = np.array([labels[i] for i in binned_indices])
        
        self.bin_edges = bin_edges
        self.bin_labels = labels
        self.n_bins = n_bins
        
        binning_info = {
            "method": method,
            "bin_edges": bin_edges.tolist(),
            "bin_labels": labels,
            "n_bins": n_bins,
            "bin_counts": pd.Series(binned_labels).value_counts().to_dict()
        }
        
        return binned_labels, binning_info
    
    def get_bin_statistics(self, y: Union[pd.Series, np.ndarray]) -> Dict:
        """Get statistics for each bin."""
        if self.bin_edges is None:
            raise ValueError("Must call bin_regression_target first")
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Remove missing values
        if np.isnan(y_array).any():
            y_array = y_array[~np.isnan(y_array)]
        
        # Recreate binning to get current bin assignments
        if len(self.bin_edges) == self.n_bins + 1:
            # Auto-binning case
            binned_indices = np.digitize(y_array, self.bin_edges[1:-1], right=True)
            binned_labels = np.array([self.bin_labels[i] for i in binned_indices])
        else:
            # Custom binning case
            binned_indices = np.digitize(y_array, self.bin_edges[1:-1], right=True)
            binned_labels = np.array([self.bin_labels[i] for i in binned_indices])
        
        # Calculate statistics for each bin
        bin_stats = {}
        for i, label in enumerate(self.bin_labels):
            mask = binned_labels == label
            if np.any(mask):
                bin_values = y_array[mask]
                bin_stats[label] = {
                    "count": len(bin_values),
                    "mean": float(np.mean(bin_values)),
                    "std": float(np.std(bin_values)),
                    "min": float(np.min(bin_values)),
                    "max": float(np.max(bin_values)),
                    "range": float(np.max(bin_values) - np.min(bin_values))
                }
        
        return bin_stats


def bin_regression_target(y: Union[pd.Series, np.ndarray],
                         method: str = "quantile",
                         n_bins: int = 3,
                         labels: Optional[List[str]] = None,
                         custom_thresholds: Optional[List[float]] = None) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to bin regression targets.
    
    Args:
        y: Target variable (regression values)
        method: Binning method ('quantile', 'uniform', 'custom')
        n_bins: Number of bins
        labels: Custom labels for bins
        custom_thresholds: Custom threshold values
        
    Returns:
        Tuple of (binned_values, binning_info)
    """
    binner = SmartBinner()
    return binner.bin_regression_target(y, method, n_bins, labels, custom_thresholds) 