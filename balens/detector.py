"""
detector.py: Imbalance detection for balens
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.utils import check_array


class ImbalanceDetector:
    """Detect class imbalance in datasets."""
    
    def __init__(self):
        self.target_column = None
        self.class_counts = None
        self.imbalance_ratio = None
        self.severity = None
        
    def detect_imbalance(self, 
                        data: Union[pd.DataFrame, np.ndarray], 
                        target: Optional[str] = None,
                        threshold_severe: float = 0.1,
                        threshold_moderate: float = 0.3) -> Dict:
        """
        Detect class imbalance in the dataset.
        
        Args:
            data: Input DataFrame or array
            target: Target column name (auto-detected if None)
            threshold_severe: Threshold for severe imbalance (default: 0.1)
            threshold_moderate: Threshold for moderate imbalance (default: 0.3)
            
        Returns:
            Dictionary with imbalance information
        """
        if isinstance(data, pd.DataFrame):
            return self._detect_imbalance_dataframe(data, target, threshold_severe, threshold_moderate)
        else:
            return self._detect_imbalance_array(data, threshold_severe, threshold_moderate)
    
    def _detect_imbalance_dataframe(self, 
                                   df: pd.DataFrame, 
                                   target: Optional[str],
                                   threshold_severe: float,
                                   threshold_moderate: float) -> Dict:
        """Detect imbalance in DataFrame."""
        if target is None:
            target = self._auto_detect_target(df)
        
        self.target_column = target
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        y = df[target]
        return self._analyze_imbalance(y, threshold_severe, threshold_moderate)
    
    def _detect_imbalance_array(self, 
                               y: np.ndarray,
                               threshold_severe: float,
                               threshold_moderate: float) -> Dict:
        """Detect imbalance in array."""
        y = check_array(y, ensure_2d=False, dtype=None)
        return self._analyze_imbalance(y, threshold_severe, threshold_moderate)
    
    def _auto_detect_target(self, df: pd.DataFrame) -> str:
        """Auto-detect the target column."""
        # Simple heuristic: look for columns with low cardinality
        # that might be categorical targets
        potential_targets = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10:
                potential_targets.append((col, df[col].nunique()))
        
        if not potential_targets:
            # If no obvious categorical columns, use the last column
            return df.columns[-1]
        
        # Return the column with the lowest cardinality
        return min(potential_targets, key=lambda x: x[1])[0]
    
    def _analyze_imbalance(self, 
                          y: Union[pd.Series, np.ndarray],
                          threshold_severe: float,
                          threshold_moderate: float) -> Dict:
        """Analyze imbalance in target variable."""
        # Get class counts
        if isinstance(y, pd.Series):
            class_counts = y.value_counts()
        else:
            unique, counts = np.unique(y, return_counts=True)
            class_counts = pd.Series(counts, index=unique)
        
        self.class_counts = class_counts
        
        # Calculate imbalance ratio (minority class / majority class)
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        self.imbalance_ratio = minority_count / majority_count
        
        # Determine severity
        if self.imbalance_ratio <= threshold_severe:
            self.severity = "severe"
        elif self.imbalance_ratio <= threshold_moderate:
            self.severity = "moderate"
        else:
            self.severity = "balanced"
        
        return {
            "target_column": self.target_column,
            "class_counts": self.class_counts.to_dict(),
            "imbalance_ratio": self.imbalance_ratio,
            "severity": self.severity,
            "total_samples": len(y),
            "n_classes": len(class_counts)
        }


def detect_imbalance(data: Union[pd.DataFrame, np.ndarray], 
                    target: Optional[str] = None,
                    threshold_severe: float = 0.1,
                    threshold_moderate: float = 0.3) -> Dict:
    """
    Convenience function to detect imbalance.
    
    Args:
        data: Input DataFrame or array
        target: Target column name (auto-detected if None)
        threshold_severe: Threshold for severe imbalance
        threshold_moderate: Threshold for moderate imbalance
        
    Returns:
        Dictionary with imbalance information
    """
    detector = ImbalanceDetector()
    return detector.detect_imbalance(data, target, threshold_severe, threshold_moderate) 