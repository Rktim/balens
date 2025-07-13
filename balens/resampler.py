"""
resampler.py: Resampling methods (SMOTE, ADASYN, etc.) for balens
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, Optional
from sklearn.utils import check_X_y
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek


class DataResampler:
    """Resample imbalanced datasets using various techniques."""
    
    def __init__(self):
        self.resampler = None
        self.method = None
        self.sampling_strategy = None
        
    def resample(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray],
                 method: str = "smote",
                 sampling_strategy: str = "auto",
                 random_state: Optional[int] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Resample the dataset to balance classes.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Resampling method ('smote', 'adasyn', 'random_oversample', 'random_undersample', 'smoteenn', 'smotetomek')
            sampling_strategy: Sampling strategy ('auto', 'balanced', 'majority', 'minority', or dict)
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for specific resamplers
            
        Returns:
            Tuple of (X_resampled, y_resampled, resampling_info)
        """
        # Validate inputs
        X, y = check_X_y(X, y, dtype=None)
        
        # Store method info
        self.method = method
        self.sampling_strategy = sampling_strategy
        
        # Get class distribution before resampling
        original_counts = pd.Series(y).value_counts().to_dict()
        
        # Create and apply resampler
        if method == "smote":
            self.resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        elif method == "adasyn":
            self.resampler = ADASYN(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        elif method == "random_oversample":
            self.resampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        elif method == "random_undersample":
            self.resampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        elif method == "smoteenn":
            self.resampler = SMOTEENN(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        elif method == "smotetomek":
            self.resampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        # Apply resampling
        X_resampled, y_resampled = self.resampler.fit_resample(X, y)
        
        # Get class distribution after resampling
        resampled_counts = pd.Series(y_resampled).value_counts().to_dict()
        
        # Calculate resampling statistics
        resampling_info = {
            "method": method,
            "sampling_strategy": sampling_strategy,
            "random_state": random_state,
            "original_shape": X.shape,
            "resampled_shape": X_resampled.shape,
            "original_counts": original_counts,
            "resampled_counts": resampled_counts,
            "samples_added": len(y_resampled) - len(y),
            "samples_removed": len(y) - len(y_resampled) if len(y_resampled) < len(y) else 0
        }
        
        return X_resampled, y_resampled, resampling_info
    
    def get_resampling_summary(self) -> Dict:
        """Get summary of the resampling operation."""
        if self.resampler is None:
            raise ValueError("Must call resample first")
        
        return {
            "method": self.method,
            "sampling_strategy": self.sampling_strategy,
            "resampler_params": self.resampler.get_params()
        }


def resample_data(X: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.Series, np.ndarray],
                 method: str = "smote",
                 random_state: Optional[int] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Resample data using various methods.
    
    Args:
        X: Feature matrix
        y: Target vector
        method: Resampling method
        random_state: Random state for reproducibility
        **kwargs: Additional arguments for resampling
        
    Returns:
        Tuple of (X_resampled, y_resampled, info_dict)
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    original_shape = X.shape
    original_samples = len(y)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    
    print(f"📊 Original class distribution: {dict(zip(unique_classes, class_counts))}")
    
    # Handle extreme imbalance cases
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    
    if min_class_count < 5:
        print(f"⚠️  Warning: Some classes have very few samples ({min_class_count}). Using random_oversample as fallback.")
        method = "random_oversample"
    
    try:
        if method == "smote":
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state, **kwargs)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif method == "adasyn":
            from imblearn.over_sampling import ADASYN
            adasyn = ADASYN(random_state=random_state, **kwargs)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
        elif method == "random_oversample":
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=random_state, **kwargs)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
        elif method == "random_undersample":
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=random_state, **kwargs)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
        elif method == "smoteenn":
            from imblearn.combine import SMOTEENN
            smoteenn = SMOTEENN(random_state=random_state, **kwargs)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            
        elif method == "smotetomek":
            from imblearn.combine import SMOTETomek
            smotetomek = SMOTETomek(random_state=random_state, **kwargs)
            X_resampled, y_resampled = smotetomek.fit_resample(X, y)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
    except Exception as e:
        print(f"⚠️  {method.upper()} failed: {str(e)}")
        print("🔄 Falling back to random_oversample...")
        
        # Fallback to random oversampling
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        method = "random_oversample (fallback)"
    
    resampled_shape = X_resampled.shape
    samples_added = len(y_resampled) - original_samples
    
    # Get new class distribution
    unique_classes_new, class_counts_new = np.unique(y_resampled, return_counts=True)
    
    print(f"📊 New class distribution: {dict(zip(unique_classes_new, class_counts_new))}")
    
    info = {
        "method": method,
        "original_shape": original_shape,
        "resampled_shape": resampled_shape,
        "samples_added": samples_added,
        "n_classes": n_classes,
        "original_class_counts": dict(zip(unique_classes, class_counts)),
        "resampled_class_counts": dict(zip(unique_classes_new, class_counts_new))
    }
    
    return X_resampled, y_resampled, info


def get_available_methods() -> Dict[str, str]:
    """Get available resampling methods and their descriptions."""
    return {
        "smote": "Synthetic Minority Over-sampling Technique",
        "adasyn": "Adaptive Synthetic Sampling",
        "random_oversample": "Random Over-sampling",
        "random_undersample": "Random Under-sampling",
        "smoteenn": "SMOTE + Edited Nearest Neighbors",
        "smotetomek": "SMOTE + Tomek Links"
    }


def validate_method(method: str) -> bool:
    """Validate if a resampling method is supported."""
    available_methods = get_available_methods()
    return method in available_methods 