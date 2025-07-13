"""
weights.py: Class weights computation for balens
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from sklearn.utils.class_weight import compute_class_weight


class ClassWeightsComputer:
    """Compute class weights for imbalanced datasets."""
    
    def __init__(self):
        self.class_weights = None
        self.classes = None
        self.weight_method = None
        
    def compute_weights(self, 
                       y: Union[pd.Series, np.ndarray],
                       method: str = "balanced",
                       classes: Optional[Union[list, np.ndarray]] = None) -> Dict:
        """
        Compute class weights for imbalanced classification.
        
        Args:
            y: Target variable
            method: Weight computation method ('balanced', 'balanced_subsample', or dict)
            classes: List of classes (auto-detected if None)
            
        Returns:
            Dictionary with class weights and metadata
        """
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Get unique classes if not provided
        if classes is None:
            self.classes = np.unique(y_array)
        else:
            self.classes = np.array(classes)
        
        self.weight_method = method
        
        # Compute class weights using sklearn
        if method in ['balanced', 'balanced_subsample']:
            self.class_weights = compute_class_weight(
                class_weight=method,
                classes=self.classes,
                y=y_array
            )
        elif isinstance(method, dict):
            # Custom weights provided
            self.class_weights = np.array([method.get(cls, 1.0) for cls in self.classes])
        else:
            raise ValueError(f"Unknown weight method: {method}")
        
        # Create weights dictionary
        weights_dict = dict(zip(self.classes, self.class_weights))
        
        # Calculate additional statistics
        class_counts = pd.Series(y_array).value_counts()
        total_samples = len(y_array)
        
        weights_info = {
            "method": method,
            "class_weights": weights_dict,
            "classes": self.classes.tolist(),
            "class_counts": class_counts.to_dict(),
            "total_samples": total_samples,
            "n_classes": len(self.classes),
            "min_weight": float(np.min(self.class_weights)),
            "max_weight": float(np.max(self.class_weights)),
            "weight_ratio": float(np.max(self.class_weights) / np.min(self.class_weights))
        }
        
        return weights_info
    
    def get_sample_weights(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Get sample weights vector for the given target variable.
        
        Args:
            y: Target variable
            
        Returns:
            Array of sample weights
        """
        if self.class_weights is None:
            raise ValueError("Must call compute_weights first")
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Create sample weights vector
        sample_weights = np.zeros(len(y_array))
        for i, cls in enumerate(self.classes):
            mask = y_array == cls
            sample_weights[mask] = self.class_weights[i]
        
        return sample_weights
    
    def get_sklearn_weights(self) -> Dict:
        """
        Get class weights in sklearn-compatible format.
        
        Returns:
            Dictionary mapping classes to weights
        """
        if self.class_weights is None:
            raise ValueError("Must call compute_weights first")
        
        return dict(zip(self.classes, self.class_weights))


def compute_class_weights(y: Union[pd.Series, np.ndarray],
                         method: str = "balanced",
                         classes: Optional[Union[list, np.ndarray]] = None) -> Dict:
    """
    Convenience function to compute class weights.
    
    Args:
        y: Target variable
        method: Weight computation method
        classes: List of classes
        
    Returns:
        Dictionary with class weights and metadata
    """
    computer = ClassWeightsComputer()
    return computer.compute_weights(y, method, classes)


def get_sample_weights(y: Union[pd.Series, np.ndarray],
                      method: str = "balanced",
                      classes: Optional[Union[list, np.ndarray]] = None) -> np.ndarray:
    """
    Convenience function to get sample weights vector.
    
    Args:
        y: Target variable
        method: Weight computation method
        classes: List of classes
        
    Returns:
        Array of sample weights
    """
    computer = ClassWeightsComputer()
    computer.compute_weights(y, method, classes)
    return computer.get_sample_weights(y)


def get_sklearn_class_weights(y: Union[pd.Series, np.ndarray],
                             method: str = "balanced",
                             classes: Optional[Union[list, np.ndarray]] = None) -> Dict:
    """
    Convenience function to get sklearn-compatible class weights.
    
    Args:
        y: Target variable
        method: Weight computation method
        classes: List of classes
        
    Returns:
        Dictionary mapping classes to weights
    """
    computer = ClassWeightsComputer()
    computer.compute_weights(y, method, classes)
    return computer.get_sklearn_weights()


def validate_weight_method(method: str) -> bool:
    """Validate if a weight computation method is supported."""
    valid_methods = ['balanced', 'balanced_subsample']
    return method in valid_methods or isinstance(method, dict) 