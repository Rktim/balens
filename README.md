# balens

A universal Python toolkit for detecting and fixing data imbalance with support for SMOTE, ADASYN, class weighting, intelligent binning, and auto export. Ideal for ML pipelines, AutoML, and data science workflows.

## Features
- Automatic imbalance detection
- Smart binning for regression targets
- Resampling (SMOTE, ADASYN, etc.)
- Class weights computation
- Export balanced datasets and reports
- CLI and Python SDK

## Installation
```bash
pip install .
```

## CLI Usage
```bash
balens fix --file data.csv --target Outcome --method smote --auto-bin --export
```

## Python SDK Usage
```python
from balens import auto_balance
X_res, y_res = auto_balance(df, target="Outcome", method="smote")
``` 