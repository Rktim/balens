# 📊 Balens: Balance Your Data Like a Pro

A universal Python toolkit for detecting and resolving data imbalance in classification and regression problems. With Balens, you get intelligent binning, resampling techniques like SMOTE and ADASYN, class weight computation, and one-line fixes for your ML datasets. Whether you're building ML pipelines, AutoML workflows, or doing data science at scale — Balens has your back.

[![PyPI version](https://badge.fury.io/py/balens.svg)](https://pypi.org/project/balens/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Downloads](https://static.pepy.tech/personalized-badge/balens?period=total&units=international_system&left_color=grey&right_color=yellow)



---

## 🌎 Key Features

* ✨ **Auto Imbalance Detection**: Instantly detects imbalance severity and distribution.
* 🌍 **Smart Binning**: Auto-bins continuous regression targets for better balance.
* 💪 **Resampling Methods**: Built-in support for SMOTE, ADASYN, Random Oversampling, and more.
* ⚖️ **Class Weights Calculation**: For models that support weighting.
* 📃 **Export Tools**: Automatically saves balanced datasets and imbalance reports.
* 🔧 **CLI + Python SDK**: Dev-friendly and script-ready.

---

## 🚀 Installation

```bash
pip install balens
```

---

## 🔍 CLI Usage

```bash
balens fix --file data.csv --target Outcome --method smote --auto-bin --export
```

### Other CLI Commands

```bash
balens detect --file data.csv --target Outcome      # Check imbalance stats
balens fix --method adasyn --auto-bin               # Fix imbalance with ADASYN and binning
```

---

## 👾 Python SDK

```python
from balens import auto_balance

X_res, y_res = auto_balance(df, target="Outcome", method="smote")
```

You can also:

* Perform just detection:

  ```python
  from balens import detect_imbalance
  detect_imbalance(df, target="Outcome")
  ```
* Use regression binning:

  ```python
  from balens import smart_bin
  binned_target = smart_bin(df["target"])
  ```

---

## 🎓 Ideal For:

* Data Scientists needing quick imbalance fixes
* ML Engineers building production pipelines
* AutoML tool developers
* Researchers working on highly skewed datasets

---


## 📦 Contributing

We welcome pull requests, feature ideas, and issue reports! 

---

## 🚫 License

[MIT](LICENSE)
