from setuptools import setup, find_packages

setup(
    name="balens",
    version="0.0.1",
    description="A universal Python toolkit for detecting and fixing data imbalance.",
    author="Raktim Kalita",
    author_email="raktimkalita.ai@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "imbalanced-learn",
        "typer[all]"
    ],
    entry_points={
        "console_scripts": [
            "balens=balens.cli:app"
        ]
    },
    include_package_data=True,
    license="MIT",
    python_requires=">=3.7",
) 