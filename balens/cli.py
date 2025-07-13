"""
cli.py: Typer-based CLI for balens
"""

import typer
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

# Import balens functions
from . import auto_balance, detect_imbalance_only, get_available_methods, validate_method

app = typer.Typer(
    name="balens",
    help="A universal Python toolkit for detecting and fixing data imbalance",
    add_completion=False
)


@app.command()
def fix(
    file: str = typer.Argument(..., help="Path to CSV file"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name (auto-detected if not provided)"),
    method: str = typer.Option("smote", "--method", "-m", help="Resampling method (smote, adasyn, random_oversample)"),
    auto_bin: bool = typer.Option(False, "--auto-bin", help="Automatically bin regression targets"),
    export: bool = typer.Option(False, "--export", help="Export balanced datasets and report"),
    output_dir: str = typer.Option("balens_output", "--output-dir", help="Output directory for exports")
):
    """
    Fix data imbalance in a dataset.
    
    This command automatically:
    - Cleans the dataset (removes duplicates, handles missing values)
    - Detects imbalance
    - Applies resampling to balance the data
    - Optionally bins regression targets
    - Exports results if requested
    
    Examples:
        balens fix data.csv --target Outcome --export
        balens fix data.csv --auto-bin --method adasyn --export
        balens fix data.csv --target HAEMOGLOBIN --auto-bin --export
    """
    try:
        # Validate file exists
        if not Path(file).exists():
            typer.echo(f"❌ Error: File '{file}' not found", err=True)
            sys.exit(1)
        
        # Validate method
        if not validate_method(method):
            available_methods = list(get_available_methods().keys())
            typer.echo(f"❌ Error: Invalid method '{method}'. Available methods: {', '.join(available_methods)}", err=True)
            sys.exit(1)
        
        typer.echo(f"🚀 Starting balens fix for: {file}")
        if target:
            typer.echo(f"🎯 Target column: {target}")
        else:
            typer.echo(f"🎯 Target column: auto-detect")
        typer.echo(f"🔄 Method: {method.upper()}")
        typer.echo(f"📊 Auto-bin: {auto_bin}")
        typer.echo(f"💾 Export: {export}")
        typer.echo()
        
        # Run auto_balance
        X_resampled, y_resampled, info = auto_balance(
            data=file,
            target=target,
            method=method,
            auto_bin=auto_bin,
            export=export,
            output_dir=output_dir
        )
        
        # Display results
        typer.echo("\n" + "="*60)
        typer.echo("🎉 BALENS BALANCING COMPLETED SUCCESSFULLY!")
        typer.echo("="*60)
        
        # Preprocessing info
        if "preprocessing" in info:
            preprocessing = info["preprocessing"]
            typer.echo(f"📊 Original shape: {preprocessing['original_shape']}")
            typer.echo(f"✅ Final shape: {preprocessing['final_shape']}")
            typer.echo(f"🎯 Target column: {preprocessing['target_column']}")
        
        # Imbalance info
        imbalance = info["imbalance_info"]
        typer.echo(f"⚖️  Original imbalance: {imbalance['severity'].upper()} (ratio: {imbalance['imbalance_ratio']:.4f})")
        typer.echo(f"📈 Classes: {imbalance['n_classes']}")
        
        # Resampling info
        resampling = info["resampling_info"]
        typer.echo(f"🔄 Resampling method: {resampling['method'].upper()}")
        typer.echo(f"📊 Original training: {resampling['original_shape']}")
        typer.echo(f"📊 Resampled training: {resampling['resampled_shape']}")
        typer.echo(f"➕ Samples added: {resampling['samples_added']}")
        
        # Binning info if applicable
        if info["binning_info"]:
            binning = info["binning_info"]
            typer.echo(f"📊 Binning method: {binning['method'].upper()}")
            typer.echo(f"📊 Number of bins: {binning['n_bins']}")
        
        # Export info if applicable
        if export and "export_info" in info:
            export_info = info["export_info"]
            typer.echo(f"\n💾 Exported files:")
            typer.echo(f"  📁 Training: {export_info['train_file']}")
            if "test_file" in export_info:
                typer.echo(f"  📁 Test: {export_info['test_file']}")
            typer.echo(f"  📄 Report: {info['report_path']}")
        
        typer.echo(f"\n✅ All done! Your balanced dataset is ready.")
        
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@app.command()
def detect(
    file: str = typer.Argument(..., help="Path to CSV file"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name")
):
    """
    Detect imbalance in a dataset without fixing it.
    
    This command analyzes your dataset and provides:
    - Imbalance severity assessment
    - Class distribution
    - Recommendations for balancing
    
    Example:
        balens detect data.csv --target Outcome
    """
    try:
        # Validate file exists
        if not Path(file).exists():
            typer.echo(f"❌ Error: File '{file}' not found", err=True)
            sys.exit(1)
        
        typer.echo(f"🔍 Analyzing imbalance in: {file}")
        if target:
            typer.echo(f"🎯 Target column: {target}")
        else:
            typer.echo(f"🎯 Target column: auto-detect")
        typer.echo()
        
        # Detect imbalance
        info = detect_imbalance_only(file, target)
        
        # Display results
        typer.echo("="*60)
        typer.echo("📊 IMBALANCE DETECTION RESULTS")
        typer.echo("="*60)
        
        typer.echo(f"🎯 Target column: {info['target_column']}")
        typer.echo(f"📊 Total samples: {info['total_samples']:,}")
        typer.echo(f"📈 Number of classes: {info['n_classes']}")
        typer.echo(f"⚖️  Imbalance ratio: {info['imbalance_ratio']:.4f}")
        typer.echo(f"🚨 Severity: {info['severity'].upper()}")
        
        typer.echo("\n📊 Class distribution:")
        for class_name, count in info['class_counts'].items():
            percentage = (count / info['total_samples']) * 100
            typer.echo(f"  • {class_name}: {count:,} ({percentage:.2f}%)")
        
        # Recommendations
        typer.echo("\n💡 Recommendations:")
        if info['severity'] == 'severe':
            typer.echo("  ⚠️  Severe imbalance detected!")
            typer.echo("  🔧 Recommended: Use 'balens fix data.csv --method smote --export'")
        elif info['severity'] == 'moderate':
            typer.echo("  ⚠️  Moderate imbalance detected!")
            typer.echo("  🔧 Recommended: Use 'balens fix data.csv --method adasyn --export'")
        else:
            typer.echo("  ✅ Dataset appears balanced. No action needed.")
        
        typer.echo(f"\n💡 To fix imbalance, run: balens fix {file} --target {info['target_column']} --export")
        
    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


@app.command()
def methods():
    """
    Show available resampling methods.
    """
    available_methods = get_available_methods()
    
    typer.echo("🔄 Available resampling methods:")
    typer.echo("="*40)
    
    for method, description in available_methods.items():
        typer.echo(f"• {method}: {description}")
    
    typer.echo(f"\n💡 Usage: balens fix data.csv --method <method_name>")


@app.command()
def version():
    """
    Show balens version.
    """
    from . import __version__
    typer.echo(f"balens version {__version__}")


if __name__ == "__main__":
    app() 