"""
Quick setup script for Google Colab
Run this in a Colab cell to set up the environment
"""
import os
import sys
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / 'src'
if not src_dir.exists():
    # Try parent directory
    src_dir = current_dir.parent / 'src'
    if not src_dir.exists():
        # Try brain_classifier/src
        src_dir = current_dir / 'brain_classifier' / 'src'

if src_dir.exists():
    sys.path.insert(0, str(src_dir))
    print(f"✓ Added {src_dir} to Python path")
else:
    print(f"✗ Could not find src directory. Current dir: {current_dir}")
    print("Please ensure the src folder is in the same directory as this notebook")

# Verify imports
try:
    from augmentations import get_train_transform, get_val_transform
    from models import get_model
    from training import train_model
    from eval import evaluate_model, plot_confusion_matrix
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nMake sure you have uploaded the 'src' folder to Colab.")

