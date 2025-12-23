"""
Run this in a Colab cell to fix the training.py file directly
This will update the file in Colab to remove the verbose parameter issue
"""

import os

# Path to training.py in Colab (adjust if your path is different)
training_file = '/content/src/training.py'

# Alternative paths to try
possible_paths = [
    '/content/src/training.py',
    '/content/brain_classifier/src/training.py',
    'src/training.py',
    'brain_classifier/src/training.py'
]

training_file = None
for path in possible_paths:
    if os.path.exists(path):
        training_file = path
        print(f"✓ Found training.py at: {path}")
        break

if training_file is None:
    print("✗ Could not find training.py. Please check your file paths.")
    print("Current directory:", os.getcwd())
    print("\nTrying to find it...")
    import subprocess
    result = subprocess.run(['find', '/content', '-name', 'training.py', '-type', 'f'], 
                          capture_output=True, text=True)
    if result.stdout:
        print("Found training.py at:")
        print(result.stdout)
    else:
        print("Could not locate training.py automatically.")
else:
    # Read the file
    with open(training_file, 'r') as f:
        content = f.read()
    
    # Check if it has the problematic verbose parameter
    if 'verbose=True' in content or 'verbose' in content and 'ReduceLROnPlateau' in content:
        print(f"\n⚠ Found verbose parameter in {training_file}")
        print("Fixing it...")
        
        # Replace the problematic line
        import re
        # Pattern to match ReduceLROnPlateau with verbose
        pattern = r'ReduceLROnPlateau\([^)]*verbose\s*=\s*True[^)]*\)'
        replacement = 'ReduceLROnPlateau(optimizer, mode=\'min\', factor=0.5, patience=3)'
        
        new_content = re.sub(pattern, replacement, content)
        
        # Also try simpler replacement
        new_content = new_content.replace('verbose=True', '')
        new_content = new_content.replace(', verbose=True', '')
        new_content = new_content.replace('verbose=True,', '')
        
        # Write back
        with open(training_file, 'w') as f:
            f.write(new_content)
        
        print("✓ Fixed! Please restart runtime or reload the module.")
    else:
        print("✓ File looks good - no verbose parameter found.")
        print("If you're still getting the error, try:")
        print("  1. Runtime → Restart runtime")
        print("  2. Or run: import importlib; import sys; del sys.modules['training']; from training import train_model")

