#!/usr/bin/env python3
"""Quick requirements check for Ivan's analysis."""

import sys
import importlib.metadata
from packaging import version

def check_requirements():
    """Check if all requirements are met."""
    print("Checking Python version...")
    py_version = sys.version_info
    if py_version.major < 3 or py_version.minor < 8:
        print(f"❌ Python 3.8+ required. Found: {sys.version}")
        return False
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    print("\nChecking required packages...")
    requirements = {
        'sentence-transformers': '3.0.0',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'pandas': '1.5.0',
        'numpy': '1.22.0',
        'scikit-learn': '1.0.0',
        'nltk': '3.8.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.12.0'
    }
    
    all_good = True
    for package, min_version in requirements.items():
        try:
            # Handle package name differences
            pkg_name = package.replace('-', '_')
            installed_version = importlib.metadata.version(package)
            
            if version.parse(installed_version) >= version.parse(min_version):
                print(f"✓ {package}: {installed_version}")
            else:
                print(f"⚠️  {package}: {installed_version} (need >= {min_version})")
                all_good = False
        except Exception as e:
            print(f"❌ {package}: Not installed")
            all_good = False
    
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("ℹ️  No GPU detected (CPU mode will be slower)")
    except:
        print("⚠️  Could not check GPU status")
    
    if all_good:
        print("\n✅ All requirements satisfied!")
    else:
        print("\n❌ Some requirements not met. Run:")
        print("   pip install -r scripts/ivan_analysis/requirements.txt")
    
    return all_good

if __name__ == "__main__":
    check_requirements()