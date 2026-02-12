"""
quick_check.py

Ultra-fast sanity check (< 30 seconds)
Run this first before test_pipeline.py

Usage:
    python quick_check.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def main():
    print("=" * 50)
    print("QUICK SANITY CHECK")
    print("=" * 50)
    
    errors = []
    
    # 1. Check imports
    print("\n[1/5] Checking imports...", end=" ")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        from sklearn.metrics import f1_score
        print("✓")
    except ImportError as e:
        print(f"✗ {e}")
        errors.append(f"Import error: {e}")
    
    # 2. Check local modules
    print("[2/5] Checking local modules...", end=" ")
    try:
        from models.models import get_model, get_model_input_size
        from preprocessing.preprocessing import get_stratified_kfold
        from evaluation.evaluation import MetricsCalculator
        print("✓")
    except ImportError as e:
        print(f"✗ {e}")
        errors.append(f"Local module error: {e}")
    
    # 3. Check dataset
    print("[3/5] Checking dataset...", end=" ")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set')
    if os.path.exists(data_dir):
        colon = os.path.exists(os.path.join(data_dir, 'colon_image_sets'))
        lung = os.path.exists(os.path.join(data_dir, 'lung_image_sets'))
        if colon and lung:
            print("✓")
        else:
            print("✗ Missing subdirectories")
            errors.append("Dataset subdirectories missing")
    else:
        print(f"✗ Not found: {data_dir}")
        errors.append("Dataset not found")
    
    # 4. Check device
    print("[4/5] Checking device...", end=" ")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CPU only (will be slow)")
    
    # 5. Quick model test
    print("[5/5] Quick model test...", end=" ")
    try:
        from models.models import get_model
        model = get_model('resnet50', num_classes=5, pretrained=False)  # No download
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5), f"Wrong shape: {out.shape}"
        print("✓")
    except Exception as e:
        print(f"✗ {e}")
        errors.append(f"Model test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    if not errors:
        print("✓ ALL CHECKS PASSED")
        print("\nNext steps:")
        print("  1. Run: python test_pipeline.py")
        print("  2. Run: python run_pipeline.py")
        return 0
    else:
        print("✗ ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())