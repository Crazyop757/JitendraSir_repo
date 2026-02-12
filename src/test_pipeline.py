"""
test_pipeline.py

Pre-flight Test Script for Multi-Model Training Pipeline
=========================================================

This script performs quick validation tests to ensure the full pipeline
will run without errors. Run this BEFORE starting the long training process.

Tests performed:
1. Import verification - All required packages
2. GPU/Device availability
3. Dataset structure and loading
4. Model initialization (all 7 models)
5. Forward pass test (single batch)
6. Backward pass test (gradient flow)
7. Metrics calculation
8. Results saving
9. Memory estimation

Usage:
    python test_pipeline.py
    python test_pipeline.py --quick    # Skip some tests for faster check
    python test_pipeline.py --verbose  # More detailed output
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")


def print_test(test_name):
    print(f"\n{Colors.BOLD}[TEST] {test_name}{Colors.END}")


def print_pass(message=""):
    print(f"  {Colors.GREEN}✓ PASS{Colors.END} {message}")


def print_fail(message=""):
    print(f"  {Colors.RED}✗ FAIL{Colors.END} {message}")


def print_warn(message=""):
    print(f"  {Colors.YELLOW}⚠ WARNING{Colors.END} {message}")


def print_info(message=""):
    print(f"  {Colors.BLUE}ℹ INFO{Colors.END} {message}")


class PipelineTest:
    """Comprehensive test suite for the training pipeline."""
    
    def __init__(self, verbose=False, quick=False):
        self.verbose = verbose
        self.quick = quick
        self.results = {}
        self.models_to_test = [
            'resnet50',
            'densenet121',
            'efficientnet_b4',
            'se_resnet50',
            'vit_b_16',
            'swin_v2_t',
            'hybrid_cnn_vit'
        ]
        
    def test_imports(self):
        """Test 1: Verify all required packages can be imported."""
        print_test("1. Import Verification")
        
        required_packages = [
            ('torch', 'PyTorch'),
            ('torchvision', 'Torchvision'),
            ('numpy', 'NumPy'),
            ('pandas', 'Pandas'),
            ('sklearn', 'Scikit-learn'),
            ('PIL', 'Pillow'),
        ]
        
        optional_packages = [
            ('timm', 'Timm (for advanced models)'),
            ('matplotlib', 'Matplotlib (for plots)'),
            ('seaborn', 'Seaborn (for plots)'),
        ]
        
        all_passed = True
        
        # Required packages
        for package, name in required_packages:
            try:
                __import__(package)
                print_pass(f"{name}")
            except ImportError as e:
                print_fail(f"{name}: {e}")
                all_passed = False
        
        # Optional packages
        for package, name in optional_packages:
            try:
                __import__(package)
                print_pass(f"{name}")
            except ImportError:
                print_warn(f"{name} not installed (optional)")
        
        # Test local imports
        print_info("Testing local module imports...")
        try:
            from preprocessing.preprocessing import get_stratified_kfold, set_seed
            print_pass("preprocessing.preprocessing")
        except ImportError as e:
            print_fail(f"preprocessing.preprocessing: {e}")
            all_passed = False
            
        try:
            from models.models import get_model, get_model_input_size
            print_pass("models.models")
        except ImportError as e:
            print_fail(f"models.models: {e}")
            all_passed = False
            
        try:
            from evaluation.evaluation import MetricsCalculator, ResultsSaver
            print_pass("evaluation.evaluation")
        except ImportError as e:
            print_fail(f"evaluation.evaluation: {e}")
            all_passed = False
        
        self.results['imports'] = all_passed
        return all_passed
    
    def test_device(self):
        """Test 2: Check GPU/device availability."""
        print_test("2. Device Availability")
        
        import torch
        
        if torch.cuda.is_available():
            print_pass(f"CUDA available")
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            print_info(f"CUDA version: {torch.version.cuda}")
            
            # Check GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_info(f"GPU Memory: {total_mem:.2f} GB")
            
            if total_mem < 4:
                print_warn("Low GPU memory. Consider reducing batch_size to 16 or 8")
            
            # Test CUDA operations
            try:
                x = torch.randn(10, 10).cuda()
                y = x @ x.T
                del x, y
                torch.cuda.empty_cache()
                print_pass("CUDA tensor operations working")
            except Exception as e:
                print_fail(f"CUDA operations failed: {e}")
                self.results['device'] = False
                return False
        else:
            print_warn("CUDA not available - will use CPU (MUCH slower)")
            print_info("Training will take significantly longer on CPU")
        
        self.results['device'] = True
        return True
    
    def test_dataset(self):
        """Test 3: Verify dataset structure and loading."""
        print_test("3. Dataset Structure & Loading")
        
        import shutil
        from torchvision import datasets
        
        # Check raw data directories
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set'))
        
        colon_dir = os.path.join(base_dir, 'colon_image_sets')
        lung_dir = os.path.join(base_dir, 'lung_image_sets')
        multiclass_dir = os.path.join(base_dir, 'multiclass')
        
        # Check colon directory
        if os.path.exists(colon_dir):
            colon_classes = [d for d in os.listdir(colon_dir) if os.path.isdir(os.path.join(colon_dir, d))]
            print_pass(f"Colon directory found: {len(colon_classes)} classes")
            if self.verbose:
                for c in colon_classes:
                    count = len(os.listdir(os.path.join(colon_dir, c)))
                    print_info(f"  {c}: {count} images")
        else:
            print_fail(f"Colon directory not found: {colon_dir}")
            self.results['dataset'] = False
            return False
        
        # Check lung directory
        if os.path.exists(lung_dir):
            lung_classes = [d for d in os.listdir(lung_dir) if os.path.isdir(os.path.join(lung_dir, d))]
            print_pass(f"Lung directory found: {len(lung_classes)} classes")
            if self.verbose:
                for c in lung_classes:
                    count = len(os.listdir(os.path.join(lung_dir, c)))
                    print_info(f"  {c}: {count} images")
        else:
            print_fail(f"Lung directory not found: {lung_dir}")
            self.results['dataset'] = False
            return False
        
        # Create/verify multiclass directory
        if not os.path.exists(multiclass_dir):
            print_info("Creating multiclass directory...")
            try:
                os.makedirs(multiclass_dir)
                for folder in os.listdir(colon_dir):
                    src = os.path.join(colon_dir, folder)
                    dst = os.path.join(multiclass_dir, folder)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                for folder in os.listdir(lung_dir):
                    src = os.path.join(lung_dir, folder)
                    dst = os.path.join(multiclass_dir, folder)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                print_pass("Multiclass directory created")
            except Exception as e:
                print_fail(f"Failed to create multiclass directory: {e}")
                self.results['dataset'] = False
                return False
        else:
            print_pass("Multiclass directory exists")
        
        # Test ImageFolder loading
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            dataset = datasets.ImageFolder(multiclass_dir, transform=transform)
            print_pass(f"ImageFolder loading successful")
            print_info(f"Total samples: {len(dataset)}")
            print_info(f"Classes: {dataset.classes}")
            print_info(f"Class to idx: {dataset.class_to_idx}")
            
            # Test loading a sample
            img, label = dataset[0]
            print_pass(f"Sample loading: shape={img.shape}, label={label}")
            
        except Exception as e:
            print_fail(f"ImageFolder loading failed: {e}")
            self.results['dataset'] = False
            return False
        
        # Test stratified k-fold
        try:
            from preprocessing.preprocessing import get_stratified_kfold
            fold_gen = get_stratified_kfold(multiclass_dir, n_splits=3, seed=42)
            fold, train_ds, val_ds, class_names = next(fold_gen)
            print_pass(f"Stratified K-Fold working")
            print_info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        except Exception as e:
            print_fail(f"Stratified K-Fold failed: {e}")
            self.results['dataset'] = False
            return False
        
        self.results['dataset'] = True
        return True
    
    def test_models(self):
        """Test 4: Verify all models can be initialized."""
        print_test("4. Model Initialization")
        
        import torch
        from models.models import get_model, get_model_input_size
        
        all_passed = True
        model_info = {}
        
        for model_name in self.models_to_test:
            try:
                start = time.time()
                model = get_model(model_name, num_classes=5, pretrained=True)
                load_time = time.time() - start
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                input_size = get_model_input_size(model_name)
                
                model_info[model_name] = {
                    'params': total_params,
                    'trainable': trainable_params,
                    'input_size': input_size,
                    'load_time': load_time
                }
                
                print_pass(f"{model_name}")
                if self.verbose:
                    print_info(f"  Total params: {total_params/1e6:.2f}M")
                    print_info(f"  Trainable: {trainable_params/1e6:.2f}M")
                    print_info(f"  Input size: {input_size}x{input_size}")
                    print_info(f"  Load time: {load_time:.2f}s")
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print_fail(f"{model_name}: {e}")
                if self.verbose:
                    traceback.print_exc()
                all_passed = False
        
        self.results['models'] = all_passed
        self.model_info = model_info
        return all_passed
    
    def test_forward_pass(self):
        """Test 5: Verify forward pass works for all models."""
        print_test("5. Forward Pass (Single Batch)")
        
        import torch
        from models.models import get_model, get_model_input_size
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 4  # Small batch for testing
        all_passed = True
        
        for model_name in self.models_to_test:
            try:
                model = get_model(model_name, num_classes=5, pretrained=True)
                model = model.to(device)
                model.eval()
                
                input_size = get_model_input_size(model_name)
                dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
                
                with torch.no_grad():
                    output = model(dummy_input)
                    
                    # Handle InceptionV3 output
                    if hasattr(output, 'logits'):
                        output = output.logits
                
                # Verify output shape
                expected_shape = (batch_size, 5)
                if output.shape == expected_shape:
                    print_pass(f"{model_name}: output shape {output.shape}")
                else:
                    print_fail(f"{model_name}: expected {expected_shape}, got {output.shape}")
                    all_passed = False
                
                del model, dummy_input, output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print_fail(f"{model_name}: {e}")
                if self.verbose:
                    traceback.print_exc()
                all_passed = False
        
        self.results['forward_pass'] = all_passed
        return all_passed
    
    def test_backward_pass(self):
        """Test 6: Verify backward pass and gradient flow."""
        print_test("6. Backward Pass (Gradient Flow)")
        
        import torch
        import torch.nn as nn
        from models.models import get_model, get_model_input_size
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 4
        all_passed = True
        
        # Test subset of models in quick mode
        models = self.models_to_test[:3] if self.quick else self.models_to_test
        
        for model_name in models:
            try:
                model = get_model(model_name, num_classes=5, pretrained=True)
                model = model.to(device)
                model.train()
                
                input_size = get_model_input_size(model_name)
                dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
                dummy_labels = torch.randint(0, 5, (batch_size,)).to(device)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                
                # Forward
                output = model(dummy_input)
                if hasattr(output, 'logits'):
                    output = output.logits
                
                # Backward
                loss = criterion(output, dummy_labels)
                optimizer.zero_grad()
                loss.backward()
                
                # Check gradients
                has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                              for p in model.parameters() if p.requires_grad)
                
                if has_grad:
                    optimizer.step()
                    print_pass(f"{model_name}: loss={loss.item():.4f}, gradients OK")
                else:
                    print_fail(f"{model_name}: no gradients")
                    all_passed = False
                
                del model, dummy_input, dummy_labels, output, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print_fail(f"{model_name}: {e}")
                if self.verbose:
                    traceback.print_exc()
                all_passed = False
        
        if self.quick and len(self.models_to_test) > 3:
            print_info(f"Quick mode: tested {len(models)}/{len(self.models_to_test)} models")
        
        self.results['backward_pass'] = all_passed
        return all_passed
    
    def test_metrics(self):
        """Test 7: Verify metrics calculation."""
        print_test("7. Metrics Calculation")
        
        import numpy as np
        
        try:
            from evaluation.evaluation import MetricsCalculator
            
            # Create dummy data
            np.random.seed(42)
            n_samples = 100
            n_classes = 5
            class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
            
            y_true = np.random.randint(0, n_classes, n_samples)
            y_pred = np.random.randint(0, n_classes, n_samples)
            
            # Create fake probabilities
            y_proba = np.random.rand(n_samples, n_classes)
            y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
            
            # Test metrics computation
            metrics = MetricsCalculator.compute_all_metrics(
                y_true, y_pred, y_proba, class_names
            )
            
            print_pass("MetricsCalculator.compute_all_metrics")
            
            # Verify required metrics exist
            required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            for metric in required_metrics:
                if metric in metrics:
                    print_pass(f"  {metric}: {metrics[metric]:.4f}")
                else:
                    print_fail(f"  {metric} missing")
                    self.results['metrics'] = False
                    return False
            
            # Test AUC if available
            if 'auc_roc_macro' in metrics:
                print_pass(f"  auc_roc_macro: {metrics['auc_roc_macro']:.4f}")
            else:
                print_warn("  auc_roc_macro not computed (may be OK for some cases)")
            
            # Test fold statistics
            fold_metrics = [
                {'accuracy': 0.85, 'f1_macro': 0.83},
                {'accuracy': 0.87, 'f1_macro': 0.85},
                {'accuracy': 0.84, 'f1_macro': 0.82},
            ]
            
            cv_stats = MetricsCalculator.compute_fold_statistics(fold_metrics)
            print_pass("MetricsCalculator.compute_fold_statistics")
            
            if self.verbose:
                print_info(f"  accuracy_mean: {cv_stats.get('accuracy_mean', 'N/A')}")
                print_info(f"  accuracy_std: {cv_stats.get('accuracy_std', 'N/A')}")
            
        except Exception as e:
            print_fail(f"Metrics calculation failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['metrics'] = False
            return False
        
        self.results['metrics'] = True
        return True
    
    def test_results_saving(self):
        """Test 8: Verify results can be saved."""
        print_test("8. Results Saving")
        
        import tempfile
        import json
        
        try:
            from evaluation.evaluation import ResultsSaver
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                saver = ResultsSaver(temp_dir)
                print_pass("ResultsSaver initialized")
                
                # Test saving model results
                fold_metrics = [
                    {'accuracy': 0.85, 'f1_macro': 0.83, 'fold': 1},
                    {'accuracy': 0.87, 'f1_macro': 0.85, 'fold': 2},
                ]
                cv_stats = {'accuracy_mean': 0.86, 'f1_macro_mean': 0.84}
                config = {'epochs': 10, 'batch_size': 32}
                
                saver.save_model_results('test_model', fold_metrics, cv_stats, config)
                
                # Verify files were created
                model_dir = os.path.join(temp_dir, 'test_model')
                if os.path.exists(model_dir):
                    print_pass("Model results directory created")
                    files = os.listdir(model_dir)
                    if self.verbose:
                        for f in files:
                            print_info(f"  Created: {f}")
                else:
                    print_fail("Model results directory not created")
                    self.results['saving'] = False
                    return False
                
                # Test comparison saving
                all_stats = {
                    'model1': {'accuracy_mean': 0.85},
                    'model2': {'accuracy_mean': 0.87}
                }
                saver.save_comparison_results(all_stats, [])
                print_pass("Comparison results saved")
                
        except Exception as e:
            print_fail(f"Results saving failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['saving'] = False
            return False
        
        self.results['saving'] = True
        return True
    
    def test_memory_estimation(self):
        """Test 9: Estimate memory requirements."""
        print_test("9. Memory Estimation")
        
        import torch
        
        if not torch.cuda.is_available():
            print_warn("GPU not available, skipping memory estimation")
            self.results['memory'] = True
            return True
        
        try:
            from models.models import get_model, get_model_input_size
            
            device = 'cuda'
            batch_size = 32  # Default batch size
            
            print_info(f"Estimating memory for batch_size={batch_size}")
            
            memory_requirements = {}
            
            # Test each model
            for model_name in self.models_to_test:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                try:
                    model = get_model(model_name, num_classes=5, pretrained=True)
                    model = model.to(device)
                    model.train()
                    
                    input_size = get_model_input_size(model_name)
                    
                    # Try with default batch size
                    try:
                        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
                        output = model(dummy_input)
                        if hasattr(output, 'logits'):
                            output = output.logits
                        
                        loss = output.sum()
                        loss.backward()
                        
                        peak_memory = torch.cuda.max_memory_allocated() / 1e9
                        memory_requirements[model_name] = {
                            'batch_size': batch_size,
                            'peak_memory_gb': peak_memory,
                            'status': 'OK'
                        }
                        
                        del dummy_input, output, loss
                        
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            memory_requirements[model_name] = {
                                'batch_size': batch_size,
                                'status': 'OOM - reduce batch_size'
                            }
                        else:
                            raise
                    
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    memory_requirements[model_name] = {
                        'status': f'Error: {str(e)}'
                    }
            
            # Print results
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_info(f"Total GPU Memory: {total_gpu_memory:.2f} GB")
            
            all_ok = True
            for model_name, info in memory_requirements.items():
                if info['status'] == 'OK':
                    mem = info['peak_memory_gb']
                    pct = (mem / total_gpu_memory) * 100
                    print_pass(f"{model_name}: {mem:.2f} GB ({pct:.1f}%)")
                else:
                    print_warn(f"{model_name}: {info['status']}")
                    if 'OOM' in info['status']:
                        all_ok = False
            
            if not all_ok:
                print_warn("Consider reducing batch_size with --batch_size 16")
            
            self.results['memory'] = all_ok
            return all_ok
            
        except Exception as e:
            print_fail(f"Memory estimation failed: {e}")
            self.results['memory'] = False
            return False
    
    def test_mini_training(self):
        """Test 10: Run a mini training loop."""
        print_test("10. Mini Training Loop (1 epoch, 2 batches)")
        
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load a small subset of data
            multiclass_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'multiclass'
            ))
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            dataset = datasets.ImageFolder(multiclass_dir, transform=transform)
            
            # Use only 32 samples
            indices = list(range(min(32, len(dataset))))
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=8, shuffle=True)
            
            from models.models import get_model
            
            # Test with smallest model
            model_name = 'resnet50'
            model = get_model(model_name, num_classes=5, pretrained=True)
            model = model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            
            model.train()
            batch_count = 0
            
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_count += 1
                print_info(f"Batch {batch_count}: loss={loss.item():.4f}")
                
                if batch_count >= 2:
                    break
            
            print_pass(f"Mini training completed successfully")
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.results['mini_training'] = True
            return True
            
        except Exception as e:
            print_fail(f"Mini training failed: {e}")
            if self.verbose:
                traceback.print_exc()
            self.results['mini_training'] = False
            return False
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        print_header("PIPELINE PRE-FLIGHT TESTS")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        tests = [
            ('Imports', self.test_imports),
            ('Device', self.test_device),
            ('Dataset', self.test_dataset),
            ('Models', self.test_models),
            ('Forward Pass', self.test_forward_pass),
            ('Backward Pass', self.test_backward_pass),
            ('Metrics', self.test_metrics),
            ('Saving', self.test_results_saving),
            ('Memory', self.test_memory_estimation),
            ('Mini Training', self.test_mini_training),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print_fail(f"Test '{name}' crashed: {e}")
                if self.verbose:
                    traceback.print_exc()
                self.results[name.lower().replace(' ', '_')] = False
        
        elapsed = time.time() - start_time
        
        # Print summary
        print_header("TEST SUMMARY")
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            if result:
                print(f"  {Colors.GREEN}✓{Colors.END} {test_name}")
            else:
                print(f"  {Colors.RED}✗{Colors.END} {test_name}")
        
        print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
        print(f"Time: {elapsed:.2f} seconds")
        
        if passed == total:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED - Pipeline ready to run!{Colors.END}")
            print(f"\nRun the full pipeline with:")
            print(f"  python run_pipeline.py")
            return True
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED - Fix issues before running pipeline{Colors.END}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Test Pipeline Pre-flight Checks')
    parser.add_argument('--quick', action='store_true', 
                        help='Run quick tests (skip some thorough checks)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    tester = PipelineTest(verbose=args.verbose, quick=args.quick)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()