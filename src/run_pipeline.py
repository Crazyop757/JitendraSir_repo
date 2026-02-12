"""
run_pipeline.py

Comprehensive Model Training and Evaluation Pipeline
=====================================================

This script runs multiple deep learning models on the lung/colon cancer dataset,
performs k-fold cross-validation, computes all metrics, and saves results.

Models included:
- ResNet50
- DenseNet121
- EfficientNet-B4
- SE-ResNet50 (Squeeze-and-Excitation)
- Vision Transformer (ViT-B/16)
- Swin Transformer V2
- Hybrid CNN-ViT

Metrics computed:
- Accuracy
- Precision (macro & per-class)
- Recall (macro & per-class)
- F1-Score (macro & per-class)
- AUC-ROC (macro & per-class)

Statistical tests:
- Paired t-test between all model pairs
- Wilcoxon signed-rank test

All results are saved to the 'results/' directory.

Usage:
    python run_pipeline.py
    python run_pipeline.py --epochs 15 --batch_size 16
    python run_pipeline.py --models resnet50 densenet121 vit_b_16
"""

import os
import sys
import argparse
import json
import time
import warnings
from datetime import datetime
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import platform
import torchvision
from torchvision import transforms

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing.preprocessing import (
    get_stratified_kfold, set_seed, get_train_transforms, get_val_transforms
)
from models.models import get_model, get_model_input_size
from evaluation.evaluation import (
    MetricsCalculator, ResultsSaver, evaluate_predictions,
    paired_t_test, wilcoxon_test
)

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
DEFAULT_CONFIG = {
    'models': [
        'resnet50',
        'densenet121', 
        'efficientnet_b4',
        'se_resnet50',
        'vit_b_16',
        'swin_v2_t',
        'hybrid_cnn_vit'
    ],
    'num_classes': 5,  # Auto-detected
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    'seed': 42,
    'n_splits': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': None,  # Set dynamically
    'results_dir': None,  # Set dynamically
    'save_models': True,
    'early_stopping_patience': 3,
}


def get_transforms_for_model(model_name, is_train=True):
    """Get appropriate transforms based on model input size requirements."""
    input_size = get_model_input_size(model_name)
    
    if is_train:
        if input_size == 299:  # Inception
            return transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif input_size == 380:  # EfficientNet-B4
            return transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:  # Default 224
            return get_train_transforms()
    else:
        if input_size == 299:
            return transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif input_size == 380:
            return transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            return get_val_transforms()


def prepare_multiclass_dataset():
    """Prepare the multiclass dataset by merging colon and lung image sets."""
    import shutil
    
    multiclass_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'multiclass'
    ))
    
    if not os.path.exists(multiclass_dir):
        os.makedirs(multiclass_dir)
        
        # Copy colon subtypes
        colon_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'colon_image_sets'
        ))
        for folder in os.listdir(colon_dir):
            src = os.path.join(colon_dir, folder)
            dst = os.path.join(multiclass_dir, folder)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
        
        # Copy lung subtypes
        lung_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'lung_image_sets'
        ))
        for folder in os.listdir(lung_dir):
            src = os.path.join(lung_dir, folder)
            dst = os.path.join(multiclass_dir, folder)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                
        print(f"[INFO] Created multiclass dataset at {multiclass_dir}")
    
    return multiclass_dir


def train_one_epoch(model, train_loader, criterion, optimizer, device, model_name):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle InceptionV3 auxiliary outputs
        if hasattr(outputs, 'logits'):
            loss = criterion(outputs.logits, labels)
            outputs = outputs.logits
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, num_classes):
    """Validate the model and return predictions with probabilities."""
    model.eval()
    running_loss = 0.0
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    
    return (np.array(all_labels), np.array(all_preds), 
            np.array(all_probs), val_loss)


def train_model_with_cv(model_name, config, data_dir, results_saver):
    """
    Train a single model with k-fold cross-validation.
    
    Returns:
        fold_metrics: List of metrics for each fold
        cv_stats: Cross-validation statistics
    """
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL: {model_name.upper()}")
    print(f"{'='*70}")
    
    fold_metrics = []
    model_start_time = time.time()
    
    # Get class information from first fold
    first_fold = next(get_stratified_kfold(data_dir, config['n_splits'], config['seed']))
    _, train_dataset0, _, class_names = first_fold
    config['num_classes'] = len(class_names)
    print(f"[INFO] Classes: {class_names}")
    
    # Compute class weights for imbalanced data
    all_labels = [train_dataset0.dataset.samples[i][1] for i in range(len(train_dataset0.dataset.samples))]
    class_counts = np.array([all_labels.count(i) for i in range(config['num_classes'])])
    class_weights = torch.tensor(1. / (class_counts + 1e-6), dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * config['num_classes']  # Normalize
    
    # Run k-fold cross-validation
    for fold, train_dataset, val_dataset, _ in get_stratified_kfold(data_dir, config['n_splits'], config['seed']):
        print(f"\n--- Fold {fold + 1}/{config['n_splits']} ---")
        fold_start_time = time.time()
        
        try:
            # Apply model-specific transforms
            train_transform = get_transforms_for_model(model_name, is_train=True)
            val_transform = get_transforms_for_model(model_name, is_train=False)
            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
                pin_memory=True if config['device'] == 'cuda' else False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False,
                num_workers=0,
                pin_memory=True if config['device'] == 'cuda' else False
            )
            
            # Initialize model
            model = get_model(model_name, config['num_classes'], pretrained=True)
            model = model.to(config['device'])
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(config['device']))
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config['lr'], 
                weight_decay=config['weight_decay']
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=2, verbose=True
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config['epochs']):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, config['device'], model_name
                )
                
                y_true, y_pred, y_proba, val_loss = validate(
                    model, val_loader, criterion, config['device'], config['num_classes']
                )
                
                val_acc = 100. * np.mean(y_true == y_pred)
                
                print(f"  Epoch {epoch+1}/{config['epochs']} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= config['early_stopping_patience']:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model and get final predictions
            model.load_state_dict(best_model_state)
            y_true, y_pred, y_proba, _ = validate(
                model, val_loader, criterion, config['device'], config['num_classes']
            )
            
            # Compute all metrics for this fold
            metrics = MetricsCalculator.compute_all_metrics(
                y_true, y_pred, y_proba, class_names
            )
            metrics['fold'] = fold + 1
            fold_metrics.append(metrics)
            
            # Print fold results
            print(f"\n  Fold {fold + 1} Results:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    Recall:    {metrics['recall_macro']:.4f}")
            print(f"    F1-Score:  {metrics['f1_macro']:.4f}")
            if 'auc_roc_macro' in metrics:
                print(f"    AUC-ROC:   {metrics['auc_roc_macro']:.4f}")
            
            fold_time = time.time() - fold_start_time
            print(f"  Fold time: {fold_time/60:.2f} min")
            
            # Save model checkpoint
            if config['save_models']:
                model_dir = os.path.join(config['results_dir'], model_name)
                os.makedirs(model_dir, exist_ok=True)
                checkpoint_path = os.path.join(model_dir, f'{model_name}_fold{fold+1}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'fold': fold + 1,
                    'metrics': metrics
                }, checkpoint_path)
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"[ERROR] Fold {fold + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute cross-validation statistics
    cv_stats = MetricsCalculator.compute_fold_statistics(fold_metrics)
    
    # Print CV summary
    model_time = time.time() - model_start_time
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} - Cross-Validation Summary")
    print(f"{'='*50}")
    print(f"  Accuracy:  {cv_stats.get('accuracy_mean', 0):.4f} ± {cv_stats.get('accuracy_std', 0):.4f}")
    print(f"  Precision: {cv_stats.get('precision_macro_mean', 0):.4f} ± {cv_stats.get('precision_macro_std', 0):.4f}")
    print(f"  Recall:    {cv_stats.get('recall_macro_mean', 0):.4f} ± {cv_stats.get('recall_macro_std', 0):.4f}")
    print(f"  F1-Score:  {cv_stats.get('f1_macro_mean', 0):.4f} ± {cv_stats.get('f1_macro_std', 0):.4f}")
    print(f"  AUC-ROC:   {cv_stats.get('auc_roc_macro_mean', 0):.4f} ± {cv_stats.get('auc_roc_macro_std', 0):.4f}")
    print(f"  Total time: {model_time/60:.2f} min")
    
    # Save model results
    results_saver.save_model_results(
        model_name, fold_metrics, cv_stats, 
        {k: v for k, v in config.items() if k not in ['device']}
    )
    
    return fold_metrics, cv_stats, class_names


def run_statistical_tests(all_results, metric='f1_macro'):
    """
    Run statistical tests between all pairs of models.
    
    Args:
        all_results: Dictionary mapping model names to their fold metrics
        metric: Metric to use for comparison
        
    Returns:
        List of test results
    """
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")
    
    test_results = []
    model_names = list(all_results.keys())
    
    for model1, model2 in combinations(model_names, 2):
        scores1 = all_results[model1]['cv_stats'].get(f'{metric}_values', [])
        scores2 = all_results[model2]['cv_stats'].get(f'{metric}_values', [])
        
        if len(scores1) >= 2 and len(scores2) >= 2:
            # Paired t-test
            t_result = paired_t_test(scores1, scores2, model1, model2)
            test_results.append(t_result)
            
            # Wilcoxon test (only if same length)
            if len(scores1) == len(scores2):
                try:
                    w_result = wilcoxon_test(scores1, scores2, model1, model2)
                    test_results.append(w_result)
                except:
                    pass
    
    return test_results


def print_final_summary(all_results):
    """Print a formatted summary table of all results."""
    print(f"\n{'='*90}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*90}")
    
    # Create summary table
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    row_format = "{:<20} {:<18} {:<18} {:<18} {:<18} {:<18}"
    
    print(row_format.format(*headers))
    print("-" * 90)
    
    for model_name, data in all_results.items():
        stats = data['cv_stats']
        row = [
            model_name,
            f"{stats.get('accuracy_mean', 0):.4f}±{stats.get('accuracy_std', 0):.4f}",
            f"{stats.get('precision_macro_mean', 0):.4f}±{stats.get('precision_macro_std', 0):.4f}",
            f"{stats.get('recall_macro_mean', 0):.4f}±{stats.get('recall_macro_std', 0):.4f}",
            f"{stats.get('f1_macro_mean', 0):.4f}±{stats.get('f1_macro_std', 0):.4f}",
            f"{stats.get('auc_roc_macro_mean', 0):.4f}±{stats.get('auc_roc_macro_std', 0):.4f}",
        ]
        print(row_format.format(*row))
    
    print(f"{'='*90}")
    
    # Find best model
    best_model = max(all_results.items(), 
                     key=lambda x: x[1]['cv_stats'].get('f1_macro_mean', 0))
    print(f"\nBest Model (by F1-Score): {best_model[0]} "
          f"(F1: {best_model[1]['cv_stats'].get('f1_macro_mean', 0):.4f})")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Multi-Model Training Pipeline')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_save_models', action='store_true',
                        help='Do not save model checkpoints')
    
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    if args.models:
        config['models'] = args.models
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr'] = args.lr
    config['n_splits'] = args.n_splits
    config['seed'] = args.seed
    config['save_models'] = not args.no_save_models
    
    # Set random seed
    set_seed(config['seed'])
    
    # Prepare directories
    config['data_dir'] = prepare_multiclass_dataset()
    config['results_dir'] = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results', 
        datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("MULTI-MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Models:      {config['models']}")
    print(f"  Epochs:      {config['epochs']}")
    print(f"  Batch Size:  {config['batch_size']}")
    print(f"  LR:          {config['lr']}")
    print(f"  CV Folds:    {config['n_splits']}")
    print(f"  Device:      {config['device']}")
    print(f"  Data Dir:    {config['data_dir']}")
    print(f"  Results Dir: {config['results_dir']}")
    print(f"\nSystem Info:")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  Torchvision: {torchvision.__version__}")
    print(f"  Platform:    {platform.platform()}")
    print(f"  CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Initialize results saver
    results_saver = ResultsSaver(config['results_dir'])
    
    # Store all results
    all_results = {}
    class_names = None
    
    # Train each model
    pipeline_start_time = time.time()
    
    for model_name in config['models']:
        try:
            fold_metrics, cv_stats, class_names = train_model_with_cv(
                model_name, config, config['data_dir'], results_saver
            )
            all_results[model_name] = {
                'fold_metrics': fold_metrics,
                'cv_stats': cv_stats
            }
        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Run statistical tests
    if len(all_results) >= 2:
        statistical_tests = run_statistical_tests(all_results)
    else:
        statistical_tests = []
    
    # Save comparison results
    all_cv_stats = {name: data['cv_stats'] for name, data in all_results.items()}
    results_saver.save_comparison_results(all_cv_stats, statistical_tests)
    
    # Generate comparison plots
    try:
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'auc_roc_macro']:
            results_saver.plot_comparison(all_cv_stats, metric)
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
    
    # Print final summary
    print_final_summary(all_results)
    
    # Save config
    config_path = os.path.join(config['results_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump({k: v for k, v in config.items() if k != 'device'}, f, indent=2)
    
    pipeline_time = time.time() - pipeline_start_time
    print(f"\nTotal Pipeline Time: {pipeline_time/60:.2f} min ({pipeline_time/3600:.2f} hours)")
    print(f"Results saved to: {config['results_dir']}")
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
