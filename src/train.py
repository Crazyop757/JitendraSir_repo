"""
train.py

- Main training script
- Loads data, model, and runs training with cross-validation
- Logs hyperparameters, optimizer, loss, and environment details
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import platform
import time
import numpy as np
import torchvision
from sklearn.metrics import accuracy_score, f1_score

# Ensure src is in sys.path for imports regardless of working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocessing import get_stratified_kfold, set_seed, get_val_transforms
from models.models import get_model
from evaluation.evaluation import evaluate_predictions, paired_t_test, wilcoxon_test

# Hyperparameters and environment
CONFIG = {
    'model_name': 'resnet50',
    'num_classes': 5,  # Will be auto-detected, but for clarity
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-4,
    'optimizer': 'adam',
    'loss_fn': 'cross_entropy',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Use the parent directory containing all subtype folders as classes
    'data_dir': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'colon_image_sets')),
    'n_splits': 5,
}

# Add the lung subtypes as well by merging both colon and lung subtype folders into a single directory for ImageFolder
import shutil
import glob
def prepare_multiclass_dataset():
    multiclass_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'multiclass'))
    if not os.path.exists(multiclass_dir):
        os.makedirs(multiclass_dir)
        # Copy colon subtypes
        colon_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'colon_image_sets'))
        for folder in os.listdir(colon_dir):
            src = os.path.join(colon_dir, folder)
            dst = os.path.join(multiclass_dir, folder)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
        # Copy lung subtypes
        lung_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'lung_colon_image_set', 'lung_image_sets'))
        for folder in os.listdir(lung_dir):
            src = os.path.join(lung_dir, folder)
            dst = os.path.join(multiclass_dir, folder)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
    return multiclass_dir

CONFIG['data_dir'] = prepare_multiclass_dataset()

def train_one_fold(model, train_loader, val_loader, config, class_weights=None):
    print("  [DEBUG] Initializing loss and optimizer...")
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(config['device']))
    else:
        criterion = nn.CrossEntropyLoss()
    # Add weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    model.to(config['device'])
    print("  [DEBUG] Starting training loop...")
    for epoch in range(config['epochs']):
        print(f"    [DEBUG] Epoch {epoch+1}/{config['epochs']}")
        model.train()
        batch_idx = 0
        for images, labels in train_loader:
            batch_idx += 1
            try:
                images, labels = images.to(config['device']), labels.to(config['device'])
                optimizer.zero_grad()
                outputs = model(images)
                # For InceptionV3, outputs is InceptionOutputs; use outputs.logits for loss
                if hasattr(outputs, 'logits'):
                    loss = criterion(outputs.logits, labels)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"      [DEBUG] Batch {batch_idx}: Loss={loss.item():.4f}")
            except Exception as e:
                print(f"      [ERROR] Exception in training batch {batch_idx}: {e}")
                raise
    # Validation
    print("  [DEBUG] Starting validation...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        batch_idx = 0
        for images, labels in val_loader:
            batch_idx += 1
            try:
                images, labels = images.to(config['device']), labels.to(config['device'])
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            except Exception as e:
                print(f"      [ERROR] Exception in validation batch {batch_idx}: {e}")
                raise
    print("  [DEBUG] Validation complete.")
    return y_true, y_pred

def main():
    set_seed(CONFIG['seed'])
    print("\n==== Experiment Configuration ====")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {CONFIG['device']}")
    print("===============================\n")

    fold_metrics = []
    fold_accs = []
    fold_f1s = []
    start_time = time.time()
    # Detect class names and set num_classes automatically
    from preprocessing.preprocessing import get_stratified_kfold
    first_fold = next(get_stratified_kfold(CONFIG['data_dir'], CONFIG['n_splits'], CONFIG['seed']))
    _, train_dataset0, _, class_names = first_fold
    CONFIG['num_classes'] = len(class_names)
    print(f"[INFO] Detected class names: {class_names}")
    # Count samples per class in the whole dataset
    all_labels = [train_dataset0.dataset.samples[i][1] for i in range(len(train_dataset0.dataset.samples))]
    class_sample_count_total = np.array([all_labels.count(i) for i in range(CONFIG['num_classes'])])
    print(f"[INFO] Total sample count per class: {class_sample_count_total}")
    if any(class_sample_count_total == 0):
        print(f"[WARNING] Some classes have zero samples! Check your dataset structure.")

    # Now run the full CV loop
    for fold, train_dataset, val_dataset, class_names in get_stratified_kfold(CONFIG['data_dir'], CONFIG['n_splits'], CONFIG['seed']):
        print(f"Fold {fold+1}/{CONFIG['n_splits']}")
        try:
            print("  [DEBUG] Preparing class weights...")
            labels = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]
            class_sample_count = np.array([labels.count(i) for i in range(CONFIG['num_classes'])])
            class_weights = torch.tensor(1. / (class_sample_count + 1e-6), dtype=torch.float)
            print(f"  [DEBUG] Class sample count: {class_sample_count}")
            print(f"  [DEBUG] Class weights: {class_weights}")
            # --- Debug: Print sample file paths and labels from train and val sets ---
            print("  [DEBUG] Sample file paths and labels from train set:")
            for idx in train_dataset.indices[:10]:
                path, label = train_dataset.dataset.samples[idx]
                print(f"    Train: {path} | Label: {label}")
            print("  [DEBUG] Sample file paths and labels from val set:")
            for idx in val_dataset.indices[:10]:
                path, label = val_dataset.dataset.samples[idx]
                print(f"    Val: {path} | Label: {label}")
            # --- Debug: Check for overlap between train and val sets ---
            train_paths = set([train_dataset.dataset.samples[i][0] for i in train_dataset.indices])
            val_paths = set([val_dataset.dataset.samples[i][0] for i in val_dataset.indices])
            overlap = train_paths.intersection(val_paths)
            print(f"  [DEBUG] Number of overlapping file paths between train and val: {len(overlap)}")
            if len(overlap) > 0:
                print(f"  [WARNING] Overlapping files detected! Example: {list(overlap)[:5]}")
            print("  [DEBUG] Creating data loaders...")
            # Enable augmentation for training set, validation set uses only val transforms
            from torchvision import transforms
            from preprocessing.preprocessing import get_train_transforms, get_val_transforms
            if CONFIG['model_name'] == 'inception_v3':
                train_transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                val_transform = transforms.Compose([
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                train_dataset.dataset.transform = train_transform
                val_dataset.dataset.transform = val_transform
            else:
                train_dataset.dataset.transform = get_train_transforms()
                val_dataset.dataset.transform = get_val_transforms()
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            print("  [DEBUG] Creating model...")
            model = get_model(CONFIG['model_name'], CONFIG['num_classes'], pretrained=True)
            print("  [DEBUG] Starting fold training...")
            y_true, y_pred = train_one_fold(model, train_loader, val_loader, CONFIG, class_weights=class_weights)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            per_class_f1 = f1_score(y_true, y_pred, average=None)
            print(f"Fold {fold+1} Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
            for idx, cname in enumerate(class_names):
                print(f"  {cname} F1: {per_class_f1[idx]:.4f}")
            evaluate_predictions(y_true, y_pred, class_names)
            fold_metrics.append({'y_true': y_true, 'y_pred': y_pred, 'acc': acc, 'f1': f1, 'per_class_f1': per_class_f1})
            fold_accs.append(acc)
            fold_f1s.append(f1)
        except Exception as e:
            print(f"[ERROR] Exception in fold {fold+1}: {e}")
            raise
    elapsed = time.time() - start_time
    print("\n==== Cross-Validation Results ====")
    print(f"Mean Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"Mean Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Variance (Accuracy): {np.var(fold_accs):.6f}")
    print(f"Variance (Macro F1): {np.var(fold_f1s):.6f}")
    print(f"Total time: {elapsed/60:.2f} min")
    # Statistical significance testing (paired t-test, Wilcoxon) on F1 scores
    if len(fold_f1s) > 1:
        print("\nPaired t-test on F1 scores across folds:")
        paired_t_test(fold_f1s[:-1], fold_f1s[1:])
        print("Wilcoxon test on F1 scores across folds:")
        wilcoxon_test(fold_f1s[:-1], fold_f1s[1:])

if __name__ == "__main__":
    main()
