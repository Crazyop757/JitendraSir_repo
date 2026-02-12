"""
README.md

# Lung and Colon Cancer Histopathological Image Classification

## Project Structure

- `src/models/`: Pretrained and custom model definitions (ResNet, Inception, ViT, etc.)
- `src/preprocessing/`: Data loading, preprocessing, augmentation, and splitting
- `src/evaluation/`: Evaluation metrics, statistical tests, per-class breakdowns
- `src/explainability/`: Grad-CAM, LIME, SHAP integration and case studies
- `src/ablation/`: Ablation study utilities
- `src/train.py`: Main training and cross-validation script

## Key Features
- Explicit preprocessing and augmentation for reproducibility
- Stratified k-fold cross-validation with fold-wise metrics
- Statistical significance testing (paired t-test, Wilcoxon)
- Per-class performance reporting
- Explainability (Grad-CAM, LIME, SHAP)
- Ablation studies for model and preprocessing components
- Support for attention-based, transformer, and hybrid models
- Imbalanced data handling (weighted loss, augmentation)
- Full documentation of hyperparameters, training schedules, and environment

## Usage

1. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn numpy pillow pytorch-grad-cam lime shap
   ```
2. Run training with cross-validation:
   ```bash
   python src/train.py
   ```

## Data
- Place the dataset in `archive/lung_colon_image_set/` as described in the dataset overview.

## Reproducibility
- All random seeds are set for reproducibility.
- Data splits and augmentations are explicitly defined in code.

## Statistical Analysis
- Use `src/evaluation/evaluation.py` for significance testing and per-class metrics.

## Explainability
- Use `src/explainability/explainability.py` for Grad-CAM, LIME, and SHAP analyses.

## Ablation
- Use `src/ablation/ablation.py` to run ablation studies on model and preprocessing components.

## Environment
- Python 3.8+
- CUDA GPU recommended for training
- All dependencies listed above

---

For further details, see code comments and docstrings in each module.
