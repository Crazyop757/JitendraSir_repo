"""
evaluation.py

- Comprehensive evaluation utilities for classification
- Computes accuracy, precision, recall, F1-score, AUC-ROC
- Generates confusion matrix and classification reports
- Supports statistical significance testing (t-test, Wilcoxon)
- Saves all results to CSV and JSON files
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_rel, wilcoxon, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Comprehensive metrics calculation and storage"""
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, y_proba=None, class_names=None):
        """
        Compute all classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for AUC calculation)
            class_names: List of class names
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Weighted metrics
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        if class_names:
            metrics['per_class'] = {}
            for i, class_name in enumerate(class_names):
                metrics['per_class'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'support': int(support[i])
                }
        
        # AUC-ROC calculation
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    metrics['auc_roc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    # Multi-class classification (One-vs-Rest)
                    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                    metrics['auc_roc_macro'] = float(roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr'))
                    metrics['auc_roc_weighted'] = float(roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr'))
                    
                    # Per-class AUC
                    if class_names:
                        for i, class_name in enumerate(class_names):
                            try:
                                class_auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                                metrics['per_class'][class_name]['auc'] = float(class_auc)
                            except:
                                metrics['per_class'][class_name]['auc'] = None
            except Exception as e:
                print(f"[WARNING] Could not compute AUC: {e}")
                metrics['auc_roc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    @staticmethod
    def compute_fold_statistics(fold_metrics_list):
        """
        Compute mean and std across folds.
        
        Args:
            fold_metrics_list: List of metrics dictionaries from each fold
            
        Returns:
            Dictionary with mean and std for each metric
        """
        stats = {}
        
        metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                       'precision_weighted', 'recall_weighted', 'f1_weighted',
                       'auc_roc_macro', 'auc_roc_weighted']
        
        for key in metric_keys:
            values = [m.get(key) for m in fold_metrics_list if m.get(key) is not None]
            if values:
                stats[f'{key}_mean'] = float(np.mean(values))
                stats[f'{key}_std'] = float(np.std(values))
                stats[f'{key}_values'] = values
        
        return stats


def evaluate_predictions(y_true, y_pred, class_names, y_proba=None):
    """
    Print detailed evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        y_proba: Prediction probabilities (optional)
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print(f"\nOVERALL METRICS:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f} (macro)")
    print(f"  Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f} (macro)")
    print(f"  F1-Score:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f} (macro)")
    
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes > 2:
                y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
                auc = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                print(f"  AUC-ROC:   {auc:.4f} (macro)")
        except Exception as e:
            print(f"  AUC-ROC:   Could not compute ({e})")
    print("="*60 + "\n")


# Statistical significance testing
def paired_t_test(scores1, scores2, name1="Model1", name2="Model2"):
    """
    Paired t-test for comparing two models on the same folds.
    
    Args:
        scores1: List of scores from model 1 across folds
        scores2: List of scores from model 2 across folds
        name1: Name of first model
        name2: Name of second model
        
    Returns:
        Dictionary with test results
    """
    if len(scores1) != len(scores2):
        print("[WARNING] Unequal number of scores. Using independent t-test instead.")
        stat, p = ttest_ind(scores1, scores2)
        test_type = "independent"
    else:
        stat, p = ttest_rel(scores1, scores2)
        test_type = "paired"
    
    result = {
        'test_type': f'{test_type}_t_test',
        'model1': name1,
        'model2': name2,
        'statistic': float(stat),
        'p_value': float(p),
        'significant_0.05': p < 0.05,
        'significant_0.01': p < 0.01,
        'mean_diff': float(np.mean(scores1) - np.mean(scores2))
    }
    
    print(f"\n{test_type.capitalize()} t-test ({name1} vs {name2}):")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value:   {p:.4g}")
    print(f"  Significant at α=0.05: {p < 0.05}")
    print(f"  Mean difference: {result['mean_diff']:.4f}")
    
    return result


def wilcoxon_test(scores1, scores2, name1="Model1", name2="Model2"):
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        scores1: List of scores from model 1 across folds
        scores2: List of scores from model 2 across folds
        name1: Name of first model
        name2: Name of second model
        
    Returns:
        Dictionary with test results
    """
    try:
        stat, p = wilcoxon(scores1, scores2)
        result = {
            'test_type': 'wilcoxon',
            'model1': name1,
            'model2': name2,
            'statistic': float(stat),
            'p_value': float(p),
            'significant_0.05': p < 0.05,
            'significant_0.01': p < 0.01
        }
        
        print(f"\nWilcoxon test ({name1} vs {name2}):")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value:   {p:.4g}")
        print(f"  Significant at α=0.05: {p < 0.05}")
        
        return result
    except Exception as e:
        print(f"[WARNING] Wilcoxon test failed: {e}")
        return {'test_type': 'wilcoxon', 'error': str(e)}


class ResultsSaver:
    """Utility class for saving results to files"""
    
    def __init__(self, output_dir):
        """
        Initialize results saver.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_model_results(self, model_name, fold_metrics, cv_stats, config):
        """
        Save all results for a single model.
        
        Args:
            model_name: Name of the model
            fold_metrics: List of metrics from each fold
            cv_stats: Cross-validation statistics
            config: Training configuration
        """
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save detailed JSON results
        results = {
            'model_name': model_name,
            'timestamp': self.timestamp,
            'config': config,
            'fold_metrics': fold_metrics,
            'cv_statistics': cv_stats
        }
        
        json_path = os.path.join(model_dir, f'{model_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary CSV
        summary_data = {
            'model': model_name,
            'accuracy_mean': cv_stats.get('accuracy_mean', 0),
            'accuracy_std': cv_stats.get('accuracy_std', 0),
            'precision_mean': cv_stats.get('precision_macro_mean', 0),
            'precision_std': cv_stats.get('precision_macro_std', 0),
            'recall_mean': cv_stats.get('recall_macro_mean', 0),
            'recall_std': cv_stats.get('recall_macro_std', 0),
            'f1_mean': cv_stats.get('f1_macro_mean', 0),
            'f1_std': cv_stats.get('f1_macro_std', 0),
            'auc_mean': cv_stats.get('auc_roc_macro_mean', 0),
            'auc_std': cv_stats.get('auc_roc_macro_std', 0),
        }
        
        csv_path = os.path.join(model_dir, f'{model_name}_summary.csv')
        pd.DataFrame([summary_data]).to_csv(csv_path, index=False)
        
        print(f"[SAVED] Results for {model_name} saved to {model_dir}")
        
        return json_path, csv_path
    
    def save_comparison_results(self, all_models_stats, statistical_tests):
        """
        Save comparison results across all models.
        
        Args:
            all_models_stats: Dictionary of stats for each model
            statistical_tests: List of statistical test results
        """
        # Create comparison DataFrame
        comparison_data = []
        for model_name, stats in all_models_stats.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{stats.get('accuracy_mean', 0):.4f} ± {stats.get('accuracy_std', 0):.4f}",
                'Precision': f"{stats.get('precision_macro_mean', 0):.4f} ± {stats.get('precision_macro_std', 0):.4f}",
                'Recall': f"{stats.get('recall_macro_mean', 0):.4f} ± {stats.get('recall_macro_std', 0):.4f}",
                'F1-Score': f"{stats.get('f1_macro_mean', 0):.4f} ± {stats.get('f1_macro_std', 0):.4f}",
                'AUC-ROC': f"{stats.get('auc_roc_macro_mean', 0):.4f} ± {stats.get('auc_roc_macro_std', 0):.4f}",
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv = os.path.join(self.output_dir, f'model_comparison_{self.timestamp}.csv')
        comparison_df.to_csv(comparison_csv, index=False)
        
        # Save statistical tests
        if statistical_tests:
            tests_csv = os.path.join(self.output_dir, f'statistical_tests_{self.timestamp}.csv')
            tests_df = pd.DataFrame(statistical_tests)
            tests_df.to_csv(tests_csv, index=False)
        
        # Save complete results JSON
        full_results = {
            'timestamp': self.timestamp,
            'models': all_models_stats,
            'statistical_tests': statistical_tests
        }
        
        json_path = os.path.join(self.output_dir, f'full_results_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n[SAVED] Comparison results saved to {self.output_dir}")
        print(f"  - Comparison CSV: {comparison_csv}")
        print(f"  - Full results JSON: {json_path}")
        
        return comparison_csv, json_path
    
    def plot_comparison(self, all_models_stats, metric='f1_macro'):
        """
        Generate comparison plots.
        
        Args:
            all_models_stats: Dictionary of stats for each model
            metric: Metric to plot
        """
        models = list(all_models_stats.keys())
        means = [all_models_stats[m].get(f'{metric}_mean', 0) for m in models]
        stds = [all_models_stats[m].get(f'{metric}_std', 0) for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f'comparison_{metric}_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"[SAVED] Comparison plot saved to {plot_path}")
        
        return plot_path
    
    def plot_confusion_matrix(self, cm, class_names, model_name):
        """
        Plot and save confusion matrix.
        
        Args:
            cm: Confusion matrix array
            class_names: List of class names
            model_name: Name of the model
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plot_path = os.path.join(model_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return plot_path
