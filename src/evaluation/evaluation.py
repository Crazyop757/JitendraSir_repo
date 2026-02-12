"""
evaluation.py

- Evaluation utilities for classification
- Computes per-class metrics, confusion matrix, and supports statistical significance testing
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import ttest_rel, wilcoxon

def evaluate_predictions(y_true, y_pred, class_names):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# Statistical significance testing
def paired_t_test(scores1, scores2):
    stat, p = ttest_rel(scores1, scores2)
    print(f"Paired t-test: stat={stat:.4f}, p-value={p:.4g}")
    return stat, p

def wilcoxon_test(scores1, scores2):
    stat, p = wilcoxon(scores1, scores2)
    print(f"Wilcoxon test: stat={stat:.4f}, p-value={p:.4g}")
    return stat, p
